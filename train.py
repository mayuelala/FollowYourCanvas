import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.data.dataset import VideoDataset
from animatediff.models.unet import UNet3DConditionModel
from animatediff.utils.util import zero_rank_print, save_list_to_txt

import numpy as np
import copy

from animatediff.utils.video_mask import video_mask, get_anchor_target

from tensorboardX import SummaryWriter


def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        print(f'num_gpus-{num_gpus}-rank-{rank}-{os.environ["WORLD_SIZE"]}-{local_rank}')
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(days=10), **kwargs)  # yjk

        
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank

class LaionImageLoader:
    def __init__(self, dataset, batch_size, distributed_sampler, num_workers=16):
        self.loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    sampler=distributed_sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                )
        
        self.iter = iter(self.loader)

    def getnext(self):
        try:
            sample = next(self.iter)
        except:
            self.iter = iter(self.loader)
            sample = next(self.iter)
        return sample

def main(
    name: str,
    use_wandb: bool,
    output_dir: str,
    pretrained_model_path: str,
    train_data: Dict,
    launcher: str='pytorch',
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    motion_pretrained_model_path: str = "",
    unet_additional_kwargs: Dict = {},
    noise_scheduler_kwargs = None,
    max_train_steps: int = 100,
    learning_rate: float = 3e-5,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",
    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_steps: int = -1,
    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,
    global_seed: int = 42,
    is_debug: bool = False,
    dataset_name: str='panda-70M',
    use_temporal_multi_scale_training: bool=False,
    use_ip_plus_cross_attention: bool=False,
    image_pretrained_model_path: str="",
    ip_pretrained_model_path: str="",
    cfg_random_null_image: bool = True,
    cfg_random_null_text_ratio_2: float = 0.2,
    use_fps_condition: bool=False,
    use_spatial_temporal_separate_lr: bool=False,
    use_outpaint=False,
    evaluation=False,
    load_ckpt_full=False,
    separate_lr_ratio=[1, 0.2],
    anchor_target_sampling=None,
    ip_plus_condition = 'image',
    image_encoder_name = 'CLIP',
):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher)
    
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = local_rank == 0  # yjk

    if global_seed < 0:
        global_seed = random.randint(1, 1000000)

    seed = global_seed + global_rank

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Logging folder
    if is_debug:
        folder_name = "debug"
    elif evaluation:
        folder_name = 'evaluation'
    else:
        folder_name = name + datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/tb", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    if is_main_process and (not is_debug) and use_wandb:
        writer = SummaryWriter(f'{output_dir}/tb')
        
    image_encoder=None
    if use_ip_plus_cross_attention:
        if image_encoder_name == 'SAM':
            from segment_anything import SamPredictor, sam_model_registry
            image_encoder = sam_model_registry["vit_b"](checkpoint=image_pretrained_model_path)
            image_encoder.requires_grad_(False)
            image_encoder.to(local_rank)
            SAMpredictor = SamPredictor(image_encoder)
            SAMProcessor = SAMpredictor.transform
        else:
            raise ValueError
        print(f'load image encoder: {image_pretrained_model_path}')

    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")

    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path, subfolder="unet", 
        unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
    )
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    noise_scheduler_train = DDPMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    if motion_pretrained_model_path!="":
        logging.info(f"from motion pretreained checkpoint: {motion_pretrained_model_path}")
    
        # motion model keys: 'epoch', 'global_step', 'state_dict'
        motion_pretrained_model_path = torch.load(motion_pretrained_model_path, map_location="cpu", weights_only=False)

        if "global_step" in motion_pretrained_model_path: zero_rank_print(f"global_step: {motion_pretrained_model_path['global_step']}")
        state_dict = motion_pretrained_model_path["state_dict"] if "state_dict" in motion_pretrained_model_path else motion_pretrained_model_path
        new_state_dict = {}
        
        for k, v in state_dict.items():
            if use_outpaint:
                if not load_ckpt_full:
                    if 'conv_in.weight' in k:
                        out_channel, in_channel, h, w = v.shape
                        outpaint_channel = torch.zeros(out_channel, 5, h, w)
                        v = torch.cat((v, outpaint_channel), dim=1)
            
            
            enc_len = unet_additional_kwargs['motion_module_kwargs']['temporal_position_encoding_max_len']
            if 'pos_encoder.pe' in k:
                t, f, d = v.shape
                if enc_len != f:
                    if is_main_process:
                        print(f'Interpolate position encoding of {k} from ({t}, {f}, {d}) to ({t}, {enc_len}, {d})!')
                    v = v.unsqueeze(0)
                    v = torch.nn.functional.interpolate(v, size=(enc_len, d), mode='bicubic', align_corners=False)
                    v = v.squeeze(0)

            new_state_dict[k.replace('module.', '')] = v
            
        m, u = unet.load_state_dict(new_state_dict, strict=False)
        logging.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        logging.info(f"missing keys: {m}, \n unexpected keys: {u}")

    
    if use_ip_plus_cross_attention and ip_pretrained_model_path!="":
        logging.info(f"from ip_pretrained_model_path checkpoint: {ip_pretrained_model_path}")
        state_dict = torch.load(ip_pretrained_model_path, map_location="cpu", weights_only=False)
        
        model_state_dict = unet.state_dict()
        image_proj_keys = []
        for k, v in state_dict['image_proj'].items():
            image_proj_keys.append(k)
            
        model_replace_keys = []
        for k,v in model_state_dict.items():
            if 'image_proj_model' in k:
                model_replace_keys.append(k)
        
        for k1, k2 in zip(model_replace_keys, image_proj_keys):
            print (f'replace {k1} with params of {k2}')
            if model_state_dict[k1].shape == state_dict['image_proj'][k2].shape:
                model_state_dict[k1] = state_dict['image_proj'][k2]
            else:
                print(f'IP adapter model state dict {k1} and {k2} shape mismatch.')
        
        ip_keys = []
        for k, v in state_dict['ip_adapter'].items():
            print(k, v.shape)
            ip_keys.append(k)
        
        model_replace_keys = []
        for k in model_state_dict:
            if '_ip' in k:
                print(k, model_state_dict[k].shape)
                model_replace_keys.append(k)
                
        
        for k1, k2 in zip(model_replace_keys, ip_keys):
            print (f'replace {k1} with params of {k2}')
            if model_state_dict[k1].shape == state_dict['ip_adapter'][k2].shape:
                model_state_dict[k1] = state_dict['ip_adapter'][k2]
            else:
                print(f'IP adapter model state dict {k1} and {k2} shape mismatch.')
 

        m, u = unet.load_state_dict(model_state_dict, strict=False)
        logging.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # Set unet trainable parameters

    
    if use_spatial_temporal_separate_lr:
        # here
        unet.requires_grad_(True)
        trainable_params_temporal = []
        trainable_params_spatial = []
        trainable_params_adapter = []
        trainable_params_temporal_names = []
        trainable_params_spatial_names = []
        trainable_params_adapter_names = []

        for name, param in unet.named_parameters():
            find=False
            find_adapter=False
            name_head = name.split('.')[0]

            for trainable_module_name in trainable_modules:
                if trainable_module_name in name:
                    find = True
                    break
            
            for trainable_module_name in trainable_modules[:3]:
                if trainable_module_name in name:
                    find_adapter = True
                    break

            if find: 
                if find_adapter:
                    trainable_params_adapter.append(param)
                    trainable_params_adapter_names.append(name)
                else:
                    trainable_params_temporal.append(param)
                    trainable_params_temporal_names.append(name)
            else: 
                trainable_params_spatial.append(param)
                trainable_params_spatial_names.append(name)
        
        if is_main_process:
            print (f'trainable_params_temporal-{len(trainable_params_temporal)}-trainable_params_spatial-{len(trainable_params_spatial)}')   
            save_list_to_txt(trainable_params_temporal_names, f"{output_dir}/full_lr_params.txt")
            save_list_to_txt(trainable_params_adapter_names , f"{output_dir}/{separate_lr_ratio[0]}_lr_params.txt")
            save_list_to_txt(trainable_params_spatial_names, f"{output_dir}/{separate_lr_ratio[1]}_lr_params.txt")
            print(f'saved trainable parameteres to {output_dir}')
        param_group = [
        {'params': trainable_params_temporal, 'lr': learning_rate},
        {'params': trainable_params_adapter, 'lr':  separate_lr_ratio[0]*learning_rate},
        {'params': trainable_params_spatial, 'lr': separate_lr_ratio[1]*learning_rate},
        ]
        optimizer = torch.optim.AdamW(
            param_group,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )
        

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)
        
    
    # Get the training dataset
    logging.info("***** Loading Data *****")
    
    train_dataset = VideoDataset(**train_data, local_rank=local_rank, SAMProcessor=SAMProcessor if image_encoder_name == 'SAM' else None)
        
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


    train_list = ["video"]
        
    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

        
    # DDP warpper
    unet.to(local_rank)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None
    
    step_ckpt_num = 7
    step_ckpt_queue = []
    epoch_ckpt_num = 7
    epoch_ckpt_queue = []
        
    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
                
        unet.train()
        
        for step, batch in enumerate(train_dataloader):
            video_loss = 0
            
            if use_temporal_multi_scale_training:
                if global_step % 100 == 0:
                    torch.cuda.empty_cache()

                skip_step_num = 4
                scale_step_range = 8
                frame_list = list(range(1, train_data.sample_n_frames+1))
                frame_len = train_data.sample_n_frames if global_step % scale_step_range < skip_step_num else random.choice(frame_list)
                frame_len_tensor = torch.cuda.LongTensor([int (frame_len)])
                torch.distributed.broadcast(frame_len_tensor, src=0)
                frame_len = frame_len_tensor.item()
                batch['pixel_values'] = batch['pixel_values'][:, :frame_len, :, :, :]
                
            video_batch = copy.deepcopy(batch)
            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['ori_text']]
                batch['text_2'] = [name if random.random() > cfg_random_null_text_ratio_2 else "" for name in batch['ori_text']]
            
            video_batch_size = batch['pixel_values'].shape[0]
            
            ### >>>> Training >>>> ###
            pixel_values = batch["pixel_values"].to(local_rank)
            if use_outpaint:
                if anchor_target_sampling is not None or (batch['relative_position'] != 0).any().item():
                    if (batch['relative_position'] != 0).any().item():
                        anchor_pixels_values, target_pixels_values, mask, relative_position = batch['anchor_pixels_values'].to(local_rank), batch['target_pixels_values'].to(local_rank), batch['mask'].to(local_rank), batch['relative_position'].to(local_rank)
                    else:
                        anchor_pixels_values, target_pixels_values, mask, relative_position = get_anchor_target(pixel_values, anchor_target_sampling)
                    pixel_values = target_pixels_values
                else:
                    mask = video_mask(pixel_values)
                pixel_values_masked = pixel_values.clone() * (mask < 0.5)
            # ============================================================
            # vae encoding
            video_length = pixel_values.shape[1]
            with torch.no_grad(), torch.autocast("cuda"):
                decode_chunk_size = 8
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                frames = []
                for i in range(0, pixel_values.shape[0], decode_chunk_size):
                    num_frames_in = pixel_values[i : i + decode_chunk_size].shape[0]
                    frame = vae.encode(pixel_values[i : i + decode_chunk_size], num_frames_in).latent_dist.sample()
                    frames.append(frame)

                latents = torch.cat(frames)
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)

                latents = latents * 0.18215

                if use_outpaint:
                    decode_chunk_size = 8
                    pixel_values_masked = rearrange(pixel_values_masked, "b f c h w -> (b f) c h w")
                    frames = []
                    for i in range(0, pixel_values_masked.shape[0], decode_chunk_size):
                        num_frames_in = pixel_values_masked[i : i + decode_chunk_size].shape[0]
                        frame = vae.encode(pixel_values_masked[i : i + decode_chunk_size], num_frames_in).latent_dist.sample()
                        frames.append(frame)

                    latents_masked = torch.cat(frames)
                    latents_masked = rearrange(latents_masked, "(b f) c h w -> b c f h w", f=video_length)
                    latents_masked = latents_masked * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler_train.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler_train.add_noise(latents, noise, timesteps)

            # ============================================================
            if use_outpaint:
                mask = mask.transpose(2, 1)
                mask = torch.nn.functional.interpolate(mask, size=(mask.shape[2], noisy_latents.shape[-2], noisy_latents.shape[-1]))
                
                noisy_latents = torch.cat((noisy_latents, mask, latents_masked), dim=1)
            # ============================================================
            
                
            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            # Get the target for loss depending on the prediction type
            if noise_scheduler_train.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler_train.config.prediction_type == "v_prediction":
                target = noise_scheduler_train.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler_train.config.prediction_type}")
                    
            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                image_embeds = None
                
                if use_ip_plus_cross_attention:
                    assert anchor_pixels_values.shape[0] == 1, 'Batch size must be one'

                    with torch.no_grad():
                        if image_encoder_name == 'SAM':
                            if ip_plus_condition != 'video':
                                assert ValueError
                            # HWC, unit8
                            if (batch['image_tensors_sam'] != 0).any().item():
                                image_tensors = batch['image_tensors_sam'].squeeze(dim=0).to(local_rank)
                            else:
                                image_array = np.uint8(((anchor_pixels_values+1.0)/2.0*255).squeeze(dim=0).cpu().numpy().transpose(0,2,3,1))
                                image_tensors = []
                                for image in image_array:
                                    # resize the long side to 1024
                                    image_tensors.append(torch.as_tensor(SAMProcessor.apply_image(image), device=anchor_pixels_values.device).permute(2, 0, 1).contiguous())
                                image_tensors = torch.stack(image_tensors)
                                # pad the short side to 1024 and get features

                            batch_size = 8
                            assert image_tensors.shape[0]%batch_size == 0
                            image_embeds = []
                            uncond_image_embeds = []
                            for i in range(int(image_tensors.shape[0]/batch_size)):
                                SAMpredictor.set_torch_image(image_tensors[i*batch_size: (i+1)*batch_size], image_tensors[0].shape[:2])
                                image_embeds.append(rearrange(SAMpredictor.get_image_embedding(), "f c h w-> f (h w) c"))
                                uncond_image_embeds.append(torch.zeros_like(image_embeds[-1]))
                            image_embeds = torch.cat(image_embeds, dim=0).unsqueeze(0)
                            uncond_image_embeds = torch.cat(uncond_image_embeds, dim=0).unsqueeze(0)
                            
                    image_embeds_ = []
                    if cfg_random_null_image:
                        for image_embed, uncond_image_embed in zip(image_embeds, uncond_image_embeds):
                            if random.random() > cfg_random_null_text_ratio:
                                image_embeds_.append(image_embed)
                            else:
                                image_embeds_.append(uncond_image_embed)
                                 
                    image_embeds = torch.stack(image_embeds_)
                if use_fps_condition:
                    fps_tensor = batch["fps"]
                else:
                    fps_tensor=None

                model_pred = unet(noisy_latents, timesteps, 
                                encoder_hidden_states, 
                                use_ip_plus_cross_attention = use_ip_plus_cross_attention,
                                reference_images_clip_feat=image_embeds,
                                use_fps_condition=use_fps_condition,
                                fps_tensor=fps_tensor,
                                relative_position_tensor = relative_position if unet_additional_kwargs.use_relative_postions else None,
                                ).sample
                
                video_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = video_loss
                            
                optimizer.zero_grad()

                # Backpropagate
                if mixed_precision_training:
                    scaler.scale(loss).backward()
                    """ >>> gradient clipping >>> """
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    """ <<< gradient clipping <<< """
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    """ >>> gradient clipping >>> """
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    """ <<< gradient clipping <<< """
                    optimizer.step()

                lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb and global_step%10==0:
                try:
                    writer.add_scalar('train_loss', loss.item(), global_step)
                    writer.add_scalar('train_video_loss', video_loss.item(), global_step)
                except:
                    print('Not enough space to write loss !!!')
                
            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": unet.state_dict(),
                }

                if step == len(train_dataloader) - 1:
                    try:
                        new_path = os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt")
                        torch.save(state_dict, new_path)
                        epoch_ckpt_queue.append(new_path)
                    except:
                        print('Not enough space to save model !!!')

                    if len(epoch_ckpt_queue) > epoch_ckpt_num:
                            old_path = epoch_ckpt_queue.pop(0)
                            if os.path.exists(old_path):
                                os.remove(old_path)
                else:
                    try:
                        new_path = os.path.join(save_path, f"checkpoint-{global_step}.ckpt")
                        torch.save(state_dict, new_path)
                        step_ckpt_queue.append(new_path)
                    except:
                        print('Not enough space to save model !!!')
                    
                    if len(step_ckpt_queue) > step_ckpt_num:
                        old_path = step_ckpt_queue.pop(0)
                        try:
                            if os.path.exists(old_path):
                                os.remove(old_path)
                        except:
                            print('')
                
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step_loss_video": video_loss.detach().item()}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, help="is used for pytorch ddp mode", default=0)
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
