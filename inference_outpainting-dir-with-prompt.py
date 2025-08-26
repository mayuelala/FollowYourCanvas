import os
import math
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
from safetensors import safe_open
from typing import Dict, Optional, Tuple
from decord import VideoReader
import torchvision.transforms as transforms

import torch
from PIL import Image
import pdb
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation_inference import AnimationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from animatediff.utils.util import save_videos_grid, zero_rank_print, save_list_to_txt

import numpy as np
from accelerate.utils import set_seed
import copy

import PIL
from matplotlib import pyplot as plt
from animatediff.utils.video_mask import video_mask, get_anchor_target
import cv2

from tensorboardX import SummaryWriter

def get_canvas_size(input_size, target_size, min_overlap, window_size):
    """Get canvas size and round number for outpainting. Currently only surppot 1 or two rounds."""
    use_multi_diff = True
    if target_size[0] <= window_size[0] and target_size[1] <= window_size[1]:
        use_multi_diff = False
    if target_size[0] < input_size[0] or target_size[1] < input_size[1]:
        print('target size is smaller than input size!')
        return None, use_multi_diff
    if target_size[0] < input_size[0] + window_size[0] - min_overlap[0] and target_size[1] < input_size[1] + window_size[1] - min_overlap[1]:
        print('outpaint one time.')
        return [target_size], use_multi_diff
    num_round_h = int(np.ceil(((target_size[0] - input_size[0])/2)/(window_size[0] - min_overlap[0])))
    num_round_w = int(np.ceil(((target_size[1] - input_size[1])/2)/(window_size[1] - min_overlap[1])))
    num_round = max(num_round_h, num_round_w)
    canvas_size = []
    for i in range(num_round-1):
        canvas_size_i = [input_size[0] + (i+1) * int((target_size[0] - input_size[0])/num_round), input_size[1] + (i+1) * int((target_size[1] - input_size[1])/num_round)]
        canvas_size_i = [int(canvas_size_i[0]//8*8), int(canvas_size_i[1]//8*8)]
        canvas_size.append(canvas_size_i)
    canvas_size.append(target_size)
    return canvas_size, use_multi_diff


def get_prompt(frame, lmm_tokenizer, lmm_model):
    path = 'infer/temp.jpg'
    frame = cv2.cvtColor((frame.numpy().transpose(1,2,0)+1)/2*255, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, frame)

    lmm_prompt = "Describe the foreground and possible background of the image content in two sentences. Answer starts with 'The image shows'."

    query = lmm_tokenizer.from_list_format([{'image': path},{'text': lmm_prompt}])
    prompt, _ = lmm_model.chat(lmm_tokenizer, query=query, history=None)
    prompt = prompt.replace('The image shows', '').strip()
    print(f'=> Get new prompt: {prompt}')
    
    return prompt
    
def main(
    name: str,
    output_dir: str,
    pretrained_model_path: str,
    validation_data: Dict,
    motion_pretrained_model_path: str = "",
    unet_additional_kwargs: Dict = {},
    noise_scheduler_kwargs = None,
    enable_xformers_memory_efficient_attention: bool = True,
    global_seed: int = 42,
    use_ip_plus_cross_attention: bool=False,
    image_pretrained_model_path: str="",
    use_fps_condition: bool=False,
    use_outpaint=False,
    anchor_target_sampling=None,
    ip_plus_condition = 'image',
    image_encoder_name = 'CLIP',
    target_size = [1080, 1920],
    min_overlap = [256, 64],
    lmm_path = '',
    start_index = 0,
    validate_overlap = None,
    video_dir = None,
    prompts_input = None,
    negative_prompt_input = None,
):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    device = 'cuda'
    if global_seed < 0:
        global_seed = random.randint(1, 1000000)

    seed = global_seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
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
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
        
    image_encoder=None
    if use_ip_plus_cross_attention:
        if image_encoder_name == 'SAM':
            from segment_anything import SamPredictor, sam_model_registry
            image_encoder = sam_model_registry["vit_b"](checkpoint=image_pretrained_model_path)
            image_encoder.requires_grad_(False)
            image_encoder.to(device)
        else:
            raise ValueError
        print(f'load image encoder: {image_pretrained_model_path}')

    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=torch.bfloat16)
        
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")

    lmm_tokenizer = AutoTokenizer.from_pretrained(lmm_path, trust_remote_code=True)
    lmm_model = AutoModelForCausalLM.from_pretrained(lmm_path, device_map="cuda", trust_remote_code=True).eval()

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
            new_state_dict[k.replace('module.', '')] = v
            
        m, u = unet.load_state_dict(new_state_dict, strict=False)
        logging.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        logging.info(f"missing keys: {m}, \n unexpected keys: {u}")

        
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # Set unet trainable parameters

    
    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Move models to GPU
    vae.to(device)
    text_encoder.to(device)
        
    
    # Get the dataset
    logging.info("***** Loading Data *****")
    

    # Validation pipeline
    validation_pipeline = AnimationPipeline(
        unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, image_encoder=image_encoder, image_encoder_name=image_encoder_name
    ).to("cuda")
        
    validation_pipeline.enable_vae_slicing()
    pixel_transforms = transforms.Compose([transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
    # DDP warpper
    unet.to(device)
    video_paths = os.listdir(video_dir)
    video_paths.sort()
    re = 0
    index = 0
    for video_name in video_paths:
        if index < start_index:
            index = index + 1
            continue
        generator = torch.Generator(device=device)
        # get different seed for each validation round
        generator.manual_seed(global_seed+re)
        
        (height_tile, width_tile) = anchor_target_sampling.target_size
        print(f'multi diff window size: {width_tile}, {height_tile} ')

        video_batch_val = {}
        video_batch_val['videoid'] = video_name
        video_batch_val["fps"] = 8
        video_reader = VideoReader(os.path.join(video_dir, video_name))
        numbers = [i for i in range(64)]
        frames = video_reader.get_batch(numbers)
        video_batch_val["pixel_values"] = pixel_transforms(torch.from_numpy(frames.asnumpy().transpose(0,3,1,2)) / 255.)
        video_batch_val['video_length'] = 64
        
        prompts = None

        # -----------------------------------------------------------------
        pixel_values_valid = video_batch_val['pixel_values'].unsqueeze(0)
        video_length = video_batch_val['video_length']
        videoid = video_batch_val['videoid']
        if 'pixel_pad' in video_batch_val.keys():
            pixel_pad = video_batch_val['pixel_pad']
        else:
            pixel_pad = None
        _, _, _, height, width = pixel_values_valid.shape
        n_frames = pixel_values_valid.shape[1]
        fps_tensor = torch.tensor(video_batch_val["fps"]).to(device).unsqueeze(0)
        negative_prompt = negative_prompt_input[index]
        canvas_sizes, use_multi_diff = get_canvas_size([height, width], target_size, min_overlap, [width_tile, height_tile])
        prompts = prompts_input[index]
        index = index + 1
        for outpainting_round in range(len(canvas_sizes)):
            print(f"outpainting round {outpainting_round}")
            outpaint_size = canvas_sizes[outpainting_round]
            # -----------------------------------------------------------------
            with torch.no_grad():
                videos = validation_pipeline(
                        prompts,
                        videos=pixel_values_valid,
                        use_outpaint=use_outpaint,
                        pixel_pad=pixel_pad,
                        generator    = generator,
                        video_length = n_frames,
                        height       = outpaint_size[0],
                        width        = outpaint_size[1],
                        use_ip_plus_cross_attention = use_ip_plus_cross_attention,
                        ip_plus_condition = ip_plus_condition,
                        use_fps_condition=use_fps_condition,
                        fps_tensor=fps_tensor,
                        negative_prompt=negative_prompt,
                        multi_diff_window = [width_tile, height_tile] if validation_data.multi_diff and use_multi_diff else None,
                        validate_overlap = None if outpainting_round>0 or not use_multi_diff else validate_overlap,
                        round=outpainting_round,
                        lmm_tokenizer=lmm_tokenizer,
                        lmm_model=lmm_model, 
                        **validation_data,
                    ).videos
                if use_outpaint:
                    [videos, videos_masekd, videos_original, video_split, videos_overlap, videos_replace, video_smooth] = videos
                    videos_original = rearrange(videos_original, "b t c h w -> b c t h w")
                    videos_masekd = rearrange(videos_masekd, "b t c h w -> b c t h w")
                    pixel_values_valid = rearrange(videos*2-1, "b c f h w -> b f c h w")
            for sidx, sample in enumerate(videos):
                if sidx > 0: break

                print (f'{sidx}-{prompts[sidx]}')
                sub_samples = []
                
                if use_outpaint:
                    sub_samples.append(video_split.cpu()[sidx].unsqueeze(0))
                    
                sample_num = len(sub_samples)
                sub_samples = torch.concat(sub_samples)
                save_path = f"{output_dir}/samples{video_name}-fps-{fps_tensor[sidx]}-{prompts[:10].replace('/', ' ')}-{n_frames}f-{outpainting_round}{re}.gif"
                save_dir = os.path.dirname(save_path)

                try:
                    os.makedirs(save_dir, exist_ok=True)
                    save_videos_grid(sub_samples, save_path[:-4] + '.mp4', n_rows=sample_num, n_frames=video_length)
                except:
                    print('\n\n=> no space on device!\n\n')
                
                if videos_overlap is not None:
                    sub_samples = []
                    for video_overlap in videos_overlap:
                        sub_samples.append(video_overlap.cpu()[sidx].unsqueeze(0))
                    sample_num = len(sub_samples)
                    sub_samples = torch.concat(sub_samples)
                    save_path = f"{output_dir}/overlaps{video_name}-fps-{fps_tensor[sidx]}-{prompts[:10].replace('/', ' ')}-{n_frames}f-{outpainting_round}{re}.gif"
                    save_dir = os.path.dirname(save_path)

                    try:
                        os.makedirs(save_dir, exist_ok=True)
                        save_videos_grid(sub_samples, save_path[:-4] + '.mp4', n_rows=sample_num, n_frames=video_length)
                    except:
                        print('\n\n=> no space on device!\n\n')
                if videos is not None:
                    sub_samples = []
                    for video in videos:
                        sub_samples.append(video.cpu().unsqueeze(0))
                    sample_num = len(sub_samples)
                    sub_samples = torch.concat(sub_samples)
                    save_path = f"{output_dir}/original{video_name}-fps-{fps_tensor[sidx]}-{prompts[:10].replace('/', ' ')}-{n_frames}f-{outpainting_round}{re}.gif"
                    save_dir = os.path.dirname(save_path)

                    try:
                        os.makedirs(save_dir, exist_ok=True)
                        save_videos_grid(sub_samples, save_path[:-4] + '.mp4', n_rows=sample_num, n_frames=video_length)
                    except:
                        print('\n\n=> no space on device!\n\n')
                if videos_replace is not None:
                    sub_samples = []
                    for video_replace in videos_replace:
                        sub_samples.append(video_replace.cpu().unsqueeze(0))
                    sample_num = len(sub_samples)
                    sub_samples = torch.concat(sub_samples)
                    save_path = f"{output_dir}/replace{video_name}-fps-{fps_tensor[sidx]}-{prompts[:10].replace('/', ' ')}-{n_frames}f-{outpainting_round}{re}.gif"
                    save_dir = os.path.dirname(save_path)
                    try:
                        os.makedirs(save_dir, exist_ok=True)
                        save_videos_grid(sub_samples, save_path[:-4] + '.mp4', n_rows=sample_num, n_frames=video_length)
                    except:
                        print('\n\n=> no space on device!\n\n')
                if video_smooth is not None:
                    sub_samples = []
                    for video_replace in video_smooth:
                        sub_samples.append(video_replace.cpu().unsqueeze(0))
                    sample_num = len(sub_samples)
                    sub_samples = torch.concat(sub_samples)
                    save_path = f"{output_dir}/smooth{video_name}-fps-{fps_tensor[sidx]}-{prompts[:10].replace('/', ' ')}-{n_frames}f-{outpainting_round}{re}.gif"
                    video_name = videoid.split('/')[-1]
                    save_path_result = f'{output_dir}/result/{video_name}'
                    save_dir = os.path.dirname(save_path)
                    try:
                        os.makedirs(save_dir, exist_ok=True)
                        save_videos_grid(sub_samples, save_path[:-4] + '.mp4', n_rows=sample_num, n_frames=video_length)
                        save_videos_grid(sub_samples, save_path_result, n_rows=sample_num, n_frames=video_length)
                    except:
                        print('\n\n=> no space on device!\n\n')
        logging.info(f"Saved samples to {save_path}")
        torch.cuda.empty_cache()
        re = re + 1
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, **config)
