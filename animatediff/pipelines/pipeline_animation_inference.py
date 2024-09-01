import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import cv2
import copy
from tqdm import tqdm
from numpy import pi, exp, sqrt
import torchvision.transforms as transforms


from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from animatediff.utils.util import get_boundingbox, replace_video

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDPMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.unet import UNet3DConditionModel

from PIL import Image
import PIL

from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

from ..utils.util import preprocess_image

from ..utils.video_mask import video_mask

logger = logging.get_logger(__name__) 


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        scheduler_train: DDPMScheduler=None,
        image_encoder = None,
        image_encoder_name = 'CLIP'
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            scheduler_train=scheduler_train,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.image_encoder_name = image_encoder_name
        if image_encoder_name == 'SAM':
            from segment_anything import SamPredictor, sam_model_registry
            self.SAMpredictor = SamPredictor(image_encoder)
            self.SAMProcessor = self.SAMpredictor.transform

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae, self.image_encoder]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings
    
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1].to(dtype=self.vae.dtype)).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        init_latents = None
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)

        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents


    def _encode_image_prompt(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image_embeds = self.image_encoder(pil_image.to(self.device)).image_embeds
        uncond_image_prompt_embeds = torch.zeros_like(clip_image_embeds)
        return clip_image_embeds, uncond_image_prompt_embeds
    
    def _encode_image_prompt_plus(self, anchor_pixels_values, ip_plus_condition):
        assert anchor_pixels_values.shape[0] == 1, 'Batch size must be one'

        if self.image_encoder_name == 'CLIP':
            pil_images = []
            if ip_plus_condition == 'image':
                pil_images.append(Image.fromarray(np.uint8(((copy.deepcopy(anchor_pixels_values[:, int(anchor_pixels_values.shape[1]/2), :, :, :])+1.0)/2.0*255).squeeze(0).cpu().numpy().transpose(1,2,0)), mode="RGB"))
            elif ip_plus_condition == 'video':
                n_frames = anchor_pixels_values.shape[1]
                for index in range(n_frames):
                    pil_images.append(Image.fromarray(np.uint8(((copy.deepcopy(anchor_pixels_values[:, index, :, :, :])+1.0)/2.0*255).squeeze(0).cpu().numpy().transpose(1,2,0)), mode="RGB"))
            cond_img = self.clip_image_processor(images=pil_images, return_tensors="pt").pixel_values.to(anchor_pixels_values.device)
            if isinstance(cond_img, Image.Image):
                cond_img = [cond_img]
            if ip_plus_condition == 'image':
                image_embeds = self.image_encoder(cond_img.to(self.device), output_hidden_states=True).hidden_states[-2]
                uncond_image_prompt_embeds = self.image_encoder(torch.zeros_like(cond_img).to(self.device), output_hidden_states=True).hidden_states[-2]
            elif ip_plus_condition == 'video':
                image_embeds = self.image_encoder(cond_img.to(self.device), output_hidden_states=True).hidden_states[-2].unsqueeze(0)
                uncond_image_prompt_embeds = self.image_encoder(torch.zeros_like(cond_img).to(self.device), output_hidden_states=True).hidden_states[-2].unsqueeze(0)
        
        elif self.image_encoder_name == 'SAM':
            assert ip_plus_condition == 'video'
            image_array = np.uint8(((anchor_pixels_values+1.0)/2.0*255).squeeze(dim=0).cpu().numpy().transpose(0,2,3,1))
            image_tensors = []
            for image in image_array:
                # resize the long side to 1024
                image_tensors.append(torch.as_tensor(self.SAMProcessor.apply_image(image), device=anchor_pixels_values.device).permute(2, 0, 1).contiguous())
            image_tensors = torch.stack(image_tensors)
            # pad the short side to 1024 and get features
            batch_size = 8
            assert image_tensors.shape[0]%batch_size == 0
            image_embeds = []
            uncond_image_embeds = []
            for i in range(int(image_tensors.shape[0]/batch_size)):
                self.SAMpredictor.set_torch_image(image_tensors[i*batch_size: (i+1)*batch_size], image_tensors[0].shape[:2])
                image_embeds.append(rearrange(self.SAMpredictor.get_image_embedding(), "f c h w-> f (h w) c"))
                uncond_image_embeds.append(torch.zeros_like(image_embeds[-1]))
            image_embeds = torch.cat(image_embeds, dim=0).unsqueeze(0)
            uncond_image_prompt_embeds = torch.cat(uncond_image_embeds, dim=0).unsqueeze(0)
        
        else:
            raise ValueError

        return image_embeds, uncond_image_prompt_embeds
    
    def get_pixel_pad(self, original_size, target_size, length):
        pad_left = int((target_size[1] - original_size[1]) / 2)
        pad_right = pad_left
        pad_up = int((target_size[0] - original_size[0]) / 2)
        pad_down = pad_up
        pixel_pad = torch.tensor([pad_up, pad_down, pad_left, pad_right], dtype=torch.int16)
        return [pixel_pad]*length   

    
    def _gaussian_weights(self, t_tile_length, t_batch_size):
        from numpy import pi, exp, sqrt

        var = 0.01
        midpoint = (t_tile_length - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        t_probs = [
            exp(-(t - midpoint) * (t - midpoint) / (t_tile_length * t_tile_length) / (2 * var)) / sqrt(2 * pi * var) for
            t in range(t_tile_length)]
        weights = torch.tensor(t_probs)
        weights = weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, t_batch_size, 1, 1)
        return weights

    def get_prompt(self, frame, lmm_tokenizer, lmm_model):
        path = 'infer/temp.jpg'
        frame = cv2.cvtColor((frame.numpy().transpose(1,2,0)+1)/2*255, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, frame)

        lmm_prompt = "Describe the foreground and possible background of the image content in two sentences. Answer starts with 'The image shows'."

        query = lmm_tokenizer.from_list_format([{'image': path},{'text': lmm_prompt}])
        prompt, _ = lmm_model.chat(lmm_tokenizer, query=query, history=None)
        prompt = prompt.replace('The image shows', '').strip()
        print(f'=> Get new prompt: {prompt}')
        
        return prompt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        videos: None,  
        use_outpaint: False, 
        pixel_pad: False,  
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale_text: float = 7.5,
        guidance_scale_adapter: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        use_ip_plus_cross_attention = False,
        use_fps_condition=False,
        fps_tensor=None,
        multi_diff_window=None,
        ip_plus_condition = 'image',
        validate_overlap = None,
        overlap_only = False,
        round = 0,
        t_length=64,
        t_overlap=16,
        lmm_tokenizer=None,
        lmm_model=None,
        **kwargs,
    ):
        # Default height and width to unet
        original_size = [videos.shape[3], videos.shape[4]]

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        do_classifier_free_guidance_text = guidance_scale_text > 1.0
        do_classifier_free_guidance_adapter = guidance_scale_adapter > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance_text, negative_prompt
        )

        # ----------------------------------------------------------------------
        text_embeddings_dynamic = []
        prompt_dynamic = []
        end = 0
        length = videos.shape[1]
        for t_i in range(10000):
            if end == length: break
            start = max(t_i * (t_length - t_overlap), 0)
            end = start + t_length
            if end > length:
                end = length
                start = max(end - t_length, 0)
            print(f'Start: {start}, end: {end}')
            idx = (start + end) // 2
            if prompt is None:
                # get prompt only if not provided
                prompt = [self.get_prompt(videos[0, idx, :,:,:], lmm_tokenizer, lmm_model)]
            text_embeddings = self._encode_prompt(
                prompt, device, num_videos_per_prompt, do_classifier_free_guidance_text, negative_prompt
            )

            text_embeddings_dynamic.append(text_embeddings)
            prompt_dynamic.append(prompt)
        # ----------------------------------------------------------------------

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        videos_overlap = None

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        initial_latent = copy.deepcopy(latents)
        if validate_overlap is not None:
            overlap_latents = []
            for overlap in validate_overlap:
                overlap_latents.append(self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                multi_diff_window[1],
                multi_diff_window[0],
                text_embeddings.dtype,
                device,
                generator,
                None,
                ))

        latents_dtype = latents.dtype
        timesteps = self.scheduler.timesteps
        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # ============================================================
        if use_outpaint or self.scheduler_train is not None:
            pixel_pad = self.get_pixel_pad(original_size, target_size = [height, width], length=videos.shape[0])
            anchor_start = [int(pixel_pad[0][0]), int(pixel_pad[0][2])] 
            pixel_values_valid = videos
            videos = torch.zeros([pixel_values_valid.shape[0], pixel_values_valid.shape[1], pixel_values_valid.shape[2], height, width])
            videos[:, :, :, anchor_start[0]:anchor_start[0]+original_size[0], anchor_start[1]:anchor_start[1]+original_size[1]] = pixel_values_valid
            videos_original = videos.clone()
            mask = video_mask(videos, pixel_pad=pixel_pad)
            anchor_pixels_values = videos[:, :, :, anchor_start[0]:anchor_start[0]+original_size[0], anchor_start[1]:anchor_start[1]+original_size[1]]

            mask_video = mask.clone()
            videos_masked = videos.clone()

            decode_chunk_size = 1
            videos = rearrange(videos, "b f c h w -> (b f) c h w")
            frames = []
            torch.cuda.empty_cache()

            for i in range(0, videos.shape[0], decode_chunk_size):
                num_frames_in = videos[i : i + decode_chunk_size].shape[0]
                frame = self.vae.encode(videos[i : i + decode_chunk_size].to(dtype=self.vae.dtype).to(latents.device), num_frames_in).latent_dist.sample()
                frames.append(frame)
            latents_image_masked = torch.cat(frames)

            latents_image_masked = rearrange(latents_image_masked, "(b f) c h w -> b c f h w", f=video_length)

            latents_image_masked = latents_image_masked * 0.18215

            mask = mask.transpose(2, 1).to(latents.device)
            # resize mask to latent shape
            mask = torch.nn.functional.interpolate(mask, size=(mask.shape[2], latents_image_masked.shape[-2], latents_image_masked.shape[-1]))
        # ============================================================

        image_prompt_embeds, uncond_image_prompt_embeds = None, None
        anchor_embeddings_dynamic = []
        end = 0
        for t_i in range(10000):
            if end == length: break
            start = max(t_i * (t_length - t_overlap), 0)
            end = start + t_length
            if end > length:
                end = length
                start = max(end - t_length, 0)
            if use_ip_plus_cross_attention:
                with torch.no_grad(), torch.autocast("cuda"):
                    image_prompt_embeds, uncond_image_prompt_embeds = self._encode_image_prompt_plus(anchor_pixels_values[:, start:end, :].to(latents.device), ip_plus_condition=ip_plus_condition)

            anchor_embeddings_dynamic.append([image_prompt_embeds, uncond_image_prompt_embeds])


        t_weights = self._gaussian_weights(t_length, 1).to(device=latents.device)
        t_weights = t_weights.to(dtype=latents.dtype)
        length = latents.shape[2]

        with torch.no_grad():
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    latents_all = torch.zeros(latents.shape, device=latents.device, dtype=latents.dtype)
                    contributors = torch.zeros(latents.shape, device=latents.device, dtype=latents.dtype)

                    end = 0
                    for t_i in range(10000):
                        if end == length: break
                        start = max(t_i * (t_length - t_overlap), 0)
                        end = start + t_length
                        if end > length:
                            end = length
                            start = max(end - t_length, 0)

                        text_embeddings = text_embeddings_dynamic[t_i]
                        prompt = prompt_dynamic[t_i]
                        [image_prompt_embeds, uncond_image_prompt_embeds] = anchor_embeddings_dynamic[t_i]

                        latents_tile = latents[:, :, start:end]
                        initial_latent_tile = initial_latent[:, :, start:end]
                        
                        mask_tile = mask[:, :, start:end]
                        latents_image_masked_tile = latents_image_masked[:, :, start:end]
                        latents_outpaint = torch.cat((latents_tile, mask_tile, latents_image_masked_tile), dim=1)
                        if do_classifier_free_guidance_text:
                            if do_classifier_free_guidance_adapter and use_ip_plus_cross_attention:
                                latent_model_input = torch.cat([latents_outpaint] * 3)
                                n_relative_position = 3
                                reference_images_clip_feat = torch.cat([uncond_image_prompt_embeds, uncond_image_prompt_embeds, image_prompt_embeds])
                                text_embeddings = torch.cat([text_embeddings[0].unsqueeze(0), text_embeddings[1].unsqueeze(0), text_embeddings[0].unsqueeze(0)])
                                fps_tensor_input = torch.cat([fps_tensor] * 3)
                            else:
                                latent_model_input = torch.cat([latents_outpaint] * 2)
                                n_relative_position = 2
                                if image_prompt_embeds is not None:
                                    reference_images_clip_feat = torch.cat([image_prompt_embeds, image_prompt_embeds])
                                else:
                                    reference_images_clip_feat = None
                                    n_relative_position = 0
                                fps_tensor_input = torch.cat([fps_tensor] * 2)
                        else:
                            latent_model_input = latents_outpaint


                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        if multi_diff_window is not None:
                            [tile_width, tile_height] = multi_diff_window
                            if not overlap_only:
                                latents_tile = self.multi_diffuse(latents=latents_tile, latent_model_input=latent_model_input,
                                            text_embeddings=text_embeddings, fps_tensor=fps_tensor_input, t=t, guidance_scale_text=guidance_scale_text, 
                                            guidance_scale_adapter=guidance_scale_adapter, extra_step_kwargs=extra_step_kwargs, 
                                            do_classifier_free_guidance_text=do_classifier_free_guidance_text,
                                            do_classifier_free_guidance_adapter=do_classifier_free_guidance_adapter,
                                            tile_width=tile_width, tile_height=tile_height, original_size=original_size, anchor_start=anchor_start, 
                                            use_fps_condition=use_fps_condition, use_ip_plus_cross_attention=use_ip_plus_cross_attention, 
                                            reference_images_clip_feat=reference_images_clip_feat, n_relative_position=n_relative_position, initial_latent=initial_latent_tile)
                        else:
                            # predict the noise residual
                            if use_ip_plus_cross_attention:
                                relative_position = torch.tensor([0, 0, original_size[0], original_size[1], 512, 512], device=latents.device)
                                relative_position = relative_position.unsqueeze(0).repeat(latents_tile.size()[0], 1)
                            else:
                                relative_position = None
                            noise_pred = self.unet(latent_model_input, t, 
                                                    encoder_hidden_states=text_embeddings, 
                                                    use_ip_plus_cross_attention = use_ip_plus_cross_attention,
                                                    reference_images_clip_feat=reference_images_clip_feat,
                                                    use_fps_condition=True,
                                                    fps_tensor= fps_tensor_input,
                                                    relative_position_tensor = torch.cat([relative_position] * n_relative_position)  if (n_relative_position>0) else relative_position,
                                                    ).sample.to(dtype=latents_dtype)
                            
                            # perform guidance
                            if do_classifier_free_guidance_text:
                                if not do_classifier_free_guidance_adapter:
                                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                    noise_pred = noise_pred_uncond + guidance_scale_text * (noise_pred_text - noise_pred_uncond)
                                else:
                                    noise_pred_uncond, noise_pred_text, noise_pred_adapter = noise_pred.chunk(3)
                                    noise_pred = noise_pred_uncond + guidance_scale_text * (noise_pred_text - noise_pred_uncond) + guidance_scale_adapter*(noise_pred_adapter - noise_pred_uncond)

                            # compute the previous noisy sample x_t -> x_t-1
                            latents_tile = self.scheduler.step(noise_pred, t, latents_tile, **extra_step_kwargs).prev_sample

                        l_cur = end - start
                        latents_all[:, :, start:end, :, :] += latents_tile * t_weights[:, :, :l_cur]
                        contributors[:, :, start:end, :, :] += t_weights[:, :, :l_cur]
                    latents = latents_all / contributors

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

                    

            # Post-processing
            # video = self.decode_latents(latents, decode_chunk_size=8)
            video = self.decode_latents(latents)
            if validate_overlap is not None:
                videos_overlap = []
                for overlap_latent in overlap_latents:
                    video_overlap = self.decode_latents(overlap_latent)
                    if output_type == "tensor":
                        video_overlap = torch.from_numpy(video_overlap)
                    videos_overlap.append(video_overlap)
        
        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        video_split, video_smooth, video_replaced = None, None, None
        if not overlap_only:
            video_split = get_boundingbox(video.cpu(), pixel_pad[0], round=round)
            video_smooth, video_replaced = replace_video(video.cpu(), videos_masked.cpu(), pixel_pad[0])
        else:
            video, videos_masked, videos_original = None, None, None
        if use_outpaint or self.scheduler_train is not None:
            return AnimationPipelineOutput(videos=[video, videos_masked, videos_original, video_split, videos_overlap, video_replaced, video_smooth])
        else:
            return AnimationPipelineOutput(videos=video)
            

    def multi_diffuse(self, latents, latent_model_input,
                            text_embeddings, fps_tensor, t, extra_step_kwargs, guidance_scale_text=7.5, guidance_scale_adapter=7.5, 
                            do_classifier_free_guidance_text=True, do_classifier_free_guidance_adapter=True,
                            tile_width=576, tile_height=576, original_size=None, anchor_start=None, use_fps_condition=True,
                            tile_overlap_width=None, tile_overlap_height=None, use_ip_plus_cross_attention=False,
                            reference_images_clip_feat=None, n_relative_position=2, initial_latent=None):
        if tile_overlap_width is None:
            tile_overlap_width = tile_width/4
        if tile_overlap_height is None:
            tile_overlap_height = tile_height/4
        tile_width_vae = int(tile_width // self.vae_scale_factor)
        tile_height_vae = int(tile_height // self.vae_scale_factor)
        batch_size = 1

        tile_overlap_width = int(tile_overlap_width // self.vae_scale_factor)
        tile_overlap_height = int(tile_overlap_height // self.vae_scale_factor)

        tile_weights = self._gaussian_weights_2d(tile_width_vae, tile_height_vae, 1, latents.shape[1], latents.shape[2])
        tile_weights = tile_weights.to(device=self.device).to(dtype=latents.dtype) # 这里一定要是float32
            # =====================================================
        b, f, c, h, w = latents.size()

        grid_rows = 0
        cur_x = 0
        while cur_x < latents.shape[-1]:
            cur_x = max(grid_rows * tile_width_vae - tile_overlap_width * grid_rows, 0) + tile_width_vae
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < latents.shape[-2]:
            cur_y = max(grid_cols * tile_height_vae - tile_overlap_height * grid_cols, 0) + tile_height_vae
            grid_cols += 1

        noise_preds = []

        for row in range(grid_rows):
            noise_preds_row = []
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    ofs_x = max(row * tile_width_vae - tile_overlap_width * row, 0)
                    ofs_y = max(col * tile_height_vae - tile_overlap_height * col, 0)
                if row == grid_rows - 1:
                    ofs_x = w - tile_width_vae
                if col == grid_cols - 1:
                    ofs_y = h - tile_height_vae

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_width_vae
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_height_vae

                torch.cuda.empty_cache()
                latents_tile = latents[:, :, :, input_start_y:input_end_y, input_start_x:input_end_x]
                latent_model_input_tile = latent_model_input[:, :, :, input_start_y:input_end_y, input_start_x:input_end_x]
                relative_position = torch.tensor([(input_start_y - anchor_start[0]), (input_start_x - anchor_start[1]), original_size[0], original_size[1], tile_height, tile_width], device=latents.device)
                relative_position = relative_position.unsqueeze(0).repeat(b, 1)
                noise_pred = self.unet(latent_model_input_tile, t, 
                                    encoder_hidden_states=text_embeddings, 
                                    use_ip_plus_cross_attention = use_ip_plus_cross_attention,
                                    reference_images_clip_feat=reference_images_clip_feat,
                                    use_fps_condition=True,
                                    fps_tensor= fps_tensor,
                                    relative_position_tensor = torch.cat([relative_position] * n_relative_position)  if (n_relative_position>0) else relative_position,
                                    ).sample.to(dtype=latents.dtype)

                # perform guidance
                if do_classifier_free_guidance_text:
                    if not do_classifier_free_guidance_adapter:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale_text * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred_uncond, noise_pred_text, noise_pred_adapter = noise_pred.chunk(3)
                        noise_pred = noise_pred_uncond + guidance_scale_text * (noise_pred_text - noise_pred_uncond) + guidance_scale_adapter*(noise_pred_adapter - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_tile = self.scheduler.step(noise_pred, t, latents_tile, **extra_step_kwargs).prev_sample

                noise_preds_row.append(latents_tile)

            noise_preds.append(noise_preds_row)

        noise_pred_all = torch.zeros(latents.shape, device=latents.device, dtype=latents.dtype)
        contributors = torch.zeros(latents.shape, device=latents.device, dtype=latents.dtype)

        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols - 1 or row < grid_rows - 1:
                    # extract tile from input image
                    ofs_x = max(row * tile_width_vae - tile_overlap_width * row, 0)
                    ofs_y = max(col * tile_height_vae - tile_overlap_height * col, 0)
                    # input tile area on total image
                if row == grid_rows - 1:
                    ofs_x = w - tile_width_vae
                if col == grid_cols - 1:
                    ofs_y = h - tile_height_vae
                input_start_x = ofs_x
                input_end_x = ofs_x + tile_width_vae
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_height_vae
                noise_pred_all[:, :, :, input_start_y:input_end_y, input_start_x:input_end_x] += \
                    noise_preds[row][col] * tile_weights

                contributors[:, :, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights

        if self.scheduler_train is not None:
            noise = initial_latent
            kown_noisy_latent = self.scheduler_train.add_noise(latent_model_input[0][5:], noise, t)
            noise_pred_all = noise_pred_all + (1.0 - latent_model_input[0][4]) * kown_noisy_latent
            contributors = contributors + (1.0 - latent_model_input[0][4])

        noise_pred_all /= contributors
        latents = noise_pred_all

        return latents
    

    def _gaussian_weights_2d(self, tile_width, tile_height, nbatches, channels, t_length):
    ### Generates a gaussian mask of weights for tile contributions

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var)
                    for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var)
                    for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights), (nbatches, channels, t_length, 1, 1))

