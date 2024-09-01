import os, io, csv, math, random
import numpy as np
from einops import rearrange, repeat
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from animatediff.utils.util import zero_rank_print
from torch.utils.data import ConcatDataset
from PIL import Image
import random
import torchvision.transforms as T
import traceback
from animatediff.utils.video_mask import get_anchor_target


from ..utils.video_mask import video_mask

def get_proper_resize_size(img, size):
    from PIL import Image
    if isinstance(size, int):
        size = (size, size)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    W, H = img.size
    h, w = size
    r = max(h / H, w / W)
    return int(H * r), int(W * r)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class VideoDatasetFull(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
            clip_image_size=224,
            dynamic_fps=False,
            add_first_image=False,
            local_rank=None,
            muti_scale_training=False,
            multi_scale_sample_size=None,
            random_sample_time = True,
            get_SAM_anchor_image = False,
            anchor_target_sampling = None,
            SAMProcessor = None
        ):
        zero_rank_print(f"VideoDatasetFull class: loading annotations from {csv_path} ...")
        with open(csv_path, 'rU', newline="\n") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"VideoDatasetFull class: data scale: {self.length}")

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        self.dynamic_fps     = dynamic_fps
        self.add_first_image = add_first_image
        self.random_sample_time = random_sample_time
        self.anchor_target_sampling = anchor_target_sampling
        self.get_SAM_anchor_image = get_SAM_anchor_image
        self.SAMProcessor = SAMProcessor
        
        if muti_scale_training:
            scale_num = len(multi_scale_sample_size) // 2
            idx = local_rank % scale_num
            idx1 = int(idx * 2)
            idx2 = int(idx * 2 + 1)
            sample_size = (multi_scale_sample_size[idx1], multi_scale_sample_size[idx2])
        else:
            sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        print('sample size', sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        
        
    def get_batch(self, idx):
        video_dict = self.dataset[idx]

        # -----------------------------------------------------------------
        if 'videoid' in video_dict:
            videoid = video_dict['videoid']
        elif 'path' in video_dict:
            videoid = video_dict['path']
        else:
            raise 'Can not find videoid!'
        
        if 'name' in video_dict:
            name = video_dict['name']
        elif 'qwen_response' in video_dict:
            name = video_dict['qwen_response']
        else:
            raise 'Can not find video name!'

        if '.' in videoid:
            video_dir = os.path.join(self.video_folder, f"{videoid}")
        else:
            video_dir = os.path.join(self.video_folder, f"{videoid}.mp4")
        # -----------------------------------------------------------------

        if self.dynamic_fps:
            self.sample_stride = random.randint(2, 16)
        
        try:
            video_reader = VideoReader(video_dir)
        except Exception as e:
            print (f'Error: {e}, remove: {video_dir}')
            raise
        video_length = len(video_reader)
        if not self.is_image:
            framelst = list(range(0, len(video_reader), self.sample_stride))
            if self.sample_n_frames == -1:
                # should be n*16
                last = framelst[-1]
                frame_number = int(np.ceil(len(framelst)/16)*16)
                framelst.extend([last for _ in range(frame_number-len(framelst))])
                batch_index = framelst
                video_length = len(framelst)
            else:
                if len(framelst) < self.sample_n_frames:
                    tmp_sample_stride = max(1, len(video_reader)//(self.sample_n_frames+1))
                    framelst = list(range(0, len(video_reader), tmp_sample_stride))
                    video_length = len(framelst)
                    if len(framelst) < self.sample_n_frames:
                        last = framelst[-1]
                        framelst.extend([last for _ in range(self.sample_n_frames-len(framelst))])
                        print(f'Warning: video is too short, duplicate last frame idx')
                else:
                    video_length = self.sample_n_frames
                if len(framelst) > self.sample_n_frames and self.random_sample_time:
                    start_idx = random.randint(0, len(framelst)-self.sample_n_frames)
                else:
                    start_idx = 0

                batch_index = framelst[start_idx:start_idx+self.sample_n_frames]
        else:
            batch_index = [random.randint(0, video_length - 1)]

        video_images = video_reader.get_batch(batch_index)  # (f, h, w, c)
        pixel_values = torch.from_numpy(video_images.asnumpy()).permute(0, 3, 1, 2).contiguous()

        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return pixel_values, name, video_images, self.sample_stride, video_length, videoid
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            pixel_values, name, video_images, sample_stride, video_length, videoid = self.get_batch(idx)

            pixel_values = self.pixel_transforms(pixel_values)
            if self.add_first_image:
                first_image = self.get_first_image_condition(video_images)
                sample = dict(pixel_values=pixel_values,
                                text=name,
                                ori_text=name,
                                clip_images = first_image,
                                fps=sample_stride)
            else:
                anchor_pixels_values, target_pixels_values, mask, relative_position, image_tensors_sam = 0, 0, 0, 0, 0
                if self.anchor_target_sampling is not None:
                    anchor_pixels_values, target_pixels_values, mask, relative_position = get_anchor_target(pixel_values, self.anchor_target_sampling)
                    anchor_pixels_values, target_pixels_values, mask, relative_position = anchor_pixels_values.squeeze(0), target_pixels_values.squeeze(0), mask.squeeze(0), relative_position.squeeze(0)
                    if self.get_SAM_anchor_image:
                        image_array = np.uint8(((anchor_pixels_values+1.0)/2.0*255).squeeze(dim=0).cpu().numpy().transpose(0,2,3,1))
                        image_tensors_sam = []
                        for image in image_array:
                            image_tensors_sam.append(torch.as_tensor(self.SAMProcessor.apply_image(image), device=anchor_pixels_values.device).permute(2, 0, 1).contiguous())
                        image_tensors_sam = torch.stack(image_tensors_sam)
                sample = dict(pixel_values=pixel_values,
                                text=name,
                                ori_text=name,
                                fps=sample_stride,
                                video_length=video_length,
                                videoid = videoid,
                                anchor_pixels_values = anchor_pixels_values, 
                                target_pixels_values = target_pixels_values, 
                                mask=mask, 
                                relative_position = relative_position, 
                                image_tensors_sam = image_tensors_sam)
            return sample
        except Exception as e:
            print(traceback.format_exc())
            new_idx = random.randint(0, self.length-1)
            return self.__getitem__(new_idx)
    
class VideoDataset:
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
            dynamic_fps=False,
            add_first_image=False,
            local_rank=None,
            muti_scale_training=False,
            multi_scale_sample_size=None,
            random_sample_time=True,
            get_SAM_anchor_image = False,
            anchor_target_sampling = None,
            SAMProcessor = None
        ):
        self.datasets = []
        for csv_p, folder in zip(csv_path, video_folder):
            dataset = VideoDatasetFull(
                csv_path=csv_p, 
                video_folder=folder,
                sample_size=sample_size, 
                sample_stride=sample_stride, 
                sample_n_frames=sample_n_frames,
                is_image=is_image,
                dynamic_fps=dynamic_fps,
                add_first_image=add_first_image,
                local_rank=local_rank,
                muti_scale_training=muti_scale_training,
                multi_scale_sample_size=multi_scale_sample_size,
                random_sample_time = random_sample_time,
                get_SAM_anchor_image = get_SAM_anchor_image,
                anchor_target_sampling = anchor_target_sampling,
                SAMProcessor = SAMProcessor
            )
            self.datasets.append(dataset)
            
        self.concat_dataset = ConcatDataset(self.datasets)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, item):
        return self.concat_dataset[item]
