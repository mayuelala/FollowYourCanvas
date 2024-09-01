import numpy as np
import torch
from PIL import Image
import cv2
import our_fvd
import pandas as pd
import pdb
import glob
from torchvision import transforms 
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# import their_fvd
# from util import open_url
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "4"



@torch.no_grad()
def compute_our_fvd(videos_fake: np.ndarray, videos_real: np.ndarray, device: str='cuda') -> float:
#     detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
    detector = torch.jit.load("YOUR_PATH/data/i3d_torchscript.pt").eval().to(device)

    videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(device)
    videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(device)
    feats_fake = detector(videos_fake, **detector_kwargs).cpu().numpy()
    feats_real = detector(videos_real, **detector_kwargs).cpu().numpy()
    print(feats_fake.mean(), feats_real.mean())
#     feats_fake
    return our_fvd.compute_fvd(feats_fake, feats_real)

def load_video_keyframes(
    video_path, size: tuple = None
):
    """
    Load keyframes from a video file or directory of images.

    Args:
        video_path (str): Path to the video file or directory of images.
        original_fps (int): The original frames per second of the video.
        target_fps (int): The desired frames per second of the output keyframes.
        num_keyframes (int): The number of keyframes to extract.
        size (tuple, optional): The desired size of the output keyframes. Defaults to None.

    Returns:
        torch.Tensor: A tensor of shape (T, C, H, W) containing the keyframes.
    """
    if os.path.isdir(video_path):
        files = sorted(os.listdir(video_path))

        num_allframes = 16
        keyindexs = [i for i in range(0, num_allframes, 1)]
        keyfiles = [files[i] for i in keyindexs]
        frames = [load_img(os.path.join(video_path, kf), size) for kf in keyfiles]
    
    return frames

def load_img(p_cond_img, size: tuple = None):
    """
    Loads an image from the given path, crops it to the center, resizes it to the given size, 
    and then converts the image to a tensor and normalizes it to the range [-1, 1].
    Shape: (1, 3, H, W)

    Args:
    - p_cond_img (str): path to the image file
    - size (tuple): height and width to resize the image to

    Returns:
    - cond_img (torch.Tensor): tensor of the cropped, resized and normalized image.
    """
    cond_img = Image.open(p_cond_img)
    cond_img = cond_img.convert("RGB")
    if size is not None:
        cond_img = cond_img.resize((size[1], size[0]))
    
    cond_img = np.array(cond_img)
    cond_img = torch.from_numpy(cond_img)
    
    return cond_img

def get_filelist(data_dir):
    file_list = glob.glob(os.path.join(data_dir, '*.mp4'))
    file_list.sort()
    return file_list

def get_videos(real_dir, fake_dir):
    real_list = get_filelist(real_dir)
    fake_list = get_filelist(fake_dir)
    assert len(real_list) == len(fake_list), "Error: samples are not paired for real and fake folders!"
    n_samples = len(real_list)
    print("n_samples: %d"%n_samples)
    real_set = []
    fake_set = []
    
    transform = transforms.Compose([
    transforms.Resize((224, 224))])

    from decord import VideoReader, cpu
    for idx in tqdm(range(n_samples), desc='Data Loading'):
        real_reader = VideoReader(real_list[idx], ctx=cpu(0))
        fake_reader = VideoReader(fake_list[idx], ctx=cpu(0))
        n = len(real_reader)-1
        numbers = range(16)
        real_frames = real_reader.get_batch(numbers)
        fake_frames = fake_reader.get_batch(numbers) # [t,h,w,c]
        real_frames = real_frames.asnumpy().astype(np.uint8)
        fake_frames = fake_frames.asnumpy().astype(np.uint8)

        real_frames = transform(torch.from_numpy(real_frames.transpose(0,3,1,2)))
        fake_frames = transform(torch.from_numpy(fake_frames.transpose(0,3,1,2)))

        real_frames = real_frames.numpy().transpose(0,2,3,1)
        fake_frames = fake_frames.numpy().transpose(0,2,3,1)

        real_set.append(real_frames)
        fake_set.append(fake_frames)
    
    real = torch.from_numpy(np.stack(real_set, axis=0))
    fake = torch.from_numpy(np.stack(fake_set, axis=0)) # [b,t,h,w,c]
    real = real/255.
    fake = fake/255.
    return real, fake

def main():
    result_path = "YOUR_PATH/data/256_25_result/"
    print(result_path)
    original_path = "YOUR_PATH/data/480p_gt"
    real, fake = get_videos(result_path, original_path)
    fvd = compute_our_fvd(real.float().numpy(), fake.float().numpy(), device = 'cpu')
    print(fvd)

    result_path = "YOUR_PATH/data/256_66_result/"
    print(result_path)
    original_path = "YOUR_PATH/data/480p_gt"
    real, fake = get_videos(result_path, original_path)
    fvd = compute_our_fvd(real.float().numpy(), fake.float().numpy(), device = 'cpu')
    print(fvd)

    result_path = "YOUR_PATH/data/2k_result"
    print(result_path)
    original_path = "YOUR_PATH/data/2k_gt"
    real, fake = get_videos(result_path, original_path)
    fvd = compute_our_fvd(real.float().numpy(), fake.float().numpy(), device = 'cpu')
    print(fvd)

    result_path = "YOUR_PATH/data/1_5k_result"
    print(result_path)
    original_path = "YOUR_PATH/data/1_5k_gt"
    real, fake = get_videos(result_path, original_path)
    fvd = compute_our_fvd(real.float().numpy(), fake.float().numpy(), device = 'cpu')
    print(fvd)

    result_path = "YOUR_PATH/data/720p_result"
    print(result_path)
    original_path = "YOUR_PATH/data/720p_gt"
    real, fake = get_videos(result_path, original_path)
    fvd = compute_our_fvd(real.float().numpy(), fake.float().numpy(), device = 'cpu')
    print(fvd)

if __name__ == "__main__":
    main()
