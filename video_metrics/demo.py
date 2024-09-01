import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim

from calculate_lpips import calculate_lpips
import argparse, os, sys, json, glob
import torchvision.transforms as transforms
import cv2
import os
import imageio
import pandas as pd
import pdb
import glob

from collections import OrderedDict

import torch
from torch import nn, optim

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import argparse, os, sys, json, glob
import numpy as np
import torch
import random

from tqdm import tqdm

from PIL import Image
import numpy as np

# ps: pixel value should be in [0, 1]!

NUMBER_OF_VIDEOS = 8
VIDEO_LENGTH = 50
CHANNEL = 3
SIZE = 224
videos1 = torch.zeros(
    NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False
)
videos2 = torch.ones(
    NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False
)
device = torch.device("cuda")
# device = torch.device("cpu")

import json

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
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),])

    from decord import VideoReader, cpu
    for idx in tqdm(range(n_samples), desc='Data Loading'):
        real_reader = VideoReader(real_list[idx], ctx=cpu(0))
        fake_reader = VideoReader(fake_list[idx], ctx=cpu(0))
        n = len(real_reader)-1
        # 计算间隔
        # interval = n // 15
        # 生成等间隔的16个数字
        # numbers = [int(round(i * interval)) for i in range(16)]
        # numbers = random.sample(range(n), 16)
        #  numbers.sort()
        numbers = range(16)
        real_frames = real_reader.get_batch(numbers)
        fake_frames = fake_reader.get_batch(numbers) # [t,h,w,c]
        real_frames = real_frames.asnumpy().astype(np.uint8)
        fake_frames = fake_frames.asnumpy().astype(np.uint8)

        if args.transform:
            real_frames = transform(torch.from_numpy(real_frames.transpose(0,3,1,2)))
            fake_frames = transform(torch.from_numpy(fake_frames.transpose(0,3,1,2)))
        else:
            real_frames = (torch.from_numpy(real_frames.transpose(0,3,1,2)))
            fake_frames = (torch.from_numpy(fake_frames.transpose(0,3,1,2)))
        real_frames = real_frames.numpy().transpose(0,2,3,1)
        fake_frames = fake_frames.numpy().transpose(0,2,3,1)

        real_set.append(real_frames)
        fake_set.append(fake_frames)
    
    real = torch.from_numpy(np.stack(real_set, axis=0))
    fake = torch.from_numpy(np.stack(fake_set, axis=0)) # [b,t,h,w,c]
    real = real.permute(0,1,4,2,3)/255.
    fake = fake.permute(0,1,4,2,3)/255.
    return real, fake

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, default='YOUR_PATH/data/480p_gt', help="generated videos saving dir")
    parser.add_argument("--fake1_dir", type=str, default='YOUR_PATH/data/256_66_result/', help="real videos saving dir")
    parser.add_argument("--fake2_dir", type=str, default='YOUR_PATH/data/256_25_result/', help="real videos saving dir")
    parser.add_argument("--save_dir", type=str, default='YOUR_PATH/data/infer/', help="results saving dir")
    parser.add_argument("--transform", type=bool, default=True)
    args = parser.parse_args()
    

    device = torch.device("cuda")
    real, fake = get_videos(args.real_dir, args.fake1_dir)
    result = {}
    result["ssim1"] = calculate_ssim(real, fake)
    result["psnr1"] = calculate_psnr(real, fake)
    result['lpips1'] = calculate_lpips(real, fake, device)

    real, fake = get_videos(args.real_dir, args.fake2_dir)
    result["ssim2"] = calculate_ssim(real, fake)
    result["psnr2"] = calculate_psnr(real, fake)
    result['lpips2'] = calculate_lpips(real, fake, device)
    
    import pdb; pdb.set_trace()
    result['ssim_mean1'] = sum(result["ssim1"]['value'].values())/len(result["ssim1"]['value'])
    result['ssim_mean2'] = sum(result["ssim2"]['value'].values())/len(result["ssim2"]['value'])
    result['psnr_mean1'] = sum(result["psnr2"]['value'].values())/len(result["psnr2"]['value'])
    result['psnr_mean2'] = sum(result["psnr1"]['value'].values())/len(result["psnr1"]['value'])
    result['lpips_mean1'] = sum(result["lpips1"]['value'].values())/len(result["lpips1"]['value'])
    result['lpips_mean2'] = sum(result["lpips2"]['value'].values())/len(result["lpips2"]['value'])
    result["ssim"] = (result["ssim_mean1"] + result["ssim_mean2"]) / 2
    result["psnr"] = (result["psnr_mean1"] + result["psnr_mean2"]) / 2
    result['lpips'] = (result['lpips_mean1'] + result['lpips_mean2']) / 2
    print(result)
    print(json.dumps(result, indent=4))


    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, default='YOUR_PATH/data/720p_gt', help="generated videos saving dir")
    parser.add_argument("--fake1_dir", type=str, default='YOUR_PATH/data/720p_result', help="real videos saving dir")
    parser.add_argument("--save_dir", type=str, default='YOUR_PATH/data/infer/', help="results saving dir")
    parser.add_argument("--transform", type=bool, default=False)
    args = parser.parse_args()

    device = torch.device("cuda")
    real, fake = get_videos(args.real_dir, args.fake1_dir)
    result = {}
    result["ssim1"] = calculate_ssim(real, fake)
    result["psnr1"] = calculate_psnr(real, fake)
    result['lpips1'] = calculate_lpips(real, fake, device)
    
    result['ssim_mean1'] = sum(result["ssim1"]['value'].values())/len(result["ssim1"]['value'])
    result['psnr_mean1'] = sum(result["psnr1"]['value'].values())/len(result["psnr1"]['value'])
    result['lpips_mean1'] = sum(result["lpips1"]['value'].values())/len(result["lpips1"]['value'])
    result["ssim"] = result["ssim_mean1"]
    result["psnr"] = result["psnr_mean1"] 
    result['lpips'] = result['lpips_mean1']
    print(result)
    print(json.dumps(result, indent=4))

    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, default='YOUR_PATH/data/15k_gt', help="generated videos saving dir")
    parser.add_argument("--fake1_dir", type=str, default='YOUR_PATH/data/1_5k_result', help="real videos saving dir")
    parser.add_argument("--save_dir", type=str, default='YOUR_PATH/data/infer/', help="results saving dir")
    parser.add_argument("--transform", type=bool, default=False)
    args = parser.parse_args()

    device = torch.device("cuda")
    real, fake = get_videos(args.real_dir, args.fake1_dir)
    result = {}
    result["ssim1"] = calculate_ssim(real, fake)
    result["psnr1"] = calculate_psnr(real, fake)
    result['lpips1'] = calculate_lpips(real, fake, device)
    
    result['ssim_mean1'] = sum(result["ssim1"]['value'].values())/len(result["ssim1"]['value'])
    result['psnr_mean1'] = sum(result["psnr1"]['value'].values())/len(result["psnr1"]['value'])
    result['lpips_mean1'] = sum(result["lpips1"]['value'].values())/len(result["lpips1"]['value'])
    result["ssim"] = result["ssim_mean1"]
    result["psnr"] = result["psnr_mean1"] 
    result['lpips'] = result['lpips_mean1']
    print(result)
    print(json.dumps(result, indent=4))

    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, default='YOUR_PATH/data/2k_gt', help="generated videos saving dir")
    parser.add_argument("--fake1_dir", type=str, default='YOUR_PATH/data/2k_result', help="real videos saving dir")
    parser.add_argument("--save_dir", type=str, default='YOUR_PATH/data/infer/', help="results saving dir")
    parser.add_argument("--transform", type=bool, default=False)
    args = parser.parse_args()

    device = torch.device("cuda")
    real, fake = get_videos(args.real_dir, args.fake1_dir)
    result = {}
    result["ssim1"] = calculate_ssim(real, fake)
    result["psnr1"] = calculate_psnr(real, fake)
    result['lpips1'] = calculate_lpips(real, fake, device)
    
    result['ssim_mean1'] = sum(result["ssim1"]['value'].values())/len(result["ssim1"]['value'])
    result['psnr_mean1'] = sum(result["psnr1"]['value'].values())/len(result["psnr1"]['value'])
    result['lpips_mean1'] = sum(result["lpips1"]['value'].values())/len(result["lpips1"]['value'])
    result["ssim"] = result["ssim_mean1"]
    result["psnr"] = result["psnr_mean1"] 
    result['lpips'] = result['lpips_mean1']
    print(result)
    print(json.dumps(result, indent=4))