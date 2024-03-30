"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_validation_data, get_test_data
from MPRNet import MPRNet
from skimage import img_as_ubyte
from pdb import set_trace as stx
from SSIM import SSIM
from skimage.measure.simple_metrics import compare_psnr

parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')
parser.add_argument("--checkpoint", type=str, required=True, help='path to model')
parser.add_argument("--data_path", type=str, required=True, help='path to training data')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

model_restoration = MPRNet()
criterion = SSIM()

utils.load_checkpoint(model_restoration, opt.checkpoint)
model_restoration.cuda()
criterion.cuda()
model_restoration.eval()

test_dataset = get_test_data(opt.data_path)
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

with torch.no_grad():
    for ii, data_test in enumerate((test_loader), 0):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
            
        gt = data_test[0].cuda()
        input_ = data_test[1].cuda()

        restored = model_restoration(input_)
        restored = torch.clamp(restored[0],0,1)

        SSIM_ += criterion(restored, gt)
        PSNR_ += batch_PSNR(restored, gt, 1.)
        count = count + 1

    print('SSIM:', SSIM_ / count)
    print('PSNR:', PSNR_ / count)
