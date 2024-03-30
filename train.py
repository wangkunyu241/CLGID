import warnings
warnings.filterwarnings("ignore")
import os
from config import Config 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
from MPRNet import MPRNet
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx
from vrgnet import EDNet
from SSIM import SSIM
import argparse

parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')
parser.add_argument('--yaml', required=True, type=str, help='Directory of validation images')
parser.add_argument('--hyper', default=1.0, type=float)
args = parser.parse_args()

opt = Config(args.yaml)

nc = 3
nz = 128
nef = 32
######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
def sample_generator(netED, gt):
    random_z = torch.randn(gt.shape[0], nz).cuda()
    rain_make = netED.sample(random_z)  # extract G
    rain_make_max = torch.max(rain_make,1)[0]
    rain_make = rain_make_max.unsqueeze(dim=1).expand_as(rain_make) #gray rain layer
    input_make = rain_make + gt
    return input_make

def train_task(opt, current_task, pre_index=0):
    start_epoch = 1
    if current_task == 0:
        model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, opt.TRAINING.NAME_TASK[current_task])
        utils.mkdir(model_dir)
        train_dir = os.path.join(opt.TRAINING.TRAIN_DIR, opt.TRAINING.NAME_TASK[current_task], 'train')
        memory_test_dir = {}
        generalization_test_dir = {}
        for i in range(0, current_task + 1):
            memory_test_dir[opt.TRAINING.NAME_TASK[i]] = os.path.join(opt.TRAINING.TRAIN_DIR, opt.TRAINING.NAME_TASK[i], 'test')
        for i in range(0, len(opt.TRAINING.TEST_TASK)):
            generalization_test_dir[opt.TRAINING.TEST_TASK[i]] = os.path.join(opt.TRAINING.TEST_PATH, opt.TRAINING.TEST_TASK[i])

        ######### Model ###########
        model_restoration = MPRNet()
        model_restoration.cuda()

        device_ids = [i for i in range(torch.cuda.device_count())]
        if torch.cuda.device_count() > 1:
            print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

        new_lr = opt.OPTIM.LR_INITIAL
        optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)

        ######### Scheduler ###########
        warmup_epochs = 3
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()

        ######### Loss ###########
        criterion_char = losses.CharbonnierLoss()
        criterion_edge = losses.EdgeLoss()
        criterion_ssim = SSIM()

        ######### DataLoaders ###########
        train_dataset = get_training_data(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
        train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

        memory_test_loader = {}
        generalization_test_loader = {}
        for key, value in memory_test_dir.items():
            val_dataset = get_validation_data(value)
            val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
            memory_test_loader[key] = val_loader
        for key, value in generalization_test_dir.items():
            val_dataset = get_validation_data(value)
            val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
            generalization_test_loader[key] = val_loader

        print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
        print('===> Loading datasets')

        for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
            epoch_start_time = time.time()
            epoch_loss = 0
            model_restoration.train()
            for i, data in enumerate((train_loader), 0):
                # zero_grad
                for param in model_restoration.parameters():
                    param.grad = None

                target = data[0].cuda()
                input_ = data[1].cuda()

                restored = model_restoration(input_)
 
                # Compute loss at each stage
                loss_char = np.sum([criterion_char(restored[j],target) for j in range(len(restored))])
                loss_edge = np.sum([criterion_edge(restored[j],target) for j in range(len(restored))])
                loss = (loss_char) + (0.05*loss_edge)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            #### Evaluation ####
            if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
                model_restoration.eval()
                memory_result = {}
                generalization_result = {}
                for key, value in memory_test_loader.items():
                    psnr_val_rgb = []
                    ssim_val_rgb = []
                    for ii, data_val in enumerate((value), 0):
                        target = data_val[0].cuda()
                        input_ = data_val[1].cuda()
                        
                        h, w = input_.shape[2], input_.shape[3]
                        H, W = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
                        padh = H - h if h % 8 != 0 else 0
                        padw = W - w if w % 8 != 0 else 0
                        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
                        target = F.pad(target, (0, padw, 0, padh), 'reflect')

                        with torch.no_grad():
                            restored = model_restoration(input_)
                        restored = restored[0]

                        for res,tar in zip(restored,target):
                            psnr_val_rgb.append(utils.torchPSNR(res, tar))
                            ssim_val_rgb.append(criterion_ssim(torch.unsqueeze(res,0), torch.unsqueeze(tar,0)))

                    psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
                    ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()
                    print('memory task: {}'.format(key), "-[epoch %d PSNR: %.4f SSIM %.4f]" % (epoch, psnr_val_rgb, ssim_val_rgb))

                for key, value in generalization_test_loader.items():
                    psnr_val_rgb = []
                    ssim_val_rgb = []
                    for ii, data_val in enumerate((value), 0):
                        target = data_val[0].cuda()
                        input_ = data_val[1].cuda()
                        
                        h, w = input_.shape[2], input_.shape[3]
                        H, W = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
                        padh = H - h if h % 8 != 0 else 0
                        padw = W - w if w % 8 != 0 else 0
                        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
                        target = F.pad(target, (0, padw, 0, padh), 'reflect')

                        with torch.no_grad():
                            restored = model_restoration(input_)
                        restored = restored[0]

                        for res,tar in zip(restored,target):
                            psnr_val_rgb.append(utils.torchPSNR(res, tar))
                            ssim_val_rgb.append(criterion_ssim(torch.unsqueeze(res,0), torch.unsqueeze(tar,0)))

                    psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
                    ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()
                    print('generalization task: {}'.format(key), "-[epoch %d PSNR: %.4f SSIM %.4f]" % (epoch, psnr_val_rgb, ssim_val_rgb))

                torch.save({'state_dict': model_restoration.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(model_dir,"model_latest.pth"))

            scheduler.step()

            print("------------------------------------------------------------------")
            print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
            print("------------------------------------------------------------------")

    if current_task > 0:
        model_dir = os.path.join(opt.TRAINING.SAVE_DIR, opt.TRAINING.NAME_TASK[current_task])
        utils.mkdir(model_dir)
        train_dir = os.path.join(opt.TRAINING.TRAIN_DIR, opt.TRAINING.NAME_TASK[current_task], 'train')
        replay_dir = {}
        memory_test_dir = {}
        generalization_test_dir = {}
        for i in range(0, current_task):
            replay_dir[opt.TRAINING.NAME_TASK[i]] = os.path.join(opt.TRAINING.TRAIN_DIR, opt.TRAINING.NAME_TASK[i], 'train')
        for i in range(0, current_task + 1):
            memory_test_dir[opt.TRAINING.NAME_TASK[i]] = os.path.join(opt.TRAINING.TRAIN_DIR, opt.TRAINING.NAME_TASK[i], 'test')
        for i in range(0, len(opt.TRAINING.TEST_TASK)):
            generalization_test_dir[opt.TRAINING.TEST_TASK[i]] = os.path.join(opt.TRAINING.TEST_PATH, opt.TRAINING.TEST_TASK[i])

        ######### Model ###########
        model_restoration = MPRNet()
        model_restoration.cuda()
        model_restoration_distill = MPRNet()
        utils.freeze(model_restoration_distill)
        model_restoration_distill.cuda()

        pre_model = torch.nn.ModuleList()
        for i in range(len(pre_index)):
            netED = EDNet(nc, nz, nef)
            netED.cuda()
            path = opt.TRAINING.ED_PATH + opt.TRAINING.NAME_TASK[i] + '/ED_state_700.pt'
            netED.load_state_dict(torch.load(str(path)))
            netED.eval()
            utils.freeze(netED)
            pre_model.append(netED)

        device_ids = [i for i in range(torch.cuda.device_count())]
        if torch.cuda.device_count() > 1:
            print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


        new_lr = opt.OPTIM.LR_INITIAL
        optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)


        ######### Scheduler ###########
        warmup_epochs = 3
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()

        ######### Resume ###########
        pre_model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, opt.TRAINING.NAME_TASK[current_task-1], "model_latest.pth")
        utils.load_checkpoint(model_restoration, pre_model_dir)
        utils.load_checkpoint(model_restoration_distill, pre_model_dir)

        ######### Loss ###########
        criterion_char = losses.CharbonnierLoss()
        criterion_edge = losses.EdgeLoss()
        criterionL1 = nn.L1Loss()
        criterion_ssim = SSIM()

        ######### DataLoaders ###########
        train_dataset = get_training_data(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
        train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

        memory_test_loader = {}
        generalization_test_loader = {}
        for key, value in memory_test_dir.items():
            val_dataset = get_validation_data(value)
            val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
            memory_test_loader[key] = val_loader
        for key, value in generalization_test_dir.items():
            val_dataset = get_validation_data(value)
            val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
            generalization_test_loader[key] = val_loader

        print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
        print('===> Loading datasets')

        for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
            epoch_start_time = time.time()
            epoch_loss = 0
            epoch_distill_loss = 0
            model_restoration.train()
            for i, data in enumerate((train_loader), 0):
                # zero_grad
                for param in model_restoration.parameters():
                    param.grad = None

                target = data[0].cuda()
                input_ = data[1].cuda()

                syn_input_list = []
                syn_target_list = []
                ind = list(range(len(pre_index)))
                for ii in range(input_.size(0)):
                    np.random.shuffle(ind)
                    syn_input = sample_generator(pre_model[ind[0]], target[ii].unsqueeze(0))
                    syn_input.clamp_(0.0, 1.0)
                    syn_input_list.append(syn_input)
                    syn_target_list.append(target[ii].unsqueeze(0))
                
                syn_input = torch.cat(syn_input_list, dim=0)
                syn_input = syn_input.cuda()
                syn_target = torch.cat(syn_target_list, dim=0)
                syn_target = syn_target.cuda()

                restored = model_restoration(input_)
                replay_restored = model_restoration(syn_input)
                replay_restored_distill = model_restoration_distill(syn_input)

                # Compute loss at each stage
                loss_char = np.sum([criterion_char(restored[j], target) for j in range(len(restored))])
                loss_edge = np.sum([criterion_edge(restored[j], target) for j in range(len(restored))])
                syn_loss_char = np.sum([criterion_char(replay_restored[j], syn_target) for j in range(len(replay_restored))])
                syn_loss_edge = np.sum([criterion_edge(replay_restored[j], syn_target) for j in range(len(replay_restored))])
                loss = ( (loss_char + 0.05 * loss_edge) + (syn_loss_char + 0.05 * syn_loss_edge) ) / 2
                distill_loss = np.sum([(criterionL1(replay_restored[j],replay_restored_distill[j])).cpu().detach().numpy() for j in range(len(replay_restored))])

                loss = loss + args.hyper * distill_loss

                loss.backward()
                optimizer.step()
                epoch_loss +=loss.item()
                epoch_distill_loss += distill_loss.item()

            if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
                model_restoration.eval()
                memory_result = {}
                generalization_result = {}
                for key, value in memory_test_loader.items():
                    psnr_val_rgb = []
                    ssim_val_rgb = []
                    for ii, data_val in enumerate((value), 0):
                        target = data_val[0].cuda()
                        input_ = data_val[1].cuda()
                        
                        h, w = input_.shape[2], input_.shape[3]
                        H, W = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
                        padh = H - h if h % 8 != 0 else 0
                        padw = W - w if w % 8 != 0 else 0
                        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
                        target = F.pad(target, (0, padw, 0, padh), 'reflect')

                        with torch.no_grad():
                            restored = model_restoration(input_)
                        restored = restored[0]

                        for res,tar in zip(restored,target):
                            psnr_val_rgb.append(utils.torchPSNR(res, tar))
                            ssim_val_rgb.append(criterion_ssim(torch.unsqueeze(res,0), torch.unsqueeze(tar,0)))

                    psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
                    ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()
                    print('memory task: {}'.format(key), "-[epoch %d PSNR: %.4f SSIM %.4f]" % (epoch, psnr_val_rgb, ssim_val_rgb))

                for key, value in generalization_test_loader.items():
                    psnr_val_rgb = []
                    ssim_val_rgb = []
                    for ii, data_val in enumerate((value), 0):
                        target = data_val[0].cuda()
                        input_ = data_val[1].cuda()
                        
                        h, w = input_.shape[2], input_.shape[3]
                        H, W = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
                        padh = H - h if h % 8 != 0 else 0
                        padw = W - w if w % 8 != 0 else 0
                        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
                        target = F.pad(target, (0, padw, 0, padh), 'reflect')

                        with torch.no_grad():
                            restored = model_restoration(input_)
                        restored = restored[0]

                        for res,tar in zip(restored,target):
                            psnr_val_rgb.append(utils.torchPSNR(res, tar))
                            ssim_val_rgb.append(criterion_ssim(torch.unsqueeze(res,0), torch.unsqueeze(tar,0)))

                    psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
                    ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()
                    print('generalization task: {}'.format(key), "-[epoch %d PSNR: %.4f SSIM %.4f]" % (epoch, psnr_val_rgb, ssim_val_rgb))

                torch.save({'state_dict': model_restoration.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(model_dir,"model_latest.pth"))

            scheduler.step()

            print("------------------------------------------------------------------")
            print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
            print("------------------------------------------------------------------")


if __name__ == "__main__":
    random_perm = list(range(opt.TRAINING.NUM_TASK))
    for i in range(opt.TRAINING.START, opt.TRAINING.NUM_TASK):
        print("-------------------Get started--------------- ")
        print("Training on Task " + str(i))
        if i == 0:
            pre_index = 0
            train_task(opt, i, pre_index)
        else:
            pre_index = random_perm[:i]
            train_task(opt, i, pre_index)