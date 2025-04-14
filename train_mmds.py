import numpy as np
import torch
import argparse
import cv2
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.optim as optim
import os
from model.model import model_fn_decorator
from model.dmmnet import *

from dataset.load_data import *
from tqdm import tqdm
from utils.loss_util import *
from utils.common import *
from config.config import args

def train_epoch(args, TrainImgLoader, model, model_fn, optimizer, epoch, iters, lr_scheduler):
    tbar = tqdm(TrainImgLoader)
    total_loss = 0
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    for batch_idx, data in enumerate(tbar):
        loss = model_fn(args, data, model, iters)
        loss = loss.mean()
        # backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iters += 1
        total_loss += loss.item()
        avg_train_loss = total_loss / (batch_idx + 1)
        desc = 'Training  : Epoch %d, lr %.7f, Avg. Loss = %.5f' % (epoch, lr, avg_train_loss)
        tbar.set_description(desc)
        tbar.update()
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    lr_scheduler.step()

    return lr, avg_train_loss, iters


def init():
    # Make dirs
    args.LOGS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'logs')
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
    args.VISUALS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'train_visual')
    mkdir(args.LOGS_DIR)
    mkdir(args.NETS_DIR)
    mkdir(args.VISUALS_DIR)

    # random seed
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    if args.SEED == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # summary writer
    logger = SummaryWriter(args.LOGS_DIR)

    return logger

def main():
    logger = init()
    device_ids = [0, 1]

    # create MMDS
    mmds = MMDS(en_feature_num=args.EN_FEATURE_NUM,
                     en_inter_num=args.EN_INTER_NUM,
                     de_feature_num=args.DE_FEATURE_NUM,
                     de_inter_num=args.DE_INTER_NUM,
                     sam_number=args.SAM_NUMBER,)
    mmds._initialize_weights()

    mmds = torch.nn.DataParallel(mmds, device_ids=device_ids)
    mmds = mmds.cuda(device=device_ids[0])

    optimizer = optim.Adam([{'params': mmds.parameters(), 'initial_lr': args.BASE_LR}], betas=(0.9, 0.999))
    learning_rate = args.BASE_LR
    iters = 0

    loss_fn = Loss_mmds()
    loss_fn = loss_fn.cuda(device=device_ids[0])

    # create learning rate scheduler
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_MULT, eta_min=args.ETA_MIN,
                                               last_epoch=args.LOAD_EPOCH - 1)
    # create training function
    model_fn = model_fn_decorator(loss_fn=loss_fn, device=device_ids[0])
    # create dataset
    train_path = args.TRAIN_DATASET
    TrainImgLoader = create_dataset(args, data_path=train_path, mode='train')

    avg_train_loss = 0
    for epoch in range(args.LOAD_EPOCH + 1, args.EPOCHS + 1):
        learning_rate, avg_train_loss, iters = train_epoch(args, TrainImgLoader, mmds, model_fn, optimizer, epoch,
                                                           iters, lr_scheduler)

if __name__ == '__main__':
    main()
