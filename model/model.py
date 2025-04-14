import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.loss_util import *
from utils.common import *

def model_fn_decorator(loss_fn, device, mode='train'):

    def model_fn(args, data, model, iters):
        model.train()

        in_img = data['in_img'].cuda(device=device)
        label = data['label'].cuda(device=device)

        out_1, out_2, out_3 = model(in_img)
        loss = loss_fn(out_1, out_2, out_3, label)

        return loss

    def fsrs_fn(args, data, model, iters, mmds):
        model.train()
        mmds.eval()

        in_img = data['in_img'].cuda(device=device)
        label = data['label'].cuda(device=device)

        mmds_out, _1, _2 = mmds(in_img)
        ll, detail3, fsrs_output = model(mmds_out)
        loss, recon_loss, color_loss, detail_loss = loss_fn(fsrs_output, label, ll, detail3)

        return loss, recon_loss, color_loss, detail_loss

    if mode == 'fsrs':
        fn = fsrs_fn
    else:
        fn = model_fn
    return fn