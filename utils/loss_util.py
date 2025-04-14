import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from utils.common import *
from torch.nn.parameter import Parameter
import os
from math import exp
from model.dmmnet import *

class Loss_mmds(torch.nn.Module):
    def __init__(self, ):
        super(Loss_mmds, self).__init__()
        self.loss_fn = VGGPerceptualLoss()

    def forward(self, out1, out2, out3, gt1, feature_layers=[2]):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)
        
        loss1 = self.loss_fn(out1, gt1, feature_layers=feature_layers) + F.l1_loss(out1, gt1)
        loss2 = self.loss_fn(out2, gt2, feature_layers=feature_layers) + F.l1_loss(out2, gt2)
        loss3 = self.loss_fn(out3, gt3, feature_layers=feature_layers) + F.l1_loss(out3, gt3)
        
        return loss1+loss2+loss3 

class Loss_fsrs(torch.nn.Module):
    def __init__(self, device):
        super(Loss_fsrs, self).__init__()

        self.device = device
        self.loss_fn = VGGPerceptualLoss().to(self.device)
        self.criterion_BCE = nn.BCELoss().to(self.device)
        self.dwt = DWT().to(self.device)
        self.criterion_MSE = nn.MSELoss().to(self.device)

    def forward(self, output, gt, structure, detail3):

        cl_img_structure, cl_img_detail9, cl_img_detail3 = self.dwt(gt)

        #---------color loss
        # better cos
        # b, c, h, w = structure.shape
        # true_reflect_view = cl_img_structure.view(b, c, h * w).permute(0, 2, 1)
        # pred_reflect_view = structure.view(b, c, h * w).permute(0, 2, 1)
        # true_reflect_norm = torch.nn.functional.normalize(true_reflect_view, dim=-1)
        # pred_reflect_norm = torch.nn.functional.normalize(pred_reflect_view, dim=-1)
        # cose_value = true_reflect_norm * pred_reflect_norm
        # cose_value = torch.sum(cose_value, dim=-1)  # 16 x (512x512)  # print(cose_value.min(), cose_value.max())
        # color_loss = torch.mean(1 - cose_value)
        # color_loss = self.criterion_BCE(cl_img_structure, structure)

        #---------detail loss
        detail_loss = self.criterion_MSE(detail3, cl_img_detail3)

        #---------recon loss
        recon_loss = self.loss_fn(output, gt, feature_layers=[2]) + F.l1_loss(output, gt)

        loss = 0.2 * color_loss + 0.2 * detail_loss + 0.8 * recon_loss

        return loss, recon_loss, color_loss, detail_loss

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate

        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        device = target.device
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device))

        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
