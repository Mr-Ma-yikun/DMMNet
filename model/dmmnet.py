import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parameter import Parameter
import numpy as np
import functools
from typing import Any, Sequence, Tuple
import einops
from math import *
import torchvision.utils as vutils
import os

#定义一些卷积块
class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, pad=0,bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias)
    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1,bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias)
    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)

def Conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def block_images_einops(x, patch_size):  #n, h, w, c
  """Image to patches."""
  batch, height, width, channels = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
  return x

def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x

class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out

class conv_leakyrelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_leakyrelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(init=0.1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(init=0.1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        _, num, h, w = x1.shape

        num_ll = num//4
        num_lh = num//2
        num_hl = 3*num//4
        num_hh = num

        ll = x1[:, 0:num_ll, :, :]
        lh = x1[:, num_ll:num_lh, :, :]
        hl = x1[:, num_lh:num_hl, :, :]
        hh = x1[:, num_hl:num_hh, :, :]

        idwt_in = torch.cat([ll, lh, hl, hh], dim=1)
        # 4*c --- c
        x1 = IDWT(idwt_in)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU(init=0.1)

    def forward(self, x):
        return self.relu(self.conv(x))

def get_wav(in_channels):
    """wavelet decomposition using conv2d"""

    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    net = nn.Conv2d
   
    #h,w - h/2, w/2
    LL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    LH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    return LL, LH, HL, HH

class Wavelet(nn.Module):
    def __init__(self, in_channels):
        super(Wavelet, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

class New_DWT(nn.Module):
    def __init__(self, in_c):
        super(New_DWT,self).__init__()
        self.dwt = Wavelet(in_c)

    def forward(self,x):
        LL, LH, HL, HH = self.dwt(x)
        out = torch.cat([LL, LH, HL, HH], dim=1)

        return out

class DWT(nn.Module):
    def __init__(self):
        super(DWT,self).__init__()
        self.dwt = Wavelet(3)

    def forward(self,x):
        LL, LH, HL, HH = self.dwt(x)

        detail9 = torch.cat([LH, HL, HH], dim=1)
        detail3 = LH + HL + HH

        return LL, detail9, detail3

def IDWT(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.shape
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class WaveGT(nn.Module):
    def __init__(self):
        super(WaveGT, self).__init__()
        self.pool = Wavelet(3)

    def forward(self, x):

        LL, LH, HL, HH = self.pool(x)
        structure = LL
        detail9 = torch.cat([LH, HL, HH], dim=1)  # (4,9,128,128)
        detail3 = LH + HL + HH  # (4,3,128,128)
        return structure, detail9, detail3

class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 8)
        self.input = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=9, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU(init=0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        res_x = x
        
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)

        out = out + res_x

        return out


class UNetEncoder(nn.Module):
    def __init__(self, n_channels=12):
        super(UNetEncoder, self).__init__()
        self.inc = inconv(n_channels, 16)
        self.down1 = down(64, 64)
        self.down2 = down(256, 256)
        self.down3 = down(1024, 1024)
        self.dwt1 = New_DWT(3)
        self.dwt2 = New_DWT(16)
        self.dwt3 = New_DWT(64)
        self.dwt4 = New_DWT(256)

    def forward(self, x):
        # 3 --- 12
        x_dwt_1= self.dwt1(x)
        # 12 --- 16
        x1 = self.inc(x_dwt_1)

        # 16 --- 64
        x_dwt_2= self.dwt2(x1)
        # 64 --- 64
        x2 = self.down1(x_dwt_2)

        # 64 --- 256
        x_dwt_3= self.dwt3(x2)
        # 256 --- 256
        x3 = self.down2(x_dwt_3)

        # 256 --- 1024
        x_dwt_4= self.dwt4(x3)
        # 1024 --- 1024
        x4 = self.down3(x_dwt_4)

        return x4, (x1, x2, x3)


class UNetDecoder(nn.Module):
    def __init__(self, n_channels=12):
        super(UNetDecoder, self).__init__()
        self.up1 = up(512, 256)
        self.up2 = up(128, 64)
        self.up3 = up(32, 16)
        self.outc = outconv(16, n_channels)

        self.sigmoid = nn.Sigmoid()
        self.c = nn.Conv2d(3, 3, 1, padding=0, bias=False)
        self.h3_detail_nn = DNet()

    def forward(self, x, enc_outs, res_x):
        x = self.sigmoid(x)

        # 256
        x = self.up1(x, enc_outs[2])
        # 64
        x = self.up2(x, enc_outs[1])
        # 16
        x = self.up3(x, enc_outs[0])
        # 16 --- 12
        x1 = self.outc(x)

        ll = x1[:, 0:3, :, :]
        lh = x1[:, 3:6, :, :]
        hl = x1[:, 6:9, :, :]
        hh = x1[:, 9:12, :, :]

        detail9 = torch.cat([lh, hl, hh], dim=1) 
        detail9 = self.h3_detail_nn(detail9)

        lh_d = detail9[:, 0:3, :, :]
        hl_d = detail9[:, 3:6, :, :]
        hh_d = detail9[:, 6:9, :, :]

        idwt_in = torch.cat([ll, lh_d, hl_d, hh_d], dim=1)
        # 4*c --- c
        x1 = IDWT(idwt_in)

        x1 = self.c(x1)
        x1 = x1 + res_x

        return x1

class FSRS(nn.Module):
    def __init__(self):
        super(FSRS, self).__init__()

        self.dwt = DWT()

        self.ll_encoder = UNetEncoder()
        self.ll_decoder = UNetDecoder()

    def forward(self, x):
        # n,3,h,w
        res_x = x

        fE_out, enc_out = self.ll_encoder(x)
        x = self.ll_decoder(fE_out, enc_out, res_x)
        
        ll, detail9, _ = self.dwt(x)

        detail3 = detail9[:, 0:3, :, :] + detail9[:, 3:6, :, :] + detail9[:, 6:9, :, :]

        output = x 

        return ll, detail3, output

class MMDS(nn.Module):
    def __init__(self,
                 en_feature_num,
                 en_inter_num,
                 de_feature_num,
                 de_inter_num,
                 sam_number=1,
                 ):
        super(MMDS, self).__init__()
        self.encoder = Encoder(feature_num=en_feature_num, inter_num=en_inter_num, sam_number=sam_number)
        self.decoder = Decoder(en_num=en_feature_num, feature_num=de_feature_num, inter_num=de_inter_num,
                               sam_number=sam_number)

        self.cross_attention1 = Cross_Attention(
                                            in_channel_x=en_feature_num,
                                            in_channel_y=en_feature_num * 2,
                                            features=en_feature_num,
                                            grid_size=[16, 16], 
                                            block_size=[16, 16],
                                            inter_num=en_inter_num)

        self.cross_attention2 = Cross_Attention(
                                            in_channel_x=en_feature_num * 2,
                                            in_channel_y=en_feature_num * 4,
                                            features=en_feature_num * 2,
                                            grid_size=[8, 8],
                                            block_size=[8, 8],
                                            inter_num=en_inter_num)

    def forward(self, x):
        y_1, y_2, y_3 = self.encoder(x)
        # [2, 48, 256, 256],  [2, 96, 128, 128],  [2, 192, 64, 64]

        aca_y_1, aca_y_2_1 = self.cross_attention1(y_1, y_2)
        aca_y_2, aca_y_3   = self.cross_attention2(aca_y_2_1, y_3)

        out_1, out_2, out_3 = self.decoder(aca_y_1, aca_y_2, aca_y_3)

        return out_1, out_2, out_3
    
    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.normal_(0.0, 0.02)
                if isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0.0, 0.02)

class Decoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, sam_number):
        super(Decoder, self).__init__()
        self.preconv_3 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_2 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_1 = conv_relu(    en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = Decoder_Level(feature_num, inter_num, sam_number)

    def forward(self, y_1, y_2, y_3):
        x_3 = y_3
        x_3 = self.preconv_3(x_3)
        out_3, feat_3 = self.decoder_3(x_3)

        x_2 = torch.cat([y_2, feat_3], dim=1)
        x_2 = self.preconv_2(x_2)
        out_2, feat_2 = self.decoder_2(x_2)

        x_1 = torch.cat([y_1, feat_2], dim=1)
        x_1 = self.preconv_1(x_1)
        out_1 = self.decoder_1(x_1, feat=False)

        return out_1, out_2, out_3


class Encoder(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Encoder, self).__init__()

        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )

        #encoder_level * 3
        self.encoder_1 = Encoder_Level(    feature_num, inter_num, level=1, sam_number=sam_number)
        self.encoder_2 = Encoder_Level(2 * feature_num, inter_num, level=2, sam_number=sam_number)
        self.encoder_3 = Encoder_Level(4 * feature_num, inter_num, level=3, sam_number=sam_number)

    def forward(self, x):

        # 3 --- 12
        x = F.pixel_unshuffle(x, 2)

        # 3 --- fn
        x = self.conv_first(x)

        # 输出特征和下一级输入特征
        out_feature_1, down_feature_1 = self.encoder_1(x)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1)
        out_feature_3 = self.encoder_3(down_feature_2)

        return out_feature_1, out_feature_2, out_feature_3


class Encoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number):
        super(Encoder_Level, self).__init__()

        # Casaded RDB and RDiB
        self.crr = CRR(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num, )

        # Layer Fusion
        self.lay_fus = nn.ModuleList()
        fusion_block = Layer_Fusion(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
        self.lay_fus.append(fusion_block)

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.level = level

    def forward(self, x):

        #级联的RDB和RDiB
        crr_out = self.crr(x)

        for lay_fu in self.lay_fus:
            out_feature = lay_fu(crr_out)

        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature

        return out_feature

class Decoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Decoder_Level, self).__init__()

        self.crr = CRR(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)

        self.lay_fus = nn.ModuleList()
        for _ in range(sam_number):
            fusion_block = Layer_Fusion(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.lay_fus.append(fusion_block)
        self.conv = conv(in_channel=feature_num, out_channel=12, kernel_size=3, padding=1)

    def forward(self, x, feat=True):

        x = self.crr(x)

        for lay_fu in self.lay_fus:
            x = lay_fu(x)

        out = self.conv(x)

        # 亚像素卷积放大两倍
        out = F.pixel_shuffle(out, 2)

        if feat:
            feature = F.interpolate(x, scale_factor=2, mode='bilinear')
            return out, feature
        else:
            return out


class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class RDB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(RDB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t

class Layer_Fusion(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(Layer_Fusion, self).__init__()
        self.basic_block = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.fusion = Fusion(3 * in_channel)

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = self.fusion(y_0, y_2, y_4)
        y = x + y

        return y

class Fusion(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(Fusion, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)
        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)

        x = x0 * w0 + x2 * w2 + x4 * w4

        return x

class CRR(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(CRR, self).__init__()
        self.d_list = d_list

        self.rdb = nn.ModuleList()
        self.rdib = nn.ModuleList()

        c = in_channel
        for i in range(len(d_list)):

            dense_conv_rdb = conv_leakyrelu(in_channel=c, out_channel=inter_num, kernel_size=3, padding=1)
            
            dense_conv_rdib = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            
            self.rdb.append(dense_conv_rdb)
            self.rdib.append(dense_conv_rdib)
            
            c = c + inter_num

        self.conv_post_rdb = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

        self.conv_post_rdib = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        res_x = x
        t = x

        # -------rdb------
        for rdb_layer in self.rdb:
            _t = rdb_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post_rdb(t)
        t = t + res_x

        # -------rdib------
        for rdib_layer in self.rdib:
            _t = rdib_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post_rdib(t)
        t = t + res_x

        return t


class CrossGating(nn.Module):
    """Get gating weights for cross-gating MLP block."""

    def __init__(self, in_channel, block_size, grid_size, input_proj_factor=2, dropout_rate=0.0, bias=True):
        super().__init__()
        self.in_channel = in_channel
        self.block_size = block_size
        self.grid_size = grid_size
        self.gh = self.grid_size[0]
        self.gw = self.grid_size[1]
        self.fh = self.block_size[0]
        self.fw = self.block_size[1]

        self.input_proj_factor = input_proj_factor
        self.dropout_rate = dropout_rate
        self.bias = bias
        self.Dense_0 = nn.Linear(self.gh*self.gw, self.gh*self.gw, bias = self.bias)
        self.Dense_1 = nn.Linear(self.fh*self.fw, self.fh*self.fw, bias = self.bias)
        self.out_project = nn.Linear(self.in_channel*self.input_proj_factor, self.in_channel, bias=self.bias)
        self.in_project = nn.Linear(self.in_channel, self.in_channel*self.input_proj_factor, bias=self.bias)
        self.gelu = nn.GELU()

        self.LayerNorm_in = Layer_norm_process(self.in_channel)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        _, h, w, _ = x.shape

        #input projection
        x = self.LayerNorm_in(x)
        x = self.in_project(x)  #channel projection
        x = self.gelu(x)
        c = x.size(-1)//2
        u, v = torch.split(x, c, dim=-1)

        #get grid MLP weights
        fh, fw = h//self.gh, w//self.gw
        u = block_images_einops(u, patch_size = (fh, fw))   #n, (gh gw) (fh fw) c
        u = u.permute(0,3,2,1)  #n, c, (fh fw) (gh gw)
        u = self.Dense_0(u)
        u = u.permute(0,3,2,1)  #n, (gh gw) (fh fw) c
        u = unblock_images_einops(u, grid_size=(self.gh, self.gw), patch_size=(fh, fw))
        #get block MLP weights
        gh, gw = h//self.fh, w//self.fw
        v = block_images_einops(v, patch_size=(self.fh, self.fw))   #n, (gh gw) (fh fw) c
        v = v.permute(0,1,3,2)  #n (gh gw) c (fh fw)
        v = self.Dense_1(v)
        v = v.permute(0,1,3,2)  #n, (gh gw) (fh fw) c
        v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(self.fh, self.fw))
        
        x = torch.cat([u,v], dim=-1)
        x = self.out_project(x)
        x = self.dropout(x)

        return x

class Cross_Attention(nn.Module):
    def __init__(self, in_channel_x, in_channel_y, features, grid_size, block_size, dropout_rate=0.0,
                 input_proj_factor=2, upsample_y=True, bias=True, inter_num=32):
        super().__init__()
        self.in_channel_x = in_channel_x

        self.in_channel_y = in_channel_y

        self.features = features

        self.grid_size = grid_size
        self.block_size = block_size
        self.dropout_rate = dropout_rate
        self.input_proj_factor = input_proj_factor
        self.upsample_y = upsample_y
        self.bias = bias
        self.inter_num = inter_num
        #1x1 conv
        self.conv1_x = Conv1x1(self.in_channel_x, self.features, bias=self.bias)
        self.conv1_y_in = Conv1x1(self.in_channel_y, self.features, bias=self.bias)
        self.conv1_y_out = Conv1x1(self.features, self.in_channel_y, bias=self.bias)
        #MLP layer
        self.in_linear_x = nn.Linear(self.features, self.features, bias=self.bias)
        self.in_linear_y = nn.Linear(self.features, self.features, bias=self.bias)
        self.out_linear_x = nn.Linear(self.features, self.features, bias=self.bias)
        self.out_linear_y = nn.Linear(self.features, self.features, bias=self.bias)
        #Cross Gating
        self.getspatialgatingweights_x = CrossGating(
            in_channel=self.features,
            block_size=self.block_size,
            grid_size=self.grid_size,
            dropout_rate=self.dropout_rate,
            bias=self.bias)
        
        self.getspatialgatingweights_y = CrossGating(
            in_channel=self.features,
            block_size=self.block_size,
            grid_size=self.grid_size,
            dropout_rate=self.dropout_rate,
            bias=self.bias)

        self.LayerNorm_x = Layer_norm_process(self.features)
        self.LayerNorm_y = Layer_norm_process(self.features)

        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.rdb_x = RDB(in_channel=in_channel_x, d_list=(1, 2, 1), inter_num=self.inter_num)
        self.rdb_y = RDB(in_channel=in_channel_x, d_list=(1, 2, 1), inter_num=self.inter_num)

    def forward(self, x, y):
        res_x = x
        res_y = y

        x = self.conv1_x(x)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        y = self.conv1_y_in(y)

        x = self.rdb_x(x)
        y = self.rdb_y(y)

        # (n,c,h,w) - (n,h,w,c)
        x = x.permute(0,2,3,1)  #n,h,w,c
        y = y.permute(0,2,3,1)  #n,h,w,c

        assert y.shape == x.shape
        
        # Get gating weights from x
        x = self.LayerNorm_x(x)
        x = self.in_linear_x(x)
        x = self.gelu1(x)
        gx = self.getspatialgatingweights_x(x)

        # Get gating weights from y
        y = self.LayerNorm_y(y)
        y = self.in_linear_y(y)
        y = self.gelu2(y)
        gy = self.getspatialgatingweights_y(y)

        # X = X * GY, Y = Y * GX
        y = y * gx
        y = self.out_linear_y(y)
        y = self.dropout1(y)

        x = x * gy  
        x = self.out_linear_x(x)
        x = self.dropout2(x)

        x = x.permute(0,3,1,2)
        y = y.permute(0,3,1,2)

        y = F.interpolate(y, scale_factor=0.5, mode='bilinear')
        y = self.conv1_y_out(y)

        x = x + res_x
        y = y + res_y

        #n,c,h,w
        return x, y  

class Layer_norm_process(nn.Module):  #n, h, w, c
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.c = c

    def forward(self, feature):
        device = feature.device
        self.beta = torch.nn.Parameter(torch.zeros(self.c).to(device), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones(self.c).to(device), requires_grad=True)

        var_mean = torch.var_mean(feature, dim=-1, unbiased=False)
        mean = var_mean[1]
        var = var_mean[0]
        # layer norm process
        feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + self.eps)
        gamma = self.gamma.expand_as(feature)
        beta = self.beta.expand_as(feature)
        feature = feature * gamma + beta
        return feature