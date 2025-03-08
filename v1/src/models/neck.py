#!/usr/bin/env python

# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["PyramidPoolingModule", "AdaptivePyramidPoolingModule", "SqueezeAndExcitation"]


class PyramidPoolingModule(nn.Module):
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            bins=(1, 2, 3, 6), 
            upsampling_mode="bilinear", 
            activation=nn.ReLU(inplace=True),
            ):
        super(PyramidPoolingModule, self).__init__()
        self.upsampling_mode = upsampling_mode
        reduction_dim = in_dim // len(bins)
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(bin) for bin in bins])
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                activation,
            ) for _ in bins
        ])
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_dim + reduction_dim * len(bins), out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            activation,
        )

    def forward(self, x):
        x_size = x.size()
        output = [x]
        for pool, conv in zip(self.pools, self.convs):
            y = pool(x)
            y = conv(y)
            y = F.interpolate(y, size=(x_size[2], x_size[3]), 
                              mode=self.upsampling_mode, 
                              align_corners=False if self.upsampling_mode == "bilinear" else None)
            output.append(y)
        output = torch.cat(output, 1)
        output = self.final_conv(output)
        return output
    
    
class AdaptivePyramidPoolingModule(nn.Module):
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            input_size, 
            bins=(1, 2, 3, 6), 
            upsampling_mode="bilinear",
            activation=nn.ReLU(inplace=True),
            ):
        super(AdaptivePyramidPoolingModule, self).__init__()
        self.input_size = input_size
        self.bins = bins
        self.upsampling_mode = upsampling_mode
        reduction_dim = in_dim // len(bins)
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d((bin * int((input_size[0] / input_size[0]) + 0.5), 
                                                          bin * int((input_size[1] / input_size[1]) + 0.5))) for bin in bins])
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                activation,
            ) for _ in bins
        ])
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_dim + reduction_dim * len(bins), out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            activation,
        )

    def forward(self, x):
        x_size = x.size()
        h, w = x_size[2:]
        output = [x]
        for pool, conv in zip(self.pools, self.convs):
            y = pool(x)
            y = conv(y)
            y = F.interpolate(y, size=(h, w), 
                              mode=self.upsampling_mode, 
                              align_corners=False if self.upsampling_mode == "bilinear" else None)
            output.append(y)
        output = torch.cat(output, 1)
        output = self.final_conv(output)
        return output


class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_dim, reduction=16):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // reduction, in_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y

if __name__ == '__main__':
    x = torch.randn(2, 2048, 7, 7)
        
    ppm = PyramidPoolingModule(in_dim=2048, out_dim=256)
    ppm_output = ppm(x)
    print("Output shape of PyramidPoolingModule:", ppm_output.shape)
    
    appm = AdaptivePyramidPoolingModule(in_dim=2048, out_dim=256, input_size=(7, 7))
    appm_output = appm(x)
    print("Output shape of AdaptivePyramidPoolingModule:", appm_output.shape)
    
    se = SqueezeAndExcitation(in_dim=2048, reduction=16)
    se_output = se(x)
    print("Output shape of SqueezeAndExcitation:", se_output.shape)


