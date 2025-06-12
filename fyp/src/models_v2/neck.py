#!/usr/bin/env python

# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models_v2.model_utils import ConvBNAct


__all__ = [
    "PyramidPoolingModule",
    "AdaptivePyramidPoolingModule",
    "SqueezeAndExcitation",
]


class PyramidPoolingModule(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bins=(1, 2, 3, 6),
        activation=nn.ReLU(inplace=True),
        upsampling_mode="bilinear",
    ):
        reduction_dim = in_dim // len(bins)
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        self.upsampling_mode = upsampling_mode
        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    ConvBNAct(
                        in_dim, reduction_dim, kernel_size=1, activation=activation
                    ),
                )
            )
        in_dim_last_conv = in_dim + reduction_dim * len(bins)
        self.features = nn.ModuleList(self.features)

        self.final_conv = ConvBNAct(
            in_dim_last_conv, out_dim, kernel_size=1, activation=activation
        )

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            h, w = x_size[2:]
            y = f(x)
            if self.upsampling_mode == "nearest":
                out.append(F.interpolate(y, (int(h), int(w)), mode="nearest"))
            elif self.upsampling_mode == "bilinear":
                out.append(
                    F.interpolate(
                        y, (int(h), int(w)), mode="bilinear", align_corners=False
                    )
                )
            else:
                raise NotImplementedError(
                    "For the PyramidPoolingModule only nearest and bilinear "
                    "interpolation are supported. "
                    f"Got: {self.upsampling_mode}"
                )
        out = torch.cat(out, 1)
        out = self.final_conv(out)
        return out


class AdaptivePyramidPoolingModule(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        input_size,
        bins=(1, 2, 3, 6),
        activation=nn.ReLU(inplace=True),
        upsampling_mode="bilinear",
    ):
        reduction_dim = in_dim // len(bins)
        super(AdaptivePyramidPoolingModule, self).__init__()
        self.features = []
        self.upsampling_mode = upsampling_mode
        self.input_size = input_size
        self.bins = bins
        for _ in bins:
            self.features.append(
                ConvBNAct(in_dim, reduction_dim, kernel_size=1, activation=activation)
            )
        in_dim_last_conv = in_dim + reduction_dim * len(bins)
        self.features = nn.ModuleList(self.features)

        self.final_conv = ConvBNAct(
            in_dim_last_conv, out_dim, kernel_size=1, activation=activation
        )

    def forward(self, x):
        x_size = x.size()
        h, w = x_size[2:]
        h_inp, w_inp = self.input_size
        bin_multiplier_h = int((h / h_inp) + 0.5)
        bin_multiplier_w = int((w / w_inp) + 0.5)
        out = [x]
        for f, bin in zip(self.features, self.bins):
            h_pool = bin * bin_multiplier_h
            w_pool = bin * bin_multiplier_w
            pooled = F.adaptive_avg_pool2d(x, (h_pool, w_pool))
            y = f(pooled)
            if self.upsampling_mode == "nearest":
                out.append(F.interpolate(y, (int(h), int(w)), mode="nearest"))
            elif self.upsampling_mode == "bilinear":
                out.append(
                    F.interpolate(
                        y, (int(h), int(w)), mode="bilinear", align_corners=False
                    )
                )
            else:
                raise NotImplementedError(
                    "For the PyramidPoolingModule only nearest and bilinear "
                    "interpolation are supported. "
                    f"Got: {self.upsampling_mode}"
                )
        out = torch.cat(out, 1)
        out = self.final_conv(out)
        return out


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
