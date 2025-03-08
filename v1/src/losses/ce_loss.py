#!/usr/bin/env python

# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, device, weight, void_label=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.void_label = void_label
        self.weight = torch.tensor(weight).to(device)
        self.num_classes = len(self.weight) + 1  # +1 for void
        # if self.num_classes < 2**8:
        #     self.dtype = torch.uint8
        # else:
        #     self.dtype = torch.int16
        # IMPORTANT: ignore index -1 (the void)
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(), reduction="none", ignore_index=void_label)
        self.ce_loss.to(device)

    def forward(self, inputs, targets):
        # targets_m = targets.clone()
        if (targets == -1).all(): # i.e. if all labels are void (-1)
            return [torch.tensor(0.0).cuda()]
            # import ipdb;ipdb.set_trace()  # fmt: skip
        # targets_m -= 1
        # assert (targets - 1 >= 0).all()
        # print("(targets-1).unique", (targets-1).unique())
        # print("inputs.shape", inputs.shape)
        # print(targets)
        # print(inputs)
        loss_all = self.ce_loss(inputs, targets)
        # if (self.weight == 1).all():
        divisor_weighted_pixel_sum = (targets >= 0).sum()
        # else:
            # number_of_pixels_per_class = torch.bincount(targets.flatten() - 1, minlength=self.num_classes)
            # divisor_weighted_pixel_sum = torch.sum(number_of_pixels_per_class[1:] * self.weight) # without void

        return [torch.sum(loss_all) / divisor_weighted_pixel_sum]


class CrossEntropyLoss2dForValidData:
    def __init__(self, device, weight, weighted_pixel_sum, void_label=-1):
        super(CrossEntropyLoss2dForValidData, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(), reduction="sum", ignore_index=void_label)
        self.ce_loss.to(device)
        self.weighted_pixel_sum = weighted_pixel_sum
        self.total_loss = 0

    def add_loss_of_batch(self, inputs, targets):
        # targets_m = targets.clone()
        # targets_m -= 1
        loss = self.ce_loss(inputs, targets)
        self.total_loss += loss

    def compute_whole_loss(self):
        return self.total_loss.detach().cpu().numpy().item() / self.weighted_pixel_sum.item()

    def reset_loss(self):
        self.total_loss = 0


class CrossEntropyLoss2dForValidDataUnweighted:
    def __init__(self, device, void_label=-1):
        super(CrossEntropyLoss2dForValidDataUnweighted, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=None, reduction="sum", ignore_index=void_label)
        self.ce_loss.to(device)
        self.nr_pixels = 0
        self.total_loss = 0

    def add_loss_of_batch(self, inputs, targets):
        # targets_m = targets.clone()
        # targets_m -= 1
        loss = self.ce_loss(inputs, targets)
        self.total_loss += loss
        self.nr_pixels += torch.sum(targets >= 0) # only non void pixels

    def compute_whole_loss(self):
        return (
            self.total_loss.detach().cpu().numpy().item()
            / self.nr_pixels.detach().cpu().numpy().item()
        )

    def reset_loss(self):
        self.total_loss = 0
        self.nr_pixels = 0