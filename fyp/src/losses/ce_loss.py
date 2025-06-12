#!/usr/bin/env python

# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

import numpy as np
import torch
from torch import nn


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, device, weight, void_label=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.void_label = void_label
        self.weight = torch.tensor(weight).to(device)
        self.num_classes = len(self.weight) + 1  # +1 for void
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction="none",
            ignore_index=void_label,
        )
        self.ce_loss.to(device)

    def forward(self, inputs, targets):
        if (targets == -1).all():  # i.e. if all labels are void (-1)
            return torch.tensor(0.0).cuda()
        loss_all = self.ce_loss(inputs, targets.long())
        divisor_weighted_pixel_sum = (targets >= 0).sum()

        return torch.sum(loss_all) / divisor_weighted_pixel_sum


class CrossEntropyLoss2dForValidData:
    def __init__(self, device, weight, void_label=-1):
        super(CrossEntropyLoss2dForValidData, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction="sum",
            ignore_index=void_label,
        )
        self.ce_loss.to(device)
        self.total_loss = 0
        self.num_of_acc = 0

    def add_loss_of_batch(self, inputs, targets):
        self.num_of_acc += 1

        if (targets == -1).all():  # i.e. if all labels are void (-1)
            return

        loss = self.ce_loss(inputs, targets.long()) / (targets >= 0).sum()
        self.total_loss += loss

    def compute_whole_loss(self):
        return (
            (self.total_loss.detach().cpu().item() / self.num_of_acc)
            if self.num_of_acc != 0
            else 0
        )

    def reset_loss(self):
        self.total_loss = 0
        self.num_of_acc = 0
