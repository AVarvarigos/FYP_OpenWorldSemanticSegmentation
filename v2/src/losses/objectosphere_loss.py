#!/usr/bin/env python

# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

import torch
from torch import nn
import torch.nn.functional as F

# This is training the model to do anomaly segmentation - it is being trained to differentiate between known and unknown classes
class ObjectosphereLoss(nn.Module):
    def __init__(self, sigma=1.0, void_label=-1):
        super().__init__()
        self.sigma = sigma
        self.void_label = void_label

    def forward(self, logits, sem_gt): # 0,1,2,3, 0, 2, 3, 1 B, C, H, W -> B, H, W, C
        logits_unk = logits.permute(0, 2, 3, 1)[sem_gt == self.void_label]
        logits_kn = logits.permute(0, 2, 3, 1)[sem_gt != self.void_label]

        loss_unk = (torch.linalg.norm(logits_unk, dim=1)).mean() if len(logits_unk) else torch.tensor(0.0).to(logits)
        loss_kn = (
            F.relu(self.sigma - (torch.linalg.norm(logits_kn, dim=1))).mean()
            if len(logits_kn)
            else torch.tensor(0.0).to(logits)
        )

        # TODO: maybe the "10" is problematic because it tries to send uknown to zero but does not do much into pushing
        # known to 1, thus not being able to differentiate between the two easily. Train with 1,2,4,8,16,32 here to see
        # what it does
        return 10 * loss_unk + loss_kn