#!/usr/bin/env python

# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

import torch
from torch import nn
import torch.nn.functional as F


class OWLoss(nn.Module):
    def __init__(self, n_classes, hinged=False, delta=0.1, void_label=-1):
        super().__init__()
        self.n_classes = n_classes
        self.hinged = hinged
        self.void_label = void_label
        self.delta = delta
        self.count = torch.zeros(self.n_classes).cuda()  # count for class
        self.features = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # for implementation of Welford Alg.
        self.ex = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.ex2 = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.var = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}

        self.criterion = torch.nn.L1Loss(reduction="none")

        self.previous_features = None
        self.previous_count = None

    @torch.no_grad()
    def cumulate(self, logits: torch.Tensor, sem_gt: torch.Tensor):
        sem_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        gt_labels = torch.unique(sem_gt).tolist()
        # 0,1,2,3, 0, 2, 3, 1 B, C, H, W -> B, H, W, C
        logits_permuted = logits.permute(0, 2, 3, 1)
        for label in gt_labels:
            if label == self.void_label:
                continue
            sem_gt_current = sem_gt == label
            sem_pred_current = sem_pred == label
            tps_current = torch.logical_and(sem_gt_current, sem_pred_current)
            if tps_current.sum() == 0:
                continue
            logits_tps = logits_permuted[torch.where(tps_current == 1)]
            # max_values = logits_tps[:, label].unsqueeze(1)
            # logits_tps = logits_tps / max_values
            avg_mav = torch.mean(logits_tps, dim=0)
            n_tps = logits_tps.shape[0]
            # features is running mean for mav
            self.features[label] = (self.features[label] * self.count[label] + avg_mav * n_tps)

            self.ex[label] += (logits_tps).sum(dim=0)
            self.ex2[label] += ((logits_tps) ** 2).sum(dim=0)
            self.count[label] += n_tps
            self.features[label] /= self.count[label] + 1e-8

    def forward(self, logits: torch.Tensor, sem_gt: torch.Tensor, is_train: bool) -> torch.Tensor:
        if is_train:
            # update mav only at training time
            sem_gt = sem_gt#.type(torch.uint8)
            self.cumulate(logits, sem_gt)
        if self.previous_features == None:
            return torch.tensor(0.0).cuda()
        gt_labels = torch.unique(sem_gt).tolist()

        logits_permuted = logits.permute(0, 2, 3, 1)

        acc_loss = torch.tensor(0.0).cuda()
        for label in gt_labels[1:]:
            mav = self.previous_features[label]
            logs = logits_permuted[torch.where(sem_gt == label)]
            mav = mav.expand(logs.shape[0], -1)
            if self.previous_count[label] > 0:
                ew_l1 = self.criterion(logs, mav)
                ew_l1 = ew_l1 / (self.var[label] + 1e-8)
                if self.hinged:
                    ew_l1 = F.relu(ew_l1 - self.delta).sum(dim=1)
                acc_loss += ew_l1.mean()

        return acc_loss

    def update(self):
        self.previous_features = self.features
        self.previous_count = self.count
        for c in self.var.keys():
            self.var[c] = (self.ex2[c] - self.ex[c] ** 2 / (self.count[c] + 1e-8)) / (self.count[c] + 1e-8)

        # resetting for next epoch
        self.count = torch.zeros(self.n_classes)  # count for class
        self.features = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.ex = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.ex2 = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}

        return self.previous_features, self.var

    def read(self):
        mav_tensor = torch.zeros(self.n_classes, self.n_classes)
        for key in self.previous_features.keys():
            mav_tensor[key] = self.previous_features[key]
        return mav_tensor