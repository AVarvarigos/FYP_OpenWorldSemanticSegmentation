#!/usr/bin/env python

# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, n_classes=19, void_label=-1):
        super().__init__()
        self.n_classes = n_classes
        self.void_label = void_label

    def forward(self, emb_k, emb_q, labels, epoch, tau=0.1):
        """
        emb_k: the feature bank with the aggregated embeddings over the iterations
        emb_q: the embeddings for the current iteration
        labels: the correspondent class labels for each sample in emb_q
        """
        if not epoch:
            return torch.tensor(0.0).cuda()
    
        total_loss = torch.tensor(0.0).cuda()
        assert (emb_q.shape[0] == labels.shape[0]), "mismatch on emb_q and labels shapes!"
        emb_k = F.normalize(emb_k, dim=-1)
        emb_q = F.normalize(emb_q, dim=1)
    
        for i, emb in enumerate(emb_q):
            label = labels[i]
            if not (self.void_label in label.unique() and len(label.unique()) == 1):
                label[label == self.void_label] = self.n_classes
                label_sq = torch.unique(label, return_inverse=True)[1]
                oh_label = (F.one_hot(label_sq)).unsqueeze(-2)  # one hot labels
                count = oh_label.view(-1, oh_label.shape[-1]).sum(
                    dim=0
                )  # num of pixels per cl
                pred = emb.permute(1, 2, 0).unsqueeze(-1)
                oh_pred = (
                    pred * oh_label
                )  # (H, W, Nc, Ncp) Ncp num classes present in the label
                oh_pred_flatten = oh_pred.view(
                    oh_pred.shape[0] * oh_pred.shape[1],
                    oh_pred.shape[2],
                    oh_pred.shape[3],
                )
                res_raw = oh_pred_flatten.sum(dim=0) / count  # avg feat per class
                res_new = (res_raw[~res_raw.isnan()]).view(
                    -1, self.n_classes
                )  # filter out nans given by intermediate classes (present because of oh)
                label_list = label.unique()
                if self.n_classes in label_list:
                    label_list = label_list[:-1]
                    res_new = res_new[:-1, :]

                # temperature-scaled cosine similarity
                final = (res_new.cuda() @ emb_k.T.cuda()) / 0.1

                loss = F.cross_entropy(final, label_list)
                total_loss += loss

        return total_loss / emb_q.shape[0]
