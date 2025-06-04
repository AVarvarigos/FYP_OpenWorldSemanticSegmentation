import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-5

    def forward(self, pred, target):
        """
        Computes Dice Loss for multi-class segmentation.
        Args:
            pred: Tensor of predictions (batch_size, C, H, W).
            target: One-hot encoded ground truth (batch_size, C, H, W).
            smooth: Smoothing factor.
        Returns:
            Scalar Dice Loss.

        https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
        """
        pred = F.softmax(pred, dim=1)  # Convert logits to probabilities
        num_classes = pred.shape[1]  # Number of classes (C)
        dice = 0  # Initialize Dice loss accumulator

        # starts from 0, void_class is -1, so we need to loop through all known classes
        for c in range(num_classes):  # Loop through each class
            pred_c = pred[:, c]  # Predictions for class c
            target_c = target == c  # One-hot encoded target for class c

            intersection = (pred_c * target_c).sum(dim=(1, 2))  # Element-wise multiplication
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))  # Sum of all pixels

            dice += (2. * intersection + self.smooth) / (union + self.smooth)  # Per-class Dice score

        return 1 - dice.mean() / num_classes  # Average Dice Loss across classes
