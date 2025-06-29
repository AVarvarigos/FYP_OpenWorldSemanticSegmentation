import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, size_average=True, void_label=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.void_label = void_label
        self.alpha = torch.FloatTensor(alpha)
        self.size_average = size_average

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # input shape: N,C,H,W and target shape: N,H,W
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1,1)

        # Flatten target: N,H,W => N*H*W
        target = target.view(-1)
        # Create mask for valid (non-void) pixels
        valid_mask = target != self.void_label

        # Filter out void targets
        input_valid = input[valid_mask]
        target_valid = target[valid_mask].unsqueeze(1)

        logpt = F.log_softmax(input_valid, dim=-1)
        logpt = logpt.gather(1,target_valid)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            alpha = self.alpha.to(input_valid.device)
            at = alpha.gather(0,target_valid.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
