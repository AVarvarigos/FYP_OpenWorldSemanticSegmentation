import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    """
    increase gamma to put more focus on hard, misclassified examples
    gamma must be >= 0
    alpha is a balance factor for positive and negative examples

    see https://github.com/clcarwin/focal_loss_pytorch/blob/e11e75bad957aecf641db6998a1016204722c1bb/focalloss.py
    """
    def __init__(self, gamma=2.0, alpha=0.25, size_average=True, void_label=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.void_label = void_label
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
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

        logpt = F.log_softmax(input_valid)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input_valid.data.type():
                self.alpha = self.alpha.type_as(input_valid.data)
            at = self.alpha.gather(0,target_valid.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
