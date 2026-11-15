from torch import nn
import torch


class RMLoss(nn.Module):
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta)

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)