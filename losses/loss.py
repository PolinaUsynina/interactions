from torch import nn
import torch

        
class RMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, RM_computed, RM_real):
        loss = torch.mean(torch.abs(RM_computed - RM_real))
        return loss