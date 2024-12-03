from torch import nn
import torch

        
class RMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, t1_computed, t1_real):
        
        residual = t1_computed - t1_real
        probabilities = torch.softmax(residual, dim=0)
        target_distribution = torch.full_like(probabilities, 1.0 / probabilities.size(0))
        kld_loss = nn.KLDivLoss(reduction='batchmean')(probabilities.log(), target_distribution)
        
        mse = nn.MSELoss()
        loss = 0.5*mse(t1_computed, t1_real) + 0.5*kld_loss
        return loss