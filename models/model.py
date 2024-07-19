import torch
from torch import nn


class SLM(torch.nn.Module):
    def __init__(self):
        super().__init__()

        #define parameters
        self.R = 6371.

        #define layers
        self.linear_layers = nn.Sequential(  
            nn.Linear(7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features, pulsar_data = x
        RM_real, VTEC, Bpar, z = torch.transpose(pulsar_data, 0, 1)
        H = self.linear_layers(features)
        RM_computed = Bpar*VTEC*2.62e-6/(1 - (self.R*torch.sin(z)/(self.R+H))**2)**0.5
        return RM_computed[0], RM_real

