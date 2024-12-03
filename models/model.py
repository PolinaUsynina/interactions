import torch
from torch import nn


class SLM(torch.nn.Module):
    def __init__(self):
        super().__init__()

        #define parameters
        self.R = 6371.

        #define layers
        self.linear_layers = nn.Sequential(  
            #big
            nn.Linear(7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features, pulsar_data = x
        time, RM_real, RM_real_stat, VTEC, Bpar, z, target1_real = torch.transpose(pulsar_data, 0, 1)
        
        #print("shape:", features.shape)
        target1 = 3*torch.sigmoid(self.linear_layers(features))
        
        #k = -1.898627945760582858e-01
        #b = 3.827098607097981358e+02
        
        #RM_computed = Bpar*VTEC*2.62e-6/(target1)**0.5
        #RM_computed_stat = torch.log(RM_computed) - k*time - b
        #return RM_computed_stat[0], RM_real
        return target1.squeeze(), target1_real

