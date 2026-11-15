import torch
from torch import nn


class PulsarNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        #define layers
        self.trend_layer = nn.Linear(11, 1)
        self.main_layers = nn.Sequential(  
            
            nn.Linear(11, 256),
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
        
        RM_real = pulsar_data[:, 0]   
        target_real = pulsar_data[:, 1]   

        trend    = self.trend_layer(features)     
        residual = self.main_layers(features)   
        pred = (trend + residual).squeeze(-1)   

        return pred, target_real

