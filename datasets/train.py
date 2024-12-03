import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    def __init__(self, config, mode='train'):
        super().__init__()
        self.config = config
        
        self.mode = mode
              
        csv_path = os.path.join(
            config["data_path"],
            self.config['train'] if (mode == 'train') else self.config['test']
        )
        
        self.dataset = self._read_csv(csv_path)
        

    def _read_csv(self, csv_path):
        df = pd.read_csv(csv_path, sep=' ', names=['X-ray B', 'SYM-H', 'F10.7', 'LST', 'SML', 'SMU', 'SMR', 'time', 'RM_real', 'RM_real_stat', 'VTEC', 'Bpar', 'z', 'target1'])
        return df

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        pulsar_cols = ['time', 'RM_real', 'RM_real_stat', 'VTEC', 'Bpar', 'z', 'target1']
        features_cols = [col for col in self.dataset.columns if col not in pulsar_cols]
        
        pulsar_dataset = self.dataset[pulsar_cols].copy()
        
        if self.mode == 'train':
            noise_factor = 0.15
            pulsar_dataset.loc[:, 'target1'] += noise_factor * np.random.randn(len(pulsar_dataset['target1']))
        
        pulsar_data_arr = pulsar_dataset.iloc[idx].to_numpy()
        features_arr = self.dataset[features_cols].iloc[idx].to_numpy()
        
        pulsar_data = torch.from_numpy(pulsar_data_arr).to(torch.float32) 
        features = torch.from_numpy(features_arr).to(torch.float32)
        
        if self.mode == 'train':
            noise_factor = 0.3
            features += noise_factor * torch.randn_like(features)
            idx = torch.as_tensor(idx).long()
        
        return {'features': features, 'pulsar_data': pulsar_data, 'idx': idx}


def collate_fn(batch):
    features = torch.stack([b['features'] for b in batch], dim=0)
    pulsar_data = torch.stack([b['pulsar_data'] for b in batch], dim=0)
    return features, pulsar_data


def get_train_dl_ds(
        config,
        mode='train'
):
    dataset = TrainDataset(
        config, mode=mode
    )

    dataloader = DataLoader(
        dataset,
        shuffle=(mode == "train"),
        collate_fn=collate_fn,
        **config['dataloader']
    )
    
    return dataloader, dataset


