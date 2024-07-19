import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    def __init__(self, config, mode='train'):
        super().__init__()
        self.config = config
        
        self.mode = mode
              
        csv_path = os.path.join(
            config["data_path"],
            self.config['train'] if (mode == 'train') else self.config['val']
        )
        
        self.dataset = self._read_csv(csv_path)
        

    def _read_csv(self, csv_path):
        df = pd.read_csv(csv_path, sep=' ', names=['X-ray B', 'SYM-H', 'Sunspots', 'F10.7', 'DST', 'Kp', 'LST', 'RM_real', 'VTEC', 'Bpar', 'z'])
        return df

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        pulsar_cols = ['RM_real', 'VTEC', 'Bpar', 'z']
        features_cols = [col for col in self.dataset.columns if col not in pulsar_cols]
        
        pulsar_data_arr = self.dataset[pulsar_cols].iloc[idx].to_numpy()
        features_arr = self.dataset[features_cols].iloc[idx].to_numpy()
        
        pulsar_data = torch.from_numpy(pulsar_data_arr).to(torch.float32)
        features = torch.from_numpy(features_arr).to(torch.float32)
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


