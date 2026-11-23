import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, StandardScaler

FEATURES = ['F10.7', 'LST_sin', 'LST_cos',
            'SML', 'SMU', 'SMR', 'sin_z2', 'day_sin', 'day_cos', 'VTEC', 'b_parallel']
TARGET = 'target3'
PULSAR_COLS = ['RM_real', TARGET]


class TrainDataset(Dataset):
    def __init__(self, config, mode='train', feature_scaler=None, target_scaler=None):
        super().__init__()
        self.config = config
        
        self.mode = mode
              
        ## train list
        self.file_list = self.config['train'] if (mode == 'train') else self.config[mode]
        if isinstance(self.file_list, str):
            self.file_list = [self.file_list]
            
        self.dataset = self._read_csv(self.file_list)
        ###
        
        ###  SCALING   ###
        if feature_scaler is None:
            # mode == 'train' -> fit + transform
            self.feature_scaler = RobustScaler().fit(
                self.dataset[FEATURES].to_numpy())
            self.target_scaler = StandardScaler().fit(
                self.dataset[[TARGET]].to_numpy())
        else:
            # val/test -> transform only, parameters from train
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler

        self.dataset[FEATURES] = self.feature_scaler.transform(
            self.dataset[FEATURES].to_numpy())
        self.dataset[[TARGET]] = self.target_scaler.transform(
            self.dataset[[TARGET]].to_numpy())
        

    #def _read_csv(self, csv_path):
    #    df = pd.read_csv(csv_path, sep=' ')
    #    return df
    
    def _read_csv(self, csv_paths):
        dfs = []
        for path in csv_paths:
            full_path = os.path.join(self.config["data_path"], path)
            df = pd.read_csv(full_path, sep=' ')
            dfs.append(df)
        
        merged_df = pd.concat(dfs, axis=0, ignore_index=True)
        return merged_df

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # pulsar_data = [RM_real, TARGET] 
        pulsar_dataset = self.dataset[PULSAR_COLS].copy()

        if self.mode == 'train':
            noise_factor = 0.05
            pulsar_dataset[TARGET] = pulsar_dataset[TARGET] + noise_factor * np.random.randn(len(pulsar_dataset[TARGET]))

        pulsar_data_arr = pulsar_dataset.iloc[idx].to_numpy()
        features_arr = self.dataset[FEATURES].iloc[idx].to_numpy()

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
        mode='train',
        feature_scaler=None,
        target_scaler=None,
):
    dataset = TrainDataset(
        config, mode=mode,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )

    dataloader = DataLoader(
        dataset,
        shuffle=(mode == "train"),
        collate_fn=collate_fn,
        **config['dataloader']
    )

    return dataloader, dataset
