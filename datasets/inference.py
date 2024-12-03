import os

import torch
from torch.utils.data import DataLoader

from datasets.train import TrainDataset


class TestDataset(TrainDataset):
    def __init__(self, config, mode='test'):
        super().__init__(config, mode='test')
        self.mode = mode
        self.config = config
              
        csv_path = os.path.join(
            config["data_path"],
            self.config['test']
        )
        
        self.dataset = self._read_csv(csv_path)


def collate_fn(batch):
    features = torch.stack([b['features'] for b in batch], dim=0)
    pulsar_data = torch.stack([b['pulsar_data'] for b in batch], dim=0)
    return features, pulsar_data


def get_inference_dl_ds(config):
    dataset = TestDataset(config)

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn,
        **config['dataloader']
    )
    
    return dataloader, dataset
