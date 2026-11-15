import os

import torch
from torch.utils.data import DataLoader

from datasets.train import TrainDataset


class TestDataset(TrainDataset):
    """Dataset для инференса: только transform, без re-fit scaler-ов."""
    def __init__(self, config, feature_scaler=None, target_scaler=None):
        super().__init__(
            config, mode='test',
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
        )


def collate_fn(batch):
    features = torch.stack([b['features'] for b in batch], dim=0)
    pulsar_data = torch.stack([b['pulsar_data'] for b in batch], dim=0)
    return features, pulsar_data


def get_inference_dl_ds(config, feature_scaler=None, target_scaler=None):
    dataset = TestDataset(
        config,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn,
        **config['dataloader']
    )

    return dataloader, dataset