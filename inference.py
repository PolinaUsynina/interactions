#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import torch
import yaml
from lightning.pytorch import Trainer
from sklearn.preprocessing import RobustScaler, StandardScaler

from datasets.inference import get_inference_dl_ds
from pl_models import TestPipeline


def load_config(config_path):
    with open(config_path, 'r') as input_file:
        config = yaml.safe_load(input_file)
    return config


def parse_args(args):
    parser = argparse.ArgumentParser(description='Template to inference networks with pytorch lightning.')
    parser.add_argument('config', help='path to yaml config file', default='configs/inference.yaml')
    return parser.parse_args(args)


def rebuild_scalers_from_checkpoint(weights_path):
    """Восстанавливаем RobustScaler(features) и StandardScaler(target) из ckpt."""
    ckpt = torch.load(weights_path, map_location='cpu')

    fs = RobustScaler()
    fs.center_ = np.asarray(ckpt['feature_center'])
    fs.scale_  = np.asarray(ckpt['feature_scale'])
    fs.n_features_in_ = fs.center_.size

    ts = StandardScaler()
    ts.mean_  = np.asarray([ckpt['target_mean']])
    ts.scale_ = np.asarray([ckpt['target_scale']])
    ts.var_   = ts.scale_ ** 2
    ts.n_features_in_ = 1

    return fs, ts


def inference(args=None):
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)
        config = load_config(args.config)
    else:
        config = args

    feature_scaler, target_scaler = rebuild_scalers_from_checkpoint(config['weights'])

    dataloader, dataset = get_inference_dl_ds(
        config,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )

    model = TestPipeline(config)

    tester = Trainer(
        logger=False, **config['trainer']
    )
    tester.test(model, dataloaders=[dataloader])


if __name__ == "__main__":
    inference()