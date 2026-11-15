import os

import numpy as np
import lightning as L
import pandas as pd
import torch
from torch import optim
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from transformers import get_cosine_schedule_with_warmup

from losses.loss import RMLoss
from models.model import PulsarNet


class TrainPipeline(L.LightningModule):
    def __init__(self, config, train_loader, val_loader) -> None:
        super().__init__()
        self.config = config

        self.model = PulsarNet()
        
        if config['weights'] is not None:
            state_dict = torch.load(config['weights'], map_location='cpu')['state_dict']
            self.load_state_dict(state_dict, strict=True)
            
        self.criterion = RMLoss()
        metrics = MetricCollection([MeanAbsoluteError()])
        self.train_metrics = metrics.clone(postfix="/train")
        self.valid_metrics = metrics.clone(postfix="/val")

        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.num_training_steps = len(self.train_loader)

        # --- НОВОЕ: параметры target-скейлера для inverse_transform метрик ---
        # Скаляры, не тензоры-буферы -> не ломают state_dict при strict=True
        ts = train_loader.dataset.target_scaler
        self.target_mean = float(ts.mean_[0])
        self.target_scale = float(ts.scale_[0])

        self.save_hyperparameters(config)

    def _unscale(self, x):
        """Перевод предсказаний/таргетов из σ-единиц в физ. единицы."""
        return x * self.target_scale + self.target_mean

    def configure_optimizers(self):
        if self.config['optimizer'] == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.config['optimizer_params']
            )
        elif self.config['optimizer'] == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                momentum=0.9, nesterov=True,
                **self.config['optimizer_params']
            )
        elif self.config['optimizer'] == "LBFGS":
            optimizer = optim.LBFGS(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.config['optimizer_params']
            )
        else:
            raise ValueError(f"Unknown optimizer name: {self.config['optimizer']}")

        scheduler_params = self.config['scheduler_params']
        if self.hparams.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=scheduler_params['patience'],
                min_lr=scheduler_params['min_lr'],
                factor=scheduler_params['factor'],
                mode=scheduler_params['mode'],
                verbose=scheduler_params['verbose'],
            )
            lr_scheduler = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': scheduler_params['target_metric']
            }
        elif self.config['scheduler'] == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.num_training_steps * scheduler_params['warmup_epochs'],
                num_training_steps=int(self.num_training_steps * self.config['trainer']['max_epochs'])
            )
            lr_scheduler = {
                'scheduler': scheduler,
                'interval': 'step'
            }
        else:
            raise ValueError(f"Unknown scheduler name: {self.config['scheduler']}")

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_idx):
        computed, real = self.model(batch)
        loss = self.criterion(computed, real)        # loss в scaled-единицах

        self.log("Loss/train", loss, prog_bar=True)
        # --- НОВОЕ: метрика в физ. единицах ---
        self.train_metrics.update(self._unscale(computed), self._unscale(real))
        return loss

    def validation_step(self, batch, batch_idx):
        computed, real = self.model(batch)
        loss = self.criterion(computed, real)

        self.log("Loss/val", loss, prog_bar=True)
        # --- НОВОЕ: метрика в физ. единицах ---
        self.valid_metrics.update(self._unscale(computed), self._unscale(real))

    def on_train_epoch_end(self):
        train_metrics = self.train_metrics.compute()
        self.log_dict(train_metrics)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        valid_metrics = self.valid_metrics.compute()
        self.log_dict(valid_metrics)
        self.valid_metrics.reset()

    # --- НОВОЕ: сохраняем target-скейлер в checkpoint вместе с весами ---
    def on_save_checkpoint(self, checkpoint):
        checkpoint['target_mean'] = self.target_mean
        checkpoint['target_scale'] = self.target_scale
        
        # --- НОВОЕ: сохраняем параметры feature-скейлера (нужны для inference) ---
        fs = self.train_loader.dataset.feature_scaler
        checkpoint['feature_center'] = fs.center_.tolist()
        checkpoint['feature_scale']  = fs.scale_.tolist()


class TestPipeline(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = PulsarNet()
        ckpt = torch.load(config['weights'], map_location='cpu')
        self.load_state_dict(ckpt['state_dict'], strict=True)

        # --- НОВОЕ: восстанавливаем target-скейлер из checkpoint ---
        self.target_mean = float(ckpt['target_mean'])
        self.target_scale = float(ckpt['target_scale'])

        self.test_outputs = []

    def _unscale(self, x):
        return x * self.target_scale + self.target_mean

    def sync_across_gpus(self, tensors):
        tensors = self.all_gather(tensors)
        return torch.cat([t for t in tensors])

    def test_step(self, batch, batch_idx):
        computed, real = self.model(batch)
        
        self.test_outputs.append({
            "computed": self._unscale(computed).cpu().detach().numpy(),
            "real":     self._unscale(real).cpu().detach().numpy(),
        })

    def on_test_epoch_end(self):
        all_test_outputs = self.all_gather(self.test_outputs)

        if self.trainer.is_global_zero:
            computed = torch.cat(
                [o['computed'] for o in all_test_outputs], dim=0
            ).cpu().detach().numpy()
            real = torch.cat(
                [o['real'] for o in all_test_outputs], dim=0
            ).cpu().detach().numpy()

            file_path = os.path.join(
                self.config['save_path'],
                self.config['test_name'],
                "predictions.txt",
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            
            np.savetxt(str(file_path), np.column_stack([computed, real]),
                       header="computed real", comments="")