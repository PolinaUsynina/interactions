#!/usr/bin/env python3
"""
Smoke test для пайплайна обучения. 2 эпохи на CPU, полная проверка.
Удаляет свой save_path в конце.
"""
import os
import shutil
import sys

import numpy as np
import torch
import yaml


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def expect(cond, msg):
    if not cond:
        print(f"❌ FAIL: {msg}")
        sys.exit(1)
    print(f"  ✓ {msg}")


# ============================================================
section("STEP 1: Imports")
try:
    from datasets.train import TrainDataset, get_train_dl_ds, FEATURES, TARGET, PULSAR_COLS
    print(f"  ✓ datasets.train (FEATURES={len(FEATURES)}, TARGET={TARGET!r})")
except Exception as e:
    print(f"❌ FAIL: datasets.train — {e}"); sys.exit(1)

try:
    from models.model import PulsarNet
    print("  ✓ models.model")
except Exception as e:
    print(f"❌ FAIL: models.model — {e}"); sys.exit(1)

try:
    from losses.loss import RMLoss
    print("  ✓ losses.loss")
except Exception as e:
    print(f"❌ FAIL: losses.loss — {e}"); sys.exit(1)

try:
    from pl_models import TrainPipeline
    print("  ✓ pl_models.TrainPipeline")
except Exception as e:
    print(f"❌ FAIL: pl_models — {e}"); sys.exit(1)


# ============================================================
section("STEP 2: Config")
with open("configs/train.yaml") as f:
    config = yaml.safe_load(f)

# Оверрайды для smoke-теста
config['save_path'] = os.path.join(
    config['exp_path'], config['project'], config['exp_name'] + "_SMOKE"
)
config['trainer']['max_epochs'] = 2
config['trainer']['accelerator'] = 'cpu'
config['trainer']['devices'] = 1
config['dataloader']['batch_size'] = 64
config['dataloader']['num_workers'] = 0

print(f"  save_path = {config['save_path']}")
print(f"  weights   = {config['weights']!r}")
expect(config['weights'] is None or config['weights'] == "",
       "weights=None/empty (training from scratch)")


# ============================================================
section("STEP 3: Data loaders (scalers shared train→val)")
train_loader, train_ds = get_train_dl_ds(config, mode='train')
val_loader, val_ds = get_train_dl_ds(
    config, mode='val',
    feature_scaler=train_ds.feature_scaler,
    target_scaler=train_ds.target_scaler,
)
print(f"  train: {len(train_ds)} samples, {len(train_loader)} batches")
print(f"  val:   {len(val_ds)} samples, {len(val_loader)} batches")
expect(val_ds.feature_scaler is train_ds.feature_scaler, "val.feature_scaler == train.feature_scaler")
expect(val_ds.target_scaler  is train_ds.target_scaler,  "val.target_scaler  == train.target_scaler")
expect(np.isfinite(train_ds.feature_scaler.center_).all(), "feature center finite")
expect(np.isfinite(train_ds.feature_scaler.scale_).all(),  "feature scale finite")


# ============================================================
section("STEP 4: First batch")
feats, puls = next(iter(train_loader))
print(f"  features:    {tuple(feats.shape)}, dtype={feats.dtype}, "
      f"range [{feats.min():.2f}, {feats.max():.2f}]")
print(f"  pulsar_data: {tuple(puls.shape)}, dtype={puls.dtype}, "
      f"range [{puls.min():.2f}, {puls.max():.2f}]")
expect(feats.shape[1] == len(FEATURES) == 11, f"features dim = {len(FEATURES)}")
expect(puls.shape[1] == len(PULSAR_COLS) == 2, f"pulsar_data dim = {len(PULSAR_COLS)}")
expect(torch.isfinite(feats).all(), "no NaN in features")
expect(torch.isfinite(puls).all(),  "no NaN in pulsar_data")


# ============================================================
section("STEP 5: Model forward + loss")
model = PulsarNet()
pred, target = model((feats, puls))
print(f"  pred:   {tuple(pred.shape)}, range [{pred.min():.2f}, {pred.max():.2f}]")
print(f"  target: {tuple(target.shape)}, range [{target.min():.2f}, {target.max():.2f}]")
expect(pred.shape == target.shape, f"pred shape == target shape {tuple(pred.shape)}")

criterion = RMLoss()
loss = criterion(pred, target)
print(f"  loss = {loss.item():.4f}")
expect(torch.isfinite(loss) and loss.item() >= 0, "loss finite and ≥ 0")

# Проверка, что loss — именно Huber, а не KLD
manual_huber = torch.nn.functional.huber_loss(pred, target, delta=1.0)
expect(torch.allclose(loss, manual_huber, atol=1e-6),
       "loss == nn.HuberLoss(delta=1.0)  (KLD удалён)")


# ============================================================
section("STEP 6: Backward — оба крыла получают градиент")
model.train()
model.zero_grad()
pred, target = model((feats, puls))
criterion(pred, target).backward()

def grad_sum(t):
    return float(t.grad.abs().sum().item()) if t.grad is not None else 0.0

g_trend = grad_sum(model.trend_layer.weight)
g_main  = grad_sum(model.main_layers[0].weight)
print(f"  Σ|grad trend_layer.weight|    = {g_trend:.3e}")
print(f"  Σ|grad main_layers[0].weight| = {g_main:.3e}")
expect(g_trend > 0, "trend_layer получает градиент")
expect(g_main  > 0, "main_layers получает градиент")


# ============================================================
section("STEP 7: TrainPipeline + ckpt fields")
pipeline = TrainPipeline(config=config, train_loader=train_loader, val_loader=val_loader)
print(f"  target_mean  = {pipeline.target_mean:.6f}")
print(f"  target_scale = {pipeline.target_scale:.6f}")

ckpt = {}
pipeline.on_save_checkpoint(ckpt)
expect('target_mean'  in ckpt, "ckpt has target_mean")
expect('target_scale' in ckpt, "ckpt has target_scale")
if 'feature_center' in ckpt and 'feature_scale' in ckpt:
    print("  ✓ feature_center/scale present — inference будет работать")
else:
    print("  ⚠ feature_center/scale отсутствуют — inference пока не заработает")
    print("    (для обучения это не блокер, но учтём позже)")


# ============================================================
section("STEP 8: _unscale invertible")
x = torch.tensor([0.0, 1.0, -1.0, 2.5])
x_phys = pipeline._unscale(x)
x_back = (x_phys - pipeline.target_mean) / pipeline.target_scale
print(f"  scaled: {x.tolist()}")
print(f"  phys:   {[round(v, 4) for v in x_phys.tolist()]}")
expect(torch.allclose(x, x_back, atol=1e-6), "_unscale invertible")


# ============================================================
section("STEP 9: Полный Trainer (2 эпохи, CPU)")
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

if os.path.exists(config['save_path']):
    shutil.rmtree(config['save_path'])
os.makedirs(config['save_path'], exist_ok=True)

ckpt_cb = ModelCheckpoint(
    dirpath=config['save_path'],
    save_last=True,
    every_n_epochs=1,
    save_top_k=1,
    save_weights_only=False,
    save_on_train_epoch_end=False,
    **config['checkpoint'],
)
trainer = Trainer(
    max_epochs=config['trainer']['max_epochs'],
    accelerator='cpu',
    devices=1,
    logger=False,
    callbacks=[ckpt_cb],
    enable_progress_bar=True,
    enable_model_summary=True,
)
pipeline = TrainPipeline(config=config, train_loader=train_loader, val_loader=val_loader)
trainer.fit(pipeline)


# ============================================================
section("STEP 10: Сохранённый чекпойнт")
best = os.path.join(config['save_path'], 'model.ckpt')
last = os.path.join(config['save_path'], 'last.ckpt')
print(f"  last.ckpt exists:  {os.path.exists(last)}")
print(f"  model.ckpt exists: {os.path.exists(best)}")
expect(os.path.exists(best), "model.ckpt сохранён")

ck = torch.load(best, map_location='cpu')
print(f"  keys: {sorted(ck.keys())}")
for k in ['state_dict', 'target_mean', 'target_scale']:
    expect(k in ck, f"ckpt has {k!r}")
if 'feature_center' in ck and 'feature_scale' in ck:
    print("  ✓ feature_center/scale в ckpt (inference ready)")


# ============================================================
section("CLEANUP")
shutil.rmtree(config['save_path'])
print(f"  removed {config['save_path']}")

section("ALL CHECKS PASSED ✓")
print("Пайплайн обучения работает. Запускай реальный train.")
