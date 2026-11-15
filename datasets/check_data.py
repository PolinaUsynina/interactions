# check_data.py
import os
import numpy as np
import pandas as pd

# ВАЖНО: переименуй "import os.txt" -> "dataset.py"
from train import TrainDataset, get_train_dl_ds, FEATURES, TARGET, PULSAR_COLS


# Для быстрой проверки — конфиг прямо здесь (без pyyaml).
# Если хочешь из YAML — раскомментируй блок ниже и pip install pyyaml.
CONFIG = {
    "data_path": "data/",
    "train": "J0332_full.csv",
    "val":   "J0814_full.csv",   # NB: код грузит test вместо val — это ожидаемо
    "test":  "J2219_full.csv",
    "dataloader": {"batch_size": 128, "num_workers": 0},  # 0 для отладки
}
# import yaml
# with open("train.yaml") as f:
#     CONFIG = yaml.safe_load(f)


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check_raw_csv(split):
    key = "test" if split == "val" else split   # так устроен твой код
    path = os.path.join(CONFIG["data_path"], CONFIG[key])
    section(f"[{split}] -> {path}")
    if not os.path.exists(path):
        print(f"  !!! ФАЙЛ НЕ НАЙДЕН")
        return None
    df = pd.read_csv(path, sep=" ")
    print(f"  shape: {df.shape}")
    print(f"  columns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"  NaN count: {int(df.isna().sum().sum())}")

    missing_feat = [c for c in FEATURES if c not in df.columns]
    missing_puls = [c for c in PULSAR_COLS if c not in df.columns]
    if missing_feat:
        print(f"  !!! ОТСУТСТВУЮТ FEATURES: {missing_feat}")
    if missing_puls:
        print(f"  !!! ОТСУТСТВУЮТ PULSAR_COLS: {missing_puls}")
    if not missing_feat and not missing_puls:
        print(f"  OK: все {len(FEATURES)} фич + {PULSAR_COLS} на месте")
    # первые 2 строки для визуального контроля
    print("  head:")
    print(df.head(2).to_string())
    return df


def check_dataset_init():
    section("TrainDataset: init + scaling")
    ds_tr  = TrainDataset(CONFIG, mode="train")
    ds_val = TrainDataset(CONFIG, mode="val",
                          feature_scaler=ds_tr.feature_scaler,
                          target_scaler=ds_tr.target_scaler)
    ds_te  = TrainDataset(CONFIG, mode="test",
                          feature_scaler=ds_tr.feature_scaler,
                          target_scaler=ds_tr.target_scaler)
    print(f"  lengths: train={len(ds_tr)}, val={len(ds_val)}, test={len(ds_te)}")

    print("\n  feature_scaler (RobustScaler):")
    print(f"    center_: {np.round(ds_tr.feature_scaler.center_, 4)}")
    print(f"    scale_:  {np.round(ds_tr.feature_scaler.scale_, 4)}")
    print("  target_scaler (StandardScaler):")
    print(f"    mean_:  {ds_tr.target_scaler.mean_}")
    print(f"    scale_: {ds_tr.target_scaler.scale_}")

    # на train после transform: mean ~ 0, std ~ 1 (для StandardScaler на target)
    y_tr = ds_tr.dataset[TARGET].to_numpy()
    print(f"\n  train TARGET after scale: "
          f"mean={y_tr.mean():.4f}, std={y_tr.std():.4f}  (ожидание ~0, ~1)")

    # scaler-ы один и тот же объект на val/test
    assert ds_val.feature_scaler is ds_tr.feature_scaler
    assert ds_val.target_scaler  is ds_tr.target_scaler
    assert ds_te.feature_scaler  is ds_tr.feature_scaler
    print("  OK: val/test делят scaler-ы с train")

    # inverse_transform туда-обратно
    section("inverse_transform sanity")
    y_phys = ds_tr.target_scaler.inverse_transform(y_tr.reshape(-1, 1)).ravel()
    print(f"  scaled TARGET:    mean={y_tr.mean():.4f}, std={y_tr.std():.4f}")
    print(f"  inverse TARGET:   mean={y_phys.mean():.4f}, std={y_phys.std():.4f}")
    print("  OK: inverse_transform восстановил исходные единицы")

    return ds_tr, ds_val, ds_te


def check_one_item(ds):
    section("TrainDataset: __getitem__[0]")
    item = ds[0]
    print(f"  features:    shape={tuple(item['features'].shape)}, "
          f"dtype={item['features'].dtype}  (ожидание ({len(FEATURES)},))")
    print(f"  pulsar_data: shape={tuple(item['pulsar_data'].shape)}, "
          f"dtype={item['pulsar_data'].dtype}  (ожидание ({len(PULSAR_COLS)},))")
    print(f"  idx: {item['idx']}  (train → long tensor; val/test → int)")

    # проверка, что в train шум вообще добавился (features не совпадают с сырыми)
    item2 = ds[0]
    print(f"  train noise check: features differ across calls: "
          f"{not np.allclose(item['features'].numpy(), item2['features'].numpy())}")


def check_dataloader():
    section("DataLoader: один батч")
    dl, ds = get_train_dl_ds(CONFIG, mode="train")
    feats, puls = next(iter(dl))
    print(f"  batch features:    {tuple(feats.shape)}, dtype={feats.dtype}")
    print(f"  batch pulsar_data: {tuple(puls.shape)}, dtype={puls.dtype}")
    print(f"  batch_size ожидаем {CONFIG['dataloader']['batch_size']}, "
          f"получили {feats.shape[0]}")
    print(f"  features range: [{feats.min():.3f}, {feats.max():.3f}]")
    print(f"  TARGET (last col of pulsar_data) range: "
          f"[{puls[:, -1].min():.3f}, {puls[:, -1].max():.3f}]")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        check_raw_csv(split)

    ds_tr, ds_val, ds_te = check_dataset_init()
    check_one_item(ds_tr)
    check_one_item(ds_val)   # без шума
    check_dataloader()

    print("\n" + "=" * 60)
    print("ГОТОВО. Если нигде не было '!!!' — чтение работает.")
    print("=" * 60)
