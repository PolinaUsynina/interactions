# file to code metric's functions and classes

# file to code metric's functions and classes.
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_quality3(y_true, y_pred, RM_real=None, show_stat=False):
    """
    y_true, y_pred : 1D array-like в ФИЗ. единицах (не scaled!).
    RM_real        : опционально, физический RM. Для sanity-check:
                     MAE(RM_real - y_true + y_pred, RM_real) == MAE(y_true, y_pred)
                     (RM_real алгебраически сокращается).
    show_stat      : печатать метрики.

    Возвращает dict: {'MAE', 'RMSE', 'R2', 'bias'[, 'MAE_RM', 'RMSE_RM']}.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))

    res = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'bias': bias}

    if RM_real is not None:
        RM_real = np.asarray(RM_real).ravel()
        rm_ion  = RM_real - y_true
        rm_rec  = rm_ion + y_pred
        res['MAE_RM']  = float(mean_absolute_error(RM_real, rm_rec))
        res['RMSE_RM'] = float(np.sqrt(mean_squared_error(RM_real, rm_rec)))

    if show_stat:
        print(f"MAE  = {mae:.6g}")
        print(f"RMSE = {rmse:.6g}")
        print(f"R^2  = {r2:.4f}")
        print(f"bias = {bias:+.4g}")
        print(f"std(y_true) = {y_true.std():.4g}")

    return res