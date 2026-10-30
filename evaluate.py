"""Evaluation metrics."""
import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nasa_score(y_true, y_pred):
    """The asymmetric scoring function from the PHM08 challenge (Saxena et al., 2008).

    error = predicted - true. A late prediction (error > 0: the model thinks the
    engine has more life than it does) is penalised more than an early one,
    because it means the engine can fail before maintenance:
        error < 0 (early): exp(-error / 13) - 1
        error >= 0 (late): exp( error / 10) - 1
    The score is the SUM over engines, so lower is better and it grows with the
    number of engines (FD002/FD004 have ~2.5x more test engines than FD001/FD003).
    """
    error = y_pred - y_true
    per_engine = np.where(error < 0, np.exp(-error / 13) - 1, np.exp(error / 10) - 1)
    return float(np.sum(per_engine))
