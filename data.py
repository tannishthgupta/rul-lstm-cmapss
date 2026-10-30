"""Loading and preprocessing for the NASA C-MAPSS turbofan dataset."""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

OP_SETTINGS = [f"op_setting_{i}" for i in range(1, 4)]
# Decimals that separate the 6 C-MAPSS flight conditions (altitude, Mach, throttle).
REGIME_ROUNDING = {"op_setting_1": 0, "op_setting_2": 2, "op_setting_3": 0}
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["unit_id", "cycle"] + OP_SETTINGS + SENSORS


def load_split(data_dir, dataset, split):
    """Load train_FD00X.txt or test_FD00X.txt as a DataFrame with named columns."""
    path = os.path.join(data_dir, f"{split}_{dataset}.txt")
    # The files are space-separated with trailing spaces; r"\s+" handles both.
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_test_rul(data_dir, dataset):
    """True RUL at the last cycle of each test engine (row i = unit_id i+1)."""
    path = os.path.join(data_dir, f"RUL_{dataset}.txt")
    return pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"])["RUL"].values


def add_rul(df, rul_cap):
    """Add the RUL column: cycles left until this engine's last (failure) cycle,
    capped at rul_cap.

    Early in its life an engine shows no measurable wear, so the sensors can't
    tell RUL=300 from RUL=150. Capping makes the target flat (=rul_cap) while
    the engine is healthy and linear as it degrades: a piecewise-linear target.
    """
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=rul_cap)
    return df


def split_engines(df, val_fraction, seed):
    """Split a training DataFrame into train and validation by engine (unit_id).

    Every row of an engine goes to the same side, so overlapping windows from
    one engine can never end up in both train and validation.
    """
    unit_ids = df["unit_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unit_ids)

    num_val = round(len(unit_ids) * val_fraction)
    val_ids = np.sort(shuffled[:num_val])

    is_val = df["unit_id"].isin(val_ids)
    return df[~is_val].copy(), df[is_val].copy(), val_ids


def add_regime(df):
    """Label each row with its operating regime (flight condition).

    The settings are noisy measurements of a few fixed flight conditions, so
    rounding them gives the condition back: 1 regime in FD001/FD003, 6 in
    FD002/FD004. The label is a string such as "42.0_0.84_100.0".
    """
    # "+ 0.0" turns -0.0 into 0.0, otherwise they'd count as different regimes.
    rounded = df[OP_SETTINGS].round(REGIME_ROUNDING) + 0.0
    df["regime"] = rounded.astype(str).agg("_".join, axis=1)
    return df


def find_constant_sensors(train_df):
    """Sensors that never change within any regime, so they carry no wear signal.

    This is checked per regime: in FD002/FD004 some sensors change only when the
    flight condition changes, and are flat inside each condition.
    """
    std_per_regime = train_df.groupby("regime")[SENSORS].std()
    max_std = std_per_regime.max()
    return max_std[max_std < 1e-8].index.tolist()


def fit_regime_scalers(train_df, features):
    """Fit one StandardScaler per regime, on training rows only."""
    scalers = {}
    for regime, rows in train_df.groupby("regime"):
        scalers[regime] = StandardScaler().fit(rows[features])
    return scalers


def apply_regime_scalers(df, features, scalers):
    """Scale each row with the scaler of its own regime.

    After this, a value of +2 means "2 standard deviations above normal for this
    flight condition", so readings from different conditions are comparable.
    """
    df = df.copy()
    # Some sensors (e.g. sensor_17) are stored as integers; scaled values aren't.
    df[features] = df[features].astype(float)
    for regime, rows in df.groupby("regime"):
        if regime not in scalers:
            raise ValueError(f"Regime {regime} was not seen in the training engines")
        df.loc[rows.index, features] = scalers[regime].transform(rows[features])
    return df


def make_windows(df, features, window_size):
    """Sliding windows inside each engine.

    Each window ends at row `end` (inclusive) and is labelled with the RUL at
    that same row, which matches how the test set is scored: the last window
    of a test engine against the RUL at its last cycle.
    """
    x_all = []
    y_all = []
    for unit_id in df["unit_id"].unique():
        engine = df[df["unit_id"] == unit_id]
        x_engine = engine[features].values
        y_engine = engine["RUL"].values

        for end in range(window_size - 1, len(engine)):
            x_all.append(x_engine[end - window_size + 1:end + 1])
            y_all.append(y_engine[end])

    return np.array(x_all), np.array(y_all)


def last_windows(df, features, window_size):
    """The last window_size cycles of each test engine.

    Engines shorter than window_size are skipped (same as v1); the returned
    unit_ids say which engines were kept.
    """
    x_test = []
    unit_ids = []
    for unit_id in df["unit_id"].unique():
        engine = df[df["unit_id"] == unit_id]
        if len(engine) >= window_size:
            x_test.append(engine[features].values[-window_size:])
            unit_ids.append(unit_id)

    return np.array(x_test), np.array(unit_ids)
