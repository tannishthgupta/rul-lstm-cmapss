"""Loading and preprocessing for the NASA C-MAPSS turbofan dataset."""
import os

import numpy as np
import pandas as pd

OP_SETTINGS = [f"op_setting_{i}" for i in range(1, 4)]
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


def select_features(train_df):
    """v1 feature selection: drop near-constant sensors, keep sensors with
    |correlation with RUL| > 0.2, then add the 3 operating settings."""
    sensor_std = train_df[SENSORS].std()
    useful_sensors = sensor_std[sensor_std > 0.01].index.tolist()

    corr_with_rul = train_df[useful_sensors + ["RUL"]].corr()["RUL"].drop("RUL")
    selected_sensors = corr_with_rul[corr_with_rul.abs() > 0.2].index.tolist()

    return selected_sensors + OP_SETTINGS


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
