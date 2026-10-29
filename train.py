"""Train and evaluate the LSTM RUL model on one C-MAPSS subset.

Usage:
    python train.py --dataset FD001 --seed 42
"""
import argparse
import os
import random

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import data
import evaluate

# Window sizes used by the v1 scripts.
DEFAULT_WINDOW_SIZE = {"FD001": 30, "FD002": 20, "FD003": 30, "FD004": 20}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=None,
                        help="default: 30 for FD001/FD003, 20 for FD002/FD004")
    parser.add_argument("--rul-cap", type=int, default=125,
                        help="cap on the training RUL target")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models")
    return parser.parse_args()


def set_seeds(seed):
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Makes TensorFlow ops deterministic, so the same seed gives the same model.
    tf.config.experimental.enable_op_determinism()


def main():
    args = parse_args()
    window_size = args.window_size or DEFAULT_WINDOW_SIZE[args.dataset]
    set_seeds(args.seed)

    # --- Training data ---
    train_df = data.load_split(args.data_dir, args.dataset, "train")
    train_df = data.add_rul(train_df, args.rul_cap)
    features = data.select_features(train_df)
    print(f"Features ({len(features)}): {features}")

    scaler = MinMaxScaler()
    train_df[features] = scaler.fit_transform(train_df[features])

    x_all, y_all = data.make_windows(train_df, features, window_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all, test_size=0.2, random_state=args.seed
    )
    print(f"Train windows: {x_train.shape}, val windows: {x_val.shape}")

    # --- Model ---
    # Imported here so `python train.py --help` works without loading TensorFlow.
    from tensorflow import keras
    from model import build_model

    model = build_model(timesteps=window_size, num_features=len(features))
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=2,
    )

    os.makedirs(args.models_dir, exist_ok=True)
    model_path = os.path.join(args.models_dir, f"{args.dataset}_seed{args.seed}.keras")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # --- Test set ---
    test_df = data.load_split(args.data_dir, args.dataset, "test")
    test_df[features] = scaler.transform(test_df[features])
    # The true test RUL is NOT capped: we score against what really happened.
    true_rul = data.load_test_rul(args.data_dir, args.dataset)

    x_test, unit_ids = data.last_windows(test_df, features, window_size)
    num_skipped = len(true_rul) - len(unit_ids)
    if num_skipped > 0:
        print(f"WARNING: skipped {num_skipped} test engines shorter than {window_size} cycles")

    y_pred = model.predict(x_test, verbose=0).flatten()
    y_true = true_rul[unit_ids - 1]  # unit_id 1 is row 0 of RUL_FD00X.txt

    print(f"Test MAE:  {evaluate.mae(y_true, y_pred):.2f}")
    print(f"Test RMSE: {evaluate.rmse(y_true, y_pred):.2f}")


if __name__ == "__main__":
    main()
