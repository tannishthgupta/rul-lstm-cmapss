"""Train and evaluate the LSTM RUL model on one C-MAPSS subset.

Usage:
    python train.py --dataset FD001 --seed 42
"""
import argparse
import json
import os
import random

import numpy as np

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
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="fraction of training ENGINES held out for validation")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
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
    full_df = data.load_split(args.data_dir, args.dataset, "train")
    full_df = data.add_rul(full_df, args.rul_cap)
    full_df = data.add_regime(full_df)

    # Split by engine FIRST, then build windows inside each side.
    train_df, val_df, val_ids = data.split_engines(full_df, args.val_fraction, args.seed)
    num_train_engines = train_df["unit_id"].nunique()
    print(f"Engines: {num_train_engines} train, {len(val_ids)} val")
    print(f"Val engine ids: {val_ids.tolist()}")

    # Anything "learned" from data (which sensors to drop, scaler mean/std)
    # comes from the training engines only; validation is treated like test data.
    regimes = sorted(train_df["regime"].unique())
    print(f"Regimes ({len(regimes)}): {regimes}")

    dropped = data.find_constant_sensors(train_df)
    features = [s for s in data.SENSORS if s not in dropped]
    print(f"Dropped constant sensors ({len(dropped)}): {dropped}")
    print(f"Features ({len(features)}): {features}")

    scalers = data.fit_regime_scalers(train_df, features)
    train_df = data.apply_regime_scalers(train_df, features, scalers)
    val_df = data.apply_regime_scalers(val_df, features, scalers)

    x_train, y_train = data.make_windows(train_df, features, window_size)
    x_val, y_val = data.make_windows(val_df, features, window_size)
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
    history = model.fit(
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
    test_df = data.add_regime(data.load_split(args.data_dir, args.dataset, "test"))
    test_df = data.apply_regime_scalers(test_df, features, scalers)
    # The true test RUL is NOT capped: we score against what really happened.
    true_rul = data.load_test_rul(args.data_dir, args.dataset)

    x_test, unit_ids, num_padded = data.last_windows(test_df, features, window_size)
    if num_padded > 0:
        print(f"Padded {num_padded} test engines shorter than {window_size} cycles")
    # Every test engine must be scored, in the same order as RUL_FD00X.txt.
    assert len(unit_ids) == len(true_rul), "number of test engines != number of RUL values"
    assert (unit_ids == np.arange(1, len(true_rul) + 1)).all(), "test unit_ids out of order"

    y_pred = model.predict(x_test, verbose=0).flatten()
    y_true = true_rul

    test_mae = evaluate.mae(y_true, y_pred)
    test_rmse = evaluate.rmse(y_true, y_pred)
    test_score = evaluate.nasa_score(y_true, y_pred)
    num_late = int(np.sum(y_pred > y_true))
    print(f"Test engines: {len(y_true)}  (late predictions: {num_late})")
    print(f"Test MAE:        {test_mae:.2f}")
    print(f"Test RMSE:       {test_rmse:.2f}")
    print(f"Test NASA score: {test_score:.1f}")

    # --- Save results ---
    val_loss = history.history["val_loss"]
    results = {
        "dataset": args.dataset,
        "seed": args.seed,
        "args": vars(args),
        "window_size": window_size,
        "regimes": regimes,
        "dropped_sensors": dropped,
        "features": features,
        "val_engine_ids": val_ids.tolist(),
        "epochs_run": len(val_loss),
        "best_epoch": int(np.argmin(val_loss)) + 1,
        "best_val_rmse": float(np.sqrt(min(val_loss))),
        "test": {
            "num_engines": len(y_true),
            "num_padded": num_padded,
            "num_late": num_late,
            "mae": test_mae,
            "rmse": test_rmse,
            "nasa_score": test_score,
        },
        # Kept for later error analysis (Phase 2).
        "per_engine": {
            "unit_id": unit_ids.tolist(),
            "true_rul": y_true.tolist(),
            "pred_rul": [round(float(p), 2) for p in y_pred],
        },
    }
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, f"{args.dataset}_seed{args.seed}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
