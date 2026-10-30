# Remaining Useful Life (RUL) Prediction with an LSTM
**NASA C-MAPSS turbofan engine dataset (FD001–FD004)**

Predicts how many operating cycles a turbofan engine has left before failure, from a
window of its recent multivariate sensor readings, using a stacked LSTM.

> **Status: being rebuilt.** The first version of this project had methodological bugs
> (leaky validation split, no RUL cap, no regime-aware normalization, incomplete test
> evaluation). They are being fixed one commit at a time. The results below are a log of
> real runs after each fix, so you can see what each fix changed. They are **not final**.

---

## Dataset

| Subset | Operating conditions | Fault modes | Train engines | Test engines |
|--------|----------------------|-------------|---------------|--------------|
| FD001  | 1                    | 1           | 100           | 100          |
| FD002  | 6                    | 1           | 260           | 259          |
| FD003  | 1                    | 2           | 100           | 100          |
| FD004  | 6                    | 2           | 249           | 248          |

Each training engine runs until failure. Test engines are cut off before failure, and
`RUL_FD00X.txt` gives the true RUL at each test engine's last cycle.

The data files are in `data/`. They come from the
[NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

---

## How to run

```bash
pip install -r requirements.txt
python train.py --dataset FD001 --seed 42
```

| Option          | Default                        | Meaning                              |
|-----------------|--------------------------------|--------------------------------------|
| `--dataset`     | (required)                     | FD001, FD002, FD003 or FD004         |
| `--seed`        | 42                             | seeds Python, NumPy and TensorFlow   |
| `--rul-cap`     | 125                            | cap on the training RUL target       |
| `--window-size` | 30 (FD001/FD003), 20 (FD002/FD004) | cycles per input window          |
| `--epochs`      | 50                             | max epochs (early stopping, patience 5) |

The trained model is saved to `models/FD00X_seed{N}.keras`.

Tested with Python 3.10 on Windows (CPU only). With the same seed, runs give identical results.

---

## Method (current state)

1. **RUL label:** `RUL = min(last_cycle − cycle, 125)`. Early in life the sensors show no
   wear, so the target is flat at 125, then decreases linearly to 0 at failure.
2. **Train/validation split by engine:** 20% of the training *engines* (chosen with the
   seed) are held out for validation. Every row of an engine goes to one side only.
3. **Operating regimes:** each row's flight condition is found by rounding the 3 operating
   settings. This gives 1 regime in FD001/FD003 and 6 in FD002/FD004.
4. **Features:** all 21 sensors, minus those that are constant within every regime in the
   training engines (the dropped sensors are printed in the log). The operating settings
   are not used as inputs.
5. **Per-regime normalization:** one StandardScaler per regime, fit on the training engines
   only and then applied to validation and test. A value of +2 means "2 standard
   deviations above normal for this flight condition".
6. **Windows:** sliding windows built separately inside each engine on each side of the
   split. Each window is labelled with the RUL at its last cycle.
7. **Model:** LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(32, ReLU) → Dense(1). MSE loss,
   Adam (lr 1e-3), batch size 64.
8. **Test:** predict on the last window of each test engine and compare with
   `RUL_FD00X.txt` (true test RUL is *not* capped).

---

## Results log

All runs: seed 42, one run per subset. Test metrics are on the official test set.

| Change | FD001 RMSE | FD002 RMSE | FD003 RMSE | FD004 RMSE |
|--------|-----------:|-----------:|-----------:|-----------:|
| 1. Refactor of v1 into `train.py` (same behaviour as v1) | 27.49 | 56.63 | 33.13 | 66.47 |
| 2. Cap RUL at 125 + fix off-by-one window label | 14.05 | 53.80 | 14.22 | 54.58 |
| 3. Split train/val by engine (no window leakage) | 15.05 | 53.80 | 18.53 | 54.65 |
| 4. Per-regime normalization; drop constant sensors | 14.90 | 27.53 | 14.66 | 27.28 |

What each change did:

1. **Refactor:** same preprocessing as the original four scripts (checked to give
   identical arrays). This is the v1 baseline.
2. **RUL cap:** FD001 and FD003 RMSE roughly halved, because the model no longer has to
   guess a large uncapped RUL for healthy engines whose sensors all look the same.
   FD002/FD004 barely improved (see limitations).
3. **Split by engine:** before, windows were shuffled into train/val randomly, so
   near-identical overlapping windows from one engine sat on both sides and the
   validation score was too optimistic. Now validation engines are never seen in training.
   Test RMSE got slightly worse on FD001 (14.05 → 15.05) and clearly worse on FD003
   (14.22 → 18.53), where early stopping now stopped at epoch 12 instead of 47. The test
   set itself never leaked; what changed is that early stopping now picks the epoch using
   an honest validation score. With one seed, part of this difference may be noise.
4. **Per-regime normalization:** in FD002/FD004 a sensor reading jumps when the flight
   condition changes, which hides the slow wear trend. In v1 no sensor correlated with
   RUL, so the correlation filter removed all of them and the model only saw the 3
   operating settings. Scaling each regime separately makes the wear trend visible again:
   in FD002's training engines, the number of sensors with |correlation with RUL| > 0.2
   went from 0 to 16. FD002 and FD004 RMSE roughly halved (53.80 → 27.53 and
   54.65 → 27.28). The correlation filter was removed; only sensors that are constant
   within every regime are dropped.

---

## Known limitations (current state)

- **Training stops very early.** Early stopping uses a patience of 5 epochs, and in the
  sub-step 4 runs it ended training after 7–11 epochs, so the best epoch was epoch 2–6.
  The model may be undertrained. Not tuned yet.
- **FD004 skips 2 of 248 test engines** that are shorter than the 20-cycle window.
  *(Planned.)*
- **No NASA asymmetric score yet**, only MAE/RMSE. *(Planned.)*
- **One seed per result.** Differences of a few RMSE points between runs may be noise.
- The old `models/rul_lstm_fd00X.keras` and `results/scatter_*.png` files are from v1 and
  don't match the current code.

---

## Project structure

```
data.py       loading, RUL labels, engine split, regimes, per-regime scaling, windowing
model.py      the LSTM model
evaluate.py   metrics
train.py      entry point: python train.py --dataset FD00X
data/         C-MAPSS text files
models/       saved models
```

---

Author: Tannishth Gupta
