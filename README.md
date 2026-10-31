# Remaining Useful Life (RUL) Prediction with an LSTM
**NASA C-MAPSS turbofan engine dataset (FD001–FD004)**

Predicts how many operating cycles a turbofan engine has left before failure, from a
window of its recent sensor readings, using a stacked LSTM.

This is a rebuild of an earlier version that had methodological bugs: a leaky validation
split, no RUL cap, no regime-aware normalization (so FD002/FD004 used no sensors at all),
and test engines that were silently skipped. Each fix is its own commit, and the
[results log](#how-each-fix-changed-the-results) shows what each one changed. Every number
in this README comes from running the code in this repository.

---

## Results

Seed 42, one run per subset, official test set, every test engine scored.
Source: `results/FD00X_seed42.json`.

| Subset | Test engines | RMSE | MAE | NASA score | Late predictions | Best epoch |
|--------|-------------:|-----:|----:|-----------:|-----------------:|-----------:|
| FD001  | 100 | 14.90 | 11.21 | 373   | 53 / 100  | 4 |
| FD002  | 259 | 27.53 | 19.02 | 11383 | 123 / 259 | 2 |
| FD003  | 100 | 14.66 | 10.47 | 546   | 52 / 100  | 6 |
| FD004  | 248 | 27.31 | 20.39 | 4395  | 117 / 248 | 5 |

- RMSE and MAE are in cycles. The true test RUL is **not** capped.
- The NASA score is a sum over engines (lower is better), so compare it only within one
  subset.
- These are **single-seed** results. Differences of a point or two may be noise.
- They are not directly comparable to published results that cap the test labels at 125
  (see [limitations](#known-limitations)).

---

## Dataset

| Subset | Operating conditions | Fault modes | Train engines | Test engines |
|--------|----------------------|-------------|---------------|--------------|
| FD001  | 1                    | 1           | 100           | 100          |
| FD002  | 6                    | 1           | 260           | 259          |
| FD003  | 1                    | 2           | 100           | 100          |
| FD004  | 6                    | 2           | 249           | 248          |

Each row is one engine on one cycle: `unit_id`, `cycle`, 3 operating settings and 21
sensors. Training engines run until failure. Test engines are cut off before failure, and
`RUL_FD00X.txt` gives the true RUL at each test engine's last cycle.

The data files are in `data/`. They come from the
[NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

---

## How to run

```bash
pip install -r requirements.txt
python train.py --dataset FD001 --seed 42
```

To run all four subsets (bash):

```bash
for d in FD001 FD002 FD003 FD004; do python train.py --dataset $d --seed 42; done
```

| Option           | Default                            | Meaning                                  |
|------------------|------------------------------------|------------------------------------------|
| `--dataset`      | (required)                         | FD001, FD002, FD003 or FD004             |
| `--seed`         | 42                                 | seeds Python, NumPy and TensorFlow, and the engine split |
| `--rul-cap`      | 125                                | cap on the training RUL target           |
| `--window-size`  | 30 (FD001/FD003), 20 (FD002/FD004) | cycles per input window                  |
| `--val-fraction` | 0.2                                | fraction of training engines held out for validation |
| `--epochs`       | 50                                 | max epochs (early stopping, patience 5)  |
| `--batch-size`   | 64                                 | training batch size                      |

Outputs:
- `models/FD00X_seed{N}.keras`: the trained model (not tracked in git).
- `results/FD00X_seed{N}.json`: settings, regimes, dropped sensors, features, validation
  engine IDs, epochs run and best epoch, test metrics, and every test engine's prediction.

Tested with Python 3.10 on Windows, CPU only (versions pinned in `requirements.txt`).
TensorFlow op determinism is enabled, so the same seed gives identical results on the same
machine. Rerunning FD001 and FD004 with seed 42 reproduced their result files exactly.
Results on other hardware or library versions may differ slightly.

---

## Method

1. **RUL label:** `RUL = min(last_cycle − cycle, 125)`. Early in life the sensors show no
   wear, so the target is flat at 125, then decreases linearly to 0 at failure: a
   piecewise-linear target.
2. **Train/validation split by engine:** 20% of the training *engines* (chosen with the
   seed) are held out for validation. Every row of an engine goes to one side only, so
   overlapping windows from one engine can't appear on both sides.
3. **Operating regimes:** each row's flight condition is found by rounding the 3 operating
   settings. This gives 1 regime in FD001/FD003 and 6 in FD002/FD004, the same regimes in
   train and test.
4. **Features:** all 21 sensors, minus those that are constant within every regime in the
   training engines (sensors 1, 5, 18 and 19 everywhere, plus 10 and 16 in FD001 and 16 in
   FD003). The operating settings are not used as inputs.
5. **Per-regime normalization:** one StandardScaler per regime, fit on the training engines
   only and then applied to validation and test. A value of +2 means "2 standard
   deviations above normal for this flight condition".
6. **Windows:** sliding windows built separately inside each engine on each side of the
   split. Each window is labelled with the RUL at its last cycle.
7. **Model:** LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(32, ReLU) → Dense(1). MSE loss,
   Adam (learning rate 1e-3), batch size 64, early stopping on validation loss (patience 5,
   best weights restored).
8. **Test:** predict on the last window of each test engine and compare with
   `RUL_FD00X.txt`. Test engines shorter than the window (2 in FD004) are padded at the
   start by repeating their first cycle, so every test engine is scored.
9. **Metrics:** MAE, RMSE, and the NASA asymmetric score from the PHM08 challenge
   (Saxena et al., 2008). With `error = predicted − true`, each engine adds
   `exp(−error/13) − 1` if early and `exp(error/10) − 1` if late. Late predictions, where
   the engine would fail before maintenance, cost more.

---

## How each fix changed the results

Test RMSE, seed 42, one run per subset.

| Change | FD001 | FD002 | FD003 | FD004 |
|--------|------:|------:|------:|------:|
| 1. Refactor of v1 into `train.py` (same behaviour as v1) | 27.49 | 56.63 | 33.13 | 66.47 |
| 2. Cap RUL at 125 + fix off-by-one window label | 14.05 | 53.80 | 14.22 | 54.58 |
| 3. Split train/val by engine (no window leakage) | 15.05 | 53.80 | 18.53 | 54.65 |
| 4. Per-regime normalization; drop constant sensors | 14.90 | 27.53 | 14.66 | 27.28 |
| 5. Score every test engine; add NASA score | 14.90 | 27.53 | 14.66 | 27.31 |

1. **Refactor:** the four per-subset scripts were merged into one `train.py`, with the same
   preprocessing (checked to produce identical arrays). This row is the v1 baseline.
2. **RUL cap and window label:** FD001 and FD003 RMSE roughly halved, because the model no
   longer has to guess a large RUL for healthy engines whose sensors all look the same.
   Each window is now labelled with the RUL at its own last cycle (v1 used the next
   cycle). FD002/FD004 barely improved, because at this point they used no sensors.
3. **Split by engine:** before, windows were shuffled into train/val randomly, so
   near-identical windows from one engine sat on both sides and the validation score was
   too optimistic. Test RMSE got worse on FD001 (14.05 → 15.05) and FD003
   (14.22 → 18.53). The test set itself never leaked; what changed is that early stopping
   now chooses the epoch using an honest validation score.
4. **Per-regime normalization:** in FD002/FD004 a sensor reading jumps when the flight
   condition changes, which hides the slow wear trend. In v1 no sensor correlated with
   RUL, so a correlation filter removed all of them and the model only saw the 3
   operating settings. After per-regime scaling, the number of FD002 sensors with
   |correlation with RUL| > 0.2 went from 0 to 16, and FD002/FD004 RMSE roughly halved.
5. **Complete test evaluation:** v1 skipped the 2 FD004 test engines shorter than the
   window. They are now padded and scored, and the code checks that every engine in
   `RUL_FD00X.txt` gets exactly one prediction. The model didn't change, so FD001–FD003
   are identical to step 4.

---

## Known limitations

- **One seed per result.** No mean ± std over seeds yet, so small differences between
  rows or subsets may be noise.
- **Training stops very early.** With early-stopping patience 5, training ended after
  7–11 epochs (best epoch 2–6). The models are probably undertrained, especially on
  FD002 (best epoch 2).
- **Hyperparameters are not tuned.** The architecture, window sizes (30/20), learning rate
  and batch size are carried over from v1.
- **The cap dominates the FD002/FD004 NASA score.** The model can't predict much above
  125, but some test engines have a true RUL near 195. Those engines get large, unavoidable
  "early" errors, which the exponential score blows up: engines with true RUL > 125 account
  for 88% of the FD002 score and 72% of the FD004 score. Many papers also cap the *test*
  labels at 125, which removes this effect, so these numbers are not directly comparable
  to published ones.
- **No baselines or uncertainty yet.** It's not yet shown how much the LSTM adds over
  simpler models, and predictions have no confidence interval.
- **Regime detection by rounding** works because C-MAPSS uses 6 fixed flight conditions.
  Real data with continuous operating conditions would need clustering or a different
  approach.
- **Leftover v1 files.** `models/rul_lstm_fd00X.keras` and `results/scatter_*.png` come
  from the original version and don't match the current code.

---

## Next steps

- Run 5 seeds per subset and report mean ± std.
- Baselines: a constant mean-RUL predictor, a tree model on hand-made window features,
  and a 1D-CNN.
- A literature comparison using the published RMSE values, scored both with uncapped and
  capped test labels.
- Ablations: window length, with and without regime normalization, with and without the
  RUL cap.
- Uncertainty: MC dropout or a seed ensemble, with a coverage check of the prediction
  intervals.
- Error analysis: error vs true RUL, early vs late predictions, example engine
  trajectories.

---

## Project structure

```
train.py      entry point: python train.py --dataset FD00X
data.py       loading, RUL labels, engine split, regimes, per-regime scaling, windowing
model.py      the LSTM model
evaluate.py   MAE, RMSE, NASA score
data/         C-MAPSS text files
models/       saved models
results/      one JSON file per run
```

---

Author: Tannishth Gupta
