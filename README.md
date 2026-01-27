# Remaining Useful Life (RUL) Prediction using LSTM  
**NASA CMAPSS Turbofan Engine Dataset (FD001–FD004)**

## Overview
This project implements an end-to-end **Remaining Useful Life (RUL)** prediction pipeline using **Long Short-Term Memory (LSTM)** networks on the NASA CMAPSS turbofan engine degradation datasets.

The goal is to predict how many operating cycles remain before engine failure using multivariate time-series sensor data, and to analyze how model performance varies across different operating conditions and fault scenarios.

---

## Datasets
The CMAPSS dataset consists of four subsets with increasing complexity:

| Dataset | Operating Conditions | Fault Modes |
|-------|----------------------|-------------|
| FD001 | Single | Single |
| FD002 | Multiple | Single |
| FD003 | Single | Multiple |
| FD004 | Multiple | Multiple |

Datasets can be downloaded from the official NASA repository:  
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

> **Note:** Raw dataset files are not included in this repository due to licensing and size considerations.

---

## Methodology

### 1. Data Preprocessing
- Assigned column names for engine ID, cycle count, operating settings, and sensor readings
- Computed Remaining Useful Life (RUL) as: RUL = max_cycle_per_engine − current_cycle
- Removed non-informative sensors using variance thresholding
- Selected relevant sensors based on correlation with RUL
- Included operating condition variables explicitly
- Normalized selected features using MinMax scaling (fit on training data only)

---

### 2. Sequence Generation
- Created sliding window sequences of fixed length
- Each sequence represents recent operational history
- Target label corresponds to RUL at the final timestep of the window

Typical input shape: (samples, timesteps, features)


---

### 3. Model Architecture
A stacked LSTM regression model was used:

- LSTM (64 units, return sequences)
- Dropout (0.2)
- LSTM (32 units)
- Dense (32, ReLU)
- Dense (1, linear output)

Training details:
- Loss: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)
- Optimizer: Adam
- Early stopping based on validation loss

Each dataset (FD001–FD004) was trained **independently** with its own model.

---

### 4. Evaluation Strategy
- Model evaluated on **unseen engines**
- For test data, prediction was made using the **last available window per engine**
- Metrics used:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- Scatter plots used to visualize True vs Predicted RUL

---

## Results Summary

| Dataset | MAE (cycles) | RMSE (cycles) | Observations |
|-------|--------------|---------------|-------------|
| FD001 | ~19 | ~28 | Clean degradation patterns |
| FD002 | ~48 | ~56 | Strong operating regime effects |
| FD003 | ~18 | ~28 | Fault diversity handled well |
| FD004 | ~56 | ~66 | Regime + fault complexity overwhelms plain LSTM |

---

## Key Insights
- LSTMs handle **fault diversity** better than **operating condition variability**
- Feature selection significantly stabilizes training
- Fixed operating conditions (FD001, FD003) lead to much lower prediction error
- FD004 performance degradation highlights the need for **regime-aware modeling**
- A plain LSTM tends to collapse to mean predictions under extreme heterogeneity

---

## Project Structure
rul-lstm-cmapss/
│
├── src/
│ ├── main_fd001.py
│ ├── main_fd002.py
│ ├── main_fd003.py
│ └── main_fd004.py
│
├── models/
│ └── saved LSTM models (.keras)
│
├── results/
│ └── evaluation plots
│
├── requirements.txt
├── README.md
└── .gitignore


---

## How to Run
1. Download the CMAPSS datasets
2. Place dataset files inside a `data/` directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt


Limitations:

1)No explicit operating regime encoding
2)Single-model approach struggles on FD004




Author:

Tannishth Gupta
