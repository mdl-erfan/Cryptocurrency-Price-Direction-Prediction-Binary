# Cryptocurrency Price Direction Prediction

A machine learning project that predicts the next 15-minute price direction (up or down) for five cryptocurrencies using historical candlestick data.

---

## Overview

This is a binary classification problem:
- `1` → Price goes **up** at t+1
- `0` → Price goes **down** at t+1

A universal model is trained across all five assets to leverage shared market structure and maximize training data.

---

## Dataset

Five CSV files of 15-minute OHLCV candlestick data from Bybit:

| Asset     | Rows    | Start Date  |
|-----------|---------|-------------|
| BTCUSDT   | 204,389 | Mar 2020    |
| ETHUSDT   | 170,350 | Mar 2021    |
| DOGEUSDT  | 162,725 | Jun 2021    |
| SOLUSDT   | 149,806 | Oct 2021    |
| XRPUSDT   | 164,648 | May 2021    |

**Total after merging:** ~851,918 rows

---

## Project Structure

```
├── project.ipynb           # Main notebook
├── BYBIT_BTCUSDT_15m.csv
├── BYBIT_ETHUSDT_15m.csv
├── BYBIT_DOGEUSDT_15m.csv
├── BYBIT_SOLUSDT_15m.csv
├── BYBIT_XRPUSDT_15m.csv
└── README.md
```

---

## Methodology

### 1. Data Cleaning
- No missing values or duplicates found across all five datasets
- Datetime parsed and sorted chronologically per asset

### 2. Feature Engineering
- **Percentage returns** computed per crypto (using `groupby` to prevent cross-asset contamination)
- **Lag features:** returns at t−1, t−2, t−3, t−6, t−12
- **Momentum:** 3, 6, 12-period price change
- **Volatility:** rolling std over 6 and 12 periods; volatility ratio
- **Mean reversion:** 12-period moving average and z-score
- **Rolling stats:** 5-period rolling mean and std of returns

### 3. Target Variable
- `target = 1` if the **next** return > 0, else `0`
- Created using `shift(-1)` per crypto group to avoid leakage

### 4. Train/Test Split
- **Time-based split** — no random shuffling (required for time series)
- **Train:** all data before the last year (≈80%)
- **Test:** most recent year (≈20%)
- Train max: `2025-01-22` | Test min: `2025-01-22` → no overlap

---

## Models & Results

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | 50.6%    | 0.509   |
| Random Forest       | 51.4%    | 0.518   |
| **LightGBM**        | **51.9%**| **0.530** |

LightGBM performed best, capturing weak nonlinear interactions that the linear baseline could not detect.

---

## Conclusion

Performance is marginal across all models. The best model (LightGBM) achieves a ROC-AUC of ~0.53, indicating a very weak predictive signal. This is consistent with the **Efficient Market Hypothesis** — high-frequency crypto markets largely price in available information, making short-term direction prediction extremely difficult with OHLCV features alone.

---

## Requirements

```bash
pip install pandas scikit-learn lightgbm matplotlib
```

---

## How to Run

1. Clone the repository and place all CSV files in the same directory as `project.ipynb`
2. Open the notebook and run all cells in order
3. Ensure the CSV filenames match exactly as listed above
