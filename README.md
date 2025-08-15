# 🚕 Hourly Taxi Order Forecasting

Notebook: `sprint_thirteen_car_dealership_nb.ipynb`  
Goal: Predict the number of taxi orders for the next hour to support driver allocation.  
Business target: Test RMSE < 48. Achieved: ~7.23 (Random Forest).

## 📂 Dataset

File: `taxi.csv`  
Original granularity: 10‑minute intervals  
Columns:
- `datetime` (index, parsed)
- `num_orders` (int): taxi orders per 10‑minute slot

Processing:
- Resampled to hourly: `df_resampled = df.resample('1H').mean()`
- 6 months coverage (Mar–Aug 2018)

## 🔎 Exploratory Insights

- Right‑skewed distribution; rare high spikes (events/peaks)
- Upward trend across months
- Daily cycle: low demand early morning (05:00–07:00), peaks around 00:00 and late afternoon
- Strong short‑lag autocorrelation

## 🕒 Time Series Analysis

Methods:
- Rolling mean (24h) + derivative
- ACF / PACF (short memory, strong lag 1)
- Seasonal decomposition (trend + modest intraday seasonality)
- AR / ARIMA baseline (ARIMA(1,1,1) for short horizon)

## 🧪 Feature Engineering

Engineered on hourly data:
- Lags: `lag_1`, `lag_6`
- Rolling means: `moving_average_3`, `moving_average_12`
- Exponential smoothing (EWM, span=7)
- Decomposition parts: `trend`, `seasonal`, `residual`
- Calendar: hour, day_of_week, day_of_month, week_of_year, month_of_year
- Target: next hour `num_orders.shift(-1)`

## 🤖 Models Evaluated

| Model | Validation/Test RMSE | Notes |
|-------|----------------------|-------|
| Linear Regression | ~7.51 | Strong linear baseline |
| Decision Tree | ~8.41 | Slightly worse, interpretable |
| Random Forest | 6.92 (val) / 7.23 (test) | Best overall |
| XGBoost | ~7.24 | Close to RF |
| AR / ARIMA | Higher (baseline) | Less expressive vs feature ML |

Final model: RandomForestRegressor (tuned; n_estimators=120, max_depth=41).  
All models far exceeded target threshold.

## 📈 Usage

Install deps (minimal):

```bash
pip install pandas numpy scikit-learn seaborn matplotlib statsmodels xgboost
```

Run notebook (Windows PowerShell):

```bash
jupyter notebook sprint_thirteen_car_dealership_nb.ipynb
```

Recommended order:
1. Load & resample data
2. EDA (distribution, rolling mean, hourly pattern)
3. Decomposition & ACF/PACF
4. Feature engineering
5. Train / validate models
6. Final Random Forest test evaluation

## 📦 Repo Structure (core)

```
taxi.csv
sprint_thirteen_car_dealership_nb.ipynb
README.md
```

## ✅ Key Results

- Robust low error (≈7 RMSE vs target 48)
- Strong short‑term predictability via recent lags + calendar features
- Ensemble models outperform linear only marginally, indicating stable signal

## 🚀 Potential Enhancements

| Area | Idea |
|------|------|
| Seasonality | Add explicit weekly pattern (lag_24, lag_168) |
| Validation | TimeSeriesSplit instead of single holdout |
| Forecast Horizon | Multi-step recursive or direct models |
| External Data | Weather, events, holidays |
| Model Ops | Persistence + inference script (job scheduling) |

## 🔐 Reproducibility

- Deterministic feature generation (shifts before dropna)
- No data leakage (future target shift)
- Shuffle disabled for temporal splits

## 🏁 Summary

A structured time series + feature‑augmented ML approach produced accurate hourly forecasts well within business tolerance, enabling proactive