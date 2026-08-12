# Appliance Energy Forecasting - Assignment 2

A reproducible hourly time-series forecasting project for the UCI **Appliances Energy Prediction** dataset. The implementation follows the supplied assignment/reference repository style: reusable code under `src/appliance_energy/`, one script per modelling stage, notebooks, tests, and saved outputs under `outputs/`.

## Assignment workflow

1. Download and clean the 10-minute UCI data.
2. Resample to hourly values.
3. Explore the series and test stationarity using ADF/KPSS, ACF/PACF and differencing.
4. Hold out the final 14 days chronologically. The forecast evaluation window is the **first 24 hours of that holdout**, matching the required 24-hour forecasting horizon.
5. Generate mean, naive, daily seasonal-naive, weekly seasonal-naive and drift benchmarks.
6. Search all **147** required non-seasonal SARIMAX orders: `p=0..6`, `d=0..2`, `q=0..6`, with daily seasonal period 24. Select by AIC.
7. Fit target-only SARIMA and weather-exogenous SARIMAX; inspect residual diagnostics and 95% confidence intervals.
8. Fit XGBoost using sensor/weather, time, lag and rolling features. The 24-hour ML forecast is recursive so future target values are not used as lag inputs. Future sensor/weather values are the realised holdout values and are explicitly treated as a **conditional forecast**.
9. Fit **real Chronos** zero-shot target-only foundation model forecasting. There is **no fallback**: if Chronos is unavailable, preflight stops before the expensive SARIMAX search.
10. Compare all models using MAE, RMSE, MASE and Bias and generate the final forecast plots.

## Project structure

```text
appliance-energy-forecasting/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_data_download_and_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_benchmark_models.ipynb
│   ├── 04_sarimax_models.ipynb
│   ├── 05_feature_based_models.ipynb
│   ├── 06_foundation_model.ipynb
│   └── 07_model_comparison.ipynb
├── src/appliance_energy/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── stationarity.py
│   ├── evaluation.py
│   ├── plotting.py
│   └── models/
│       ├── benchmarks.py
│       ├── sarimax.py
│       ├── feature_models.py
│       └── foundation.py
├── scripts/
│   ├── preflight.py
│   ├── run_eda_stationarity.py
│   ├── run_benchmarks.py
│   ├── run_sarimax.py
│   ├── run_feature_model.py
│   ├── run_foundation_model.py
│   ├── run_final_comparison.py
│   └── run_pipeline.py
├── tests/
└── outputs/
    ├── figures/
    ├── forecasts/
    ├── metrics/
    └── model_objects/
```

## Windows setup

Use Python 3.10-3.12. From the project root:

```powershell
C:\Python312\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH="$PWD\src"
python -m pytest
```

## Run safely

**Run preflight first.** It checks every dependency and loads the actual Chronos model before the long SARIMAX search:

```powershell
python scripts\preflight.py
```

Only after `PREFLIGHT PASSED` should you run:

```powershell
python scripts\run_pipeline.py
```

The SARIMAX order search is the slowest step because the assignment explicitly requires 147 candidate fits. Results are checkpointed to `outputs/metrics/order_selection.csv`; subsequent runs reuse that completed search. Delete the file only if you intentionally want to repeat the grid search.

## Main outputs

The pipeline creates the same style of outputs expected in the reference project:

### Figures

- `eda_overview.png`
- `eda_seasonal_profiles.png`
- `eda_decomposition.png`
- `acf_pacf_raw.png`
- `acf_pacf_diff1.png`
- `acf_pacf_sdiff24.png`
- `differencing_comparison.png`
- `benchmark_forecast_comparison.png`
- `sarimax_forecast_comparison.png`
- `sarima_target_only_forecast_ci.png`
- `sarima_target_only_residual_diagnostics.png`
- `sarima_target_only_residual_acf.png`
- `sarimax_exog_residual_diagnostics.png`
- `sarimax_exog_residual_acf.png`
- `feature_importance.png`
- `feature_model_forecast_comparison.png`
- `foundation_model_forecast_comparison.png`
- `forecast_comparison.png`
- `forecast_comparison_split.png`

### Metrics

- `stationarity_tests.csv`
- `benchmark_comparison.csv`
- `order_selection.csv`
- `sarimax_residual_diagnostics.csv`
- `sarimax_comparison.csv`
- `feature_importance.csv`
- `feature_model_comparison.csv`
- `foundation_model_comparison.csv`
- `model_comparison.csv`

### Forecasts

- `benchmark_forecasts.csv`
- `sarimax_forecasts.csv`
- `sarimax_confidence_intervals.csv`
- `feature_model_forecasts.csv`
- `foundation_model_forecasts.csv`
- `all_forecasts.csv`

## Important forecast-realism note

Time-derived variables are known at the forecast origin. Future indoor sensor and weather values are not normally known. The SARIMAX-exogenous and XGBoost results therefore use realised holdout sensor/weather values and are reported as **conditional forecasts**. Chronos is target-only and does not use future covariates.

## Reproducibility

The report should be written from the numerical results and figures generated by this repository. Do not copy the reference project's report or pre-generated result values; run the pipeline and interpret your own outputs.
