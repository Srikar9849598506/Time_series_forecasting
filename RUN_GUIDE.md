# Appliance Energy Forecasting - Final Run Guide

## 1. Open the project
Open the folder containing `README.md` in VS Code.

## 2. Create the Windows virtual environment
In PowerShell, from the project root:

```powershell
python --version
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If `python` is not recognised but the Python launcher is installed:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Set the package path

```powershell
$env:PYTHONPATH="$PWD\src"
```

## 5. Run preflight BEFORE the expensive SARIMA search

```powershell
python scripts\preflight.py
```

You must see `PREFLIGHT PASSED` before continuing.

## 6. Run the complete assignment pipeline

```powershell
python scripts\run_pipeline.py
```

The pipeline runs:

1. Data download, cleaning, hourly resampling, EDA and stationarity.
2. Mean, naive, daily seasonal-naive, weekly seasonal-naive and drift benchmarks.
3. Full 147-order SARIMA/SARIMAX AIC search, target-only SARIMA, weather SARIMAX, diagnostics and confidence intervals.
4. Recursive XGBoost feature model.
5. Real Chronos zero-shot forecast.
6. Final all-model comparison and final forecast plots.

## Important runtime note
The 147-order SARIMA search is intentionally required by the assignment and can take hours on a normal CPU. The search is checkpointed to:

`outputs/metrics/order_selection.csv`

If the process stops after some candidates, rerun the pipeline and the completed candidates are reused. If the file contains a completed search, the pipeline reuses it and does not repeat the 147 fits.

## Important SARIMAX fix
The project holds out 14 days (336 hours) but evaluates a 24-hour forecast. The SARIMAX code now aligns future weather variables to the exact 24 forecast timestamps, preventing:

`Required (24, 5), got (336, 5)`

## Main final outputs

- `outputs/forecasts/all_forecasts.csv`
- `outputs/metrics/model_comparison.csv`
- `outputs/figures/forecast_comparison.png`
- `outputs/figures/forecast_comparison_split.png`
- `outputs/metrics/stationarity_tests.csv`
- `outputs/metrics/order_selection.csv`
- `outputs/metrics/sarimax_residual_diagnostics.csv`
- `outputs/forecasts/sarimax_confidence_intervals.csv`
- `outputs/metrics/feature_importance.csv`

The SARIMAX-exogenous and XGBoost forecasts use realised holdout sensor/weather values and are therefore conditional forecasts; Chronos is target-only.
