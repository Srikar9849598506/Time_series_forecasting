# scripts/run_feature_model.py

# ============================================================
# Part 4: Feature-based machine-learning model (XGBoost).
#
# Usage:
#   python scripts/run_feature_model.py
#
# Requires scripts/run_benchmarks.py to have been run first.
#
# Output:
#   outputs/metrics/feature_model_comparison.csv
#   outputs/forecasts/feature_model_forecasts.csv
#   outputs/figures/feature_importance.png
#   outputs/figures/feature_model_forecast_comparison.png
# ============================================================

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from appliance_energy.config import (
    TARGET, TEST_STEPS, FORECAST_HORIZON, DAILY_PERIOD, FORECAST_DIR, METRICS_DIR,
)
from appliance_energy.data import load_hourly_data, train_test_split_series
from appliance_energy.features import make_feature_table, get_feature_columns
from appliance_energy.evaluation import evaluate_all, get_strongest_benchmark
from appliance_energy.plotting import plot_forecasts
from appliance_energy.models.feature_models import (
    fit_feature_model,
    forecast_feature_model,
    get_feature_importance,
    plot_feature_importance,
)


def main():
    hourly = load_hourly_data()
    full_y = hourly[TARGET]
    full_train, holdout = train_test_split_series(full_y, TEST_STEPS)
    horizon = min(FORECAST_HORIZON, len(holdout))
    test = holdout.iloc[:horizon]

    print("NOTE: future sensor/weather columns used below are realised values from the holdout.")
    print("This is a CONDITIONAL forecast, not a fully operational forecast.")

    # Training feature table uses only observations before the 14-day holdout.
    train_frame = hourly.loc[:full_train.index[-1]].copy()
    train_table = make_feature_table(train_frame, target=TARGET)
    feature_cols = get_feature_columns(train_table, target=TARGET)

    X_train = train_table[feature_cols]
    y_train = train_table[TARGET]
    model = fit_feature_model(X_train, y_train)

    # Recursive 24-hour forecast. For each future timestamp, target lags and
    # rolling windows use only observed history plus predictions already made
    # during this forecast. Future sensor/weather values are the realised
    # holdout covariates and are therefore explicitly conditional.
    history = full_train.copy()
    predictions = []
    rows = []

    for timestamp in test.index:
        row = hourly.loc[timestamp].copy()
        row_dict = row.to_dict()
        row_dict["hour"] = timestamp.hour
        row_dict["dayofweek"] = timestamp.dayofweek
        row_dict["is_weekend"] = int(timestamp.dayofweek >= 5)
        row_dict["hour_sin"] = np.sin(2 * np.pi * timestamp.hour / 24)
        row_dict["hour_cos"] = np.cos(2 * np.pi * timestamp.hour / 24)
        row_dict["dow_sin"] = np.sin(2 * np.pi * timestamp.dayofweek / 7)
        row_dict["dow_cos"] = np.cos(2 * np.pi * timestamp.dayofweek / 7)

        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            row_dict[f"lag_{lag}"] = history.iloc[-lag]
        shifted = history
        for window in [3, 6, 12, 24, 168]:
            row_dict[f"roll_mean_{window}"] = shifted.iloc[-window:].mean()
            row_dict[f"roll_std_{window}"] = shifted.iloc[-window:].std()

        feature_row = pd.DataFrame([row_dict], index=[timestamp])
        X_next = feature_row[feature_cols]
        pred = float(model.predict(X_next)[0])
        predictions.append(pred)
        history = pd.concat([history, pd.Series([pred], index=[timestamp], name=TARGET)])
        rows.append(feature_row.iloc[0])

    feature_forecast = pd.Series(predictions, index=test.index, name="feature_model")

    importance_df = get_feature_importance(model, feature_cols)
    importance_df.to_csv(METRICS_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(importance_df)

    benchmark_results = pd.read_csv(METRICS_DIR / "benchmark_comparison.csv")
    strongest_name, strongest_row = get_strongest_benchmark(benchmark_results)
    print(f"\nStrongest benchmark: {strongest_name} (MASE={strongest_row['MASE']:.3f})")

    feature_results = evaluate_all(
        {"feature_model": feature_forecast}, test, full_train,
        seasonality=DAILY_PERIOD,
    )
    combined_results = pd.concat(
        [feature_results, benchmark_results[benchmark_results["model"] == strongest_name]],
        ignore_index=True,
    ).sort_values("MASE").reset_index(drop=True)
    combined_results.to_csv(METRICS_DIR / "feature_model_comparison.csv", index=False)

    forecast_df = pd.DataFrame({"actual": test, "feature_model": feature_forecast})
    benchmark_forecasts = pd.read_csv(
        FORECAST_DIR / "benchmark_forecasts.csv", index_col=0, parse_dates=True
    )
    forecast_df[strongest_name] = benchmark_forecasts[strongest_name].reindex(test.index)
    forecast_df.to_csv(FORECAST_DIR / "feature_model_forecasts.csv")

    plot_forecasts(
        train=full_train, test=test, forecast_df=forecast_df,
        title="Feature-based model forecasts vs. strongest benchmark",
        fname="feature_model_forecast_comparison.png",
    )

    print("\nFeature model comparison:")
    print(combined_results.round(3).to_string(index=False))
    print("\nSaved feature-based model outputs.")
    return combined_results, forecast_df


if __name__ == "__main__":
    main()
