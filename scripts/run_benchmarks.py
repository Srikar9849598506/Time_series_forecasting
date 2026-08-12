# scripts/run_benchmarks.py

# ============================================================
# Part 2: Train/test split and benchmark forecasting models.
#
# Usage:
#   python scripts/run_benchmarks.py
#
# Output:
#   outputs/forecasts/benchmark_forecasts.csv
#   outputs/metrics/benchmark_comparison.csv
#   outputs/figures/benchmark_forecast_comparison.png
# ============================================================

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from appliance_energy.config import (
    TARGET, TEST_STEPS, FORECAST_HORIZON, DAILY_PERIOD, WEEKLY_PERIOD,
    FORECAST_DIR, METRICS_DIR,
)
from appliance_energy.data import load_hourly_data, train_test_split_series
from appliance_energy.models.benchmarks import generate_all_benchmarks
from appliance_energy.evaluation import evaluate_all
from appliance_energy.plotting import plot_forecasts


def main():
    # --------------------------------------------------------
    # 1. Load data and split chronologically
    # --------------------------------------------------------

    hourly = load_hourly_data()
    y = hourly[TARGET]

    train, test = train_test_split_series(y, TEST_STEPS)
    horizon = min(FORECAST_HORIZON, len(test))
    test = test.iloc[:horizon]

    print("Train period:", train.index.min(), "to", train.index.max())
    print("Test period: ", test.index.min(), "to", test.index.max())
    print(f"14-day holdout size: {TEST_STEPS} hours; forecast evaluation horizon: {horizon} hours")

    # --------------------------------------------------------
    # 2. Generate benchmark forecasts
    # --------------------------------------------------------

    forecasts = generate_all_benchmarks(
        train=train,
        horizon=horizon,
        index=test.index,
        daily_period=DAILY_PERIOD,
        weekly_period=WEEKLY_PERIOD,
    )

    # --------------------------------------------------------
    # 3. Evaluate
    # --------------------------------------------------------

    results = evaluate_all(forecasts, test, train, seasonality=DAILY_PERIOD)

    print("\nBenchmark model comparison (sorted by MASE):")
    print(results.round(3).to_string(index=False))

    strongest = results.iloc[0]["model"]
    print(f"\nStrongest benchmark: {strongest}")
    print("Every later model (SARIMAX, feature-based, foundation) "
          "must be compared against this benchmark, not just against "
          "each other.")

    # --------------------------------------------------------
    # 4. Save outputs
    # --------------------------------------------------------

    forecast_df = pd.DataFrame({"actual": test})
    for name, pred in forecasts.items():
        forecast_df[name] = pred.reindex(test.index)

    forecast_df.to_csv(FORECAST_DIR / "benchmark_forecasts.csv")
    results.to_csv(METRICS_DIR / "benchmark_comparison.csv", index=False)

    plot_forecasts(
        train=train,
        test=test,
        forecast_df=forecast_df,
        title="Benchmark forecasts vs. actual appliance energy use",
        fname="benchmark_forecast_comparison.png",
    )

    print("\nSaved:")
    print(" ", FORECAST_DIR / "benchmark_forecasts.csv")
    print(" ", METRICS_DIR / "benchmark_comparison.csv")
    print(" ", "outputs/figures/benchmark_forecast_comparison.png")

    return results, forecast_df


if __name__ == "__main__":
    main()
