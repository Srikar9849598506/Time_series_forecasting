# scripts/run_foundation_model.py

# ============================================================
# Part 5: Time-series foundation model (Chronos), zero-shot,
#         target-only forecasting.
#
# Usage:
#   python scripts/run_foundation_model.py
#
# Requires scripts/run_benchmarks.py to have been run first.
#
# Output:
#   outputs/metrics/foundation_model_comparison.csv
#   outputs/forecasts/foundation_model_forecasts.csv
#   outputs/figures/foundation_model_forecast_comparison.png
# ============================================================

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from appliance_energy.config import TARGET, TEST_STEPS, FORECAST_HORIZON, DAILY_PERIOD, FORECAST_DIR, METRICS_DIR
from appliance_energy.data import load_hourly_data, train_test_split_series
from appliance_energy.evaluation import evaluate_all, get_strongest_benchmark
from appliance_energy.plotting import plot_forecasts
from appliance_energy.models.foundation import forecast_chronos


def main():
    # --------------------------------------------------------
    # 1. Load data and split (same split as Parts 2-4)
    # --------------------------------------------------------

    hourly = load_hourly_data()
    y = hourly[TARGET]

    train, test = train_test_split_series(y, TEST_STEPS)
    horizon = min(FORECAST_HORIZON, len(test))
    test = test.iloc[:horizon]

    print("Train period:", train.index.min(), "to", train.index.max())
    print("Test period: ", test.index.min(), "to", test.index.max())

    # --------------------------------------------------------
    # 2. Chronos zero-shot forecast
    #
    # Target-only: Chronos receives only the Appliances history
    # as context, no sensor/weather covariates. This is therefore
    # NOT subject to the same forecast-realism caveat as the
    # SARIMAX-exog / feature_model results in Parts 3-4.
    # --------------------------------------------------------

    foundation_forecast = forecast_chronos(
        y_train=train,
        horizon=horizon,
        index=test.index,
    )

    # --------------------------------------------------------
    # 3. Evaluate against the strongest benchmark
    # --------------------------------------------------------

    benchmark_path = METRICS_DIR / "benchmark_comparison.csv"
    if not benchmark_path.exists():
        raise FileNotFoundError(
            "benchmark_comparison.csv not found - run "
            "scripts/run_benchmarks.py first."
        )

    benchmark_results = pd.read_csv(benchmark_path)
    strongest_name, strongest_row = get_strongest_benchmark(benchmark_results)
    print(f"\nStrongest benchmark to beat: {strongest_name} "
          f"(MASE={strongest_row['MASE']:.3f})")

    foundation_results = evaluate_all(
        {"foundation_model": foundation_forecast}, test, train, seasonality=DAILY_PERIOD,
    )

    combined_results = pd.concat(
        [foundation_results, benchmark_results[benchmark_results["model"] == strongest_name]],
        ignore_index=True,
    ).sort_values("MASE").reset_index(drop=True)

    print("\nFoundation model vs. strongest benchmark:")
    print(combined_results.round(3).to_string(index=False))

    combined_results.to_csv(METRICS_DIR / "foundation_model_comparison.csv", index=False)

    # --------------------------------------------------------
    # 4. Save forecasts and plot
    # --------------------------------------------------------

    forecast_df = pd.DataFrame({"actual": test})
    forecast_df["foundation_model"] = foundation_forecast.reindex(test.index)

    benchmark_forecasts = pd.read_csv(
        FORECAST_DIR / "benchmark_forecasts.csv", index_col=0, parse_dates=True
    )
    forecast_df[strongest_name] = benchmark_forecasts[strongest_name].reindex(test.index)

    forecast_df.to_csv(FORECAST_DIR / "foundation_model_forecasts.csv")

    plot_forecasts(
        train=train,
        test=test,
        forecast_df=forecast_df,
        title="Foundation model (Chronos) forecasts vs. strongest benchmark",
        fname="foundation_model_forecast_comparison.png",
    )

    print("\nSaved:")
    print(" ", METRICS_DIR / "foundation_model_comparison.csv")
    print(" ", FORECAST_DIR / "foundation_model_forecasts.csv")
    print(" ", "outputs/figures/foundation_model_forecast_comparison.png")

    return combined_results, forecast_df


if __name__ == "__main__":
    main()
