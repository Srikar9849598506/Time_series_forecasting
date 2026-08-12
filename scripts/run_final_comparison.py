# scripts/run_final_comparison.py

# ============================================================
# Part 6: Final combined comparison across every model class.
#
# Run this LAST, after all of:
#   run_benchmarks.py
#   run_sarimax.py
#   run_feature_model.py
#   run_foundation_model.py
#
# This is the "main pipeline" deliverable requested in the
# README: it assembles every forecast onto a single test index,
# evaluates all of them on identical metrics, and produces the
# one plot with all models overlaid together, as required by the
# assignment's "final plot of all modelling and forecasts" mark.
#
# Usage:
#   python scripts/run_final_comparison.py
#
# Output:
#   outputs/forecasts/all_forecasts.csv
#   outputs/metrics/model_comparison.csv
#   outputs/figures/forecast_comparison.png
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
from appliance_energy.evaluation import evaluate_all
from appliance_energy.plotting import plot_forecasts, plot_forecast_comparison_split


REQUIRED_FILES = {
    "benchmarks": FORECAST_DIR / "benchmark_forecasts.csv",
    "sarimax": FORECAST_DIR / "sarimax_forecasts.csv",
    "feature_model": FORECAST_DIR / "feature_model_forecasts.csv",
    "foundation_model": FORECAST_DIR / "foundation_model_forecasts.csv",
}


def main():
    missing = [name for name, path in REQUIRED_FILES.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing forecast files from: {missing}. "
            "Run run_benchmarks.py, run_sarimax.py, run_feature_model.py "
            "and run_foundation_model.py first."
        )

    # --------------------------------------------------------
    # 1. Reload the common test set
    # --------------------------------------------------------

    hourly = load_hourly_data()
    y = hourly[TARGET]
    train, holdout = train_test_split_series(y, TEST_STEPS)
    test = holdout.iloc[:min(FORECAST_HORIZON, len(holdout))]

    # --------------------------------------------------------
    # 2. Collect every forecast onto one DataFrame
    #
    # Each Part's script already saved its own forecasts,
    # possibly alongside the benchmark it happened to compare
    # itself to - only pull out the model-specific column from
    # each file to avoid duplicating benchmark columns.
    # --------------------------------------------------------

    benchmark_forecasts = pd.read_csv(
        REQUIRED_FILES["benchmarks"], index_col=0, parse_dates=True
    )
    sarimax_forecasts = pd.read_csv(
        REQUIRED_FILES["sarimax"], index_col=0, parse_dates=True
    )
    feature_forecasts = pd.read_csv(
        REQUIRED_FILES["feature_model"], index_col=0, parse_dates=True
    )
    foundation_forecasts = pd.read_csv(
        REQUIRED_FILES["foundation_model"], index_col=0, parse_dates=True
    )

    all_forecasts = pd.DataFrame({"actual": test})

    # Every benchmark model
    for col in ["mean", "naive", "seasonal_naive_daily",
                "seasonal_naive_weekly", "drift"]:
        if col in benchmark_forecasts.columns:
            all_forecasts[col] = benchmark_forecasts[col].reindex(test.index)

    # SARIMA/SARIMAX
    for col in ["sarima_target_only", "sarimax_exog"]:
        if col in sarimax_forecasts.columns:
            all_forecasts[col] = sarimax_forecasts[col].reindex(test.index)

    # Feature-based model
    all_forecasts["feature_model"] = feature_forecasts["feature_model"].reindex(test.index)

    # Foundation model
    all_forecasts["foundation_model"] = foundation_forecasts["foundation_model"].reindex(test.index)

    all_forecasts.to_csv(FORECAST_DIR / "all_forecasts.csv")

    # --------------------------------------------------------
    # 3. Evaluate every model on identical metrics
    # --------------------------------------------------------

    model_cols = [c for c in all_forecasts.columns if c != "actual"]
    forecasts_dict = {col: all_forecasts[col] for col in model_cols}

    results = evaluate_all(forecasts_dict, test, train, seasonality=DAILY_PERIOD)

    print("\nFinal model comparison (all models, sorted by MASE):")
    print(results.round(3).to_string(index=False))

    results.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    best_model = results.iloc[0]["model"]
    strongest_benchmark_row = results[
        results["model"].isin(["mean", "naive", "seasonal_naive_daily",
                                "seasonal_naive_weekly", "drift"])
    ].iloc[0]

    print(f"\nBest overall model: {best_model}")
    print(f"Strongest benchmark: {strongest_benchmark_row['model']} "
          f"(MASE={strongest_benchmark_row['MASE']:.3f})")

    # --------------------------------------------------------
    # 4. Final combined plot
    # --------------------------------------------------------

    plot_forecasts(
        train=train,
        test=test,
        forecast_df=all_forecasts,
        title="Appliance energy forecasting - all models",
        fname="forecast_comparison.png",
    )

    plot_forecast_comparison_split(
        train=train,
        test=test,
        forecast_df=all_forecasts,
        fname="forecast_comparison_split.png",
    )

    print("\nSaved:")
    print(" ", FORECAST_DIR / "all_forecasts.csv")
    print(" ", METRICS_DIR / "model_comparison.csv")
    print(" ", "outputs/figures/forecast_comparison.png")
    print(" ", "outputs/figures/forecast_comparison_split.png")

    return results, all_forecasts


if __name__ == "__main__":
    main()
