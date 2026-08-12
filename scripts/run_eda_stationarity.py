# scripts/run_eda_stationarity.py

# ============================================================
# Part 1: Data preparation, exploratory analysis and
#         stationarity testing.
#
# Usage:
#   python scripts/run_eda_stationarity.py
#
# Output:
#   data/processed/appliance_hourly.csv
#   outputs/figures/eda_*.png
#   outputs/figures/acf_pacf_*.png
#   outputs/figures/differencing_comparison.png
#   outputs/metrics/stationarity_tests.csv
# ============================================================

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Make src/ importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display

from appliance_energy.config import TARGET, METRICS_DIR
from appliance_energy.data import (
    download_raw_data,
    load_raw_data,
    report_missing_and_gaps,
    resample_to_hourly,
)
from appliance_energy.plotting import (
    plot_series_overview,
    plot_seasonal_profiles,
    plot_decomposition,
)
from appliance_energy.stationarity import run_stationarity_analysis


def main():
    # --------------------------------------------------------
    # 1. Load and prepare the data
    # --------------------------------------------------------

    download_raw_data()
    raw = load_raw_data()

    print("Raw data shape:", raw.shape)

    report_missing_and_gaps(raw)

    hourly = resample_to_hourly(raw)

    # --------------------------------------------------------
    # 2. Exploratory analysis
    # --------------------------------------------------------

    plot_series_overview(hourly)
    plot_seasonal_profiles(hourly)
    plot_decomposition(hourly)

    print("\nEDA figures saved to outputs/figures/")

    # --------------------------------------------------------
    # 3. Stationarity testing
    # --------------------------------------------------------

    results = run_stationarity_analysis(hourly[TARGET])

    results.to_csv(METRICS_DIR / "stationarity_tests.csv", index=False)

    print("\nStationarity test summary:")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
