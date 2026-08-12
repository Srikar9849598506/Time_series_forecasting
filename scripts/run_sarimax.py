# scripts/run_sarimax.py

# ============================================================
# Part 3: SARIMA (target-only) and SARIMAX (with exogenous
#         weather variables) modelling.
#
# Usage:
#   python scripts/run_sarimax.py
#
# Requires scripts/run_benchmarks.py to have been run first,
# since this script compares against outputs/metrics/benchmark_comparison.csv
#
# Output:
#   outputs/metrics/order_selection.csv
#   outputs/metrics/sarimax_residual_diagnostics.csv
#   outputs/metrics/sarimax_comparison.csv
#   outputs/forecasts/sarimax_forecasts.csv
#   outputs/figures/sarima_target_only_residual_diagnostics.png
#   outputs/figures/sarimax_exog_residual_diagnostics.png
#   outputs/figures/sarimax_forecast_comparison.png
# ============================================================

import sys
import warnings
from ast import literal_eval
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from appliance_energy.config import (
    TARGET, TEST_STEPS, FORECAST_HORIZON, DAILY_PERIOD, EXOG_COLS,
    FORECAST_DIR, METRICS_DIR,
)
from appliance_energy.data import load_hourly_data, train_test_split_series
from appliance_energy.evaluation import evaluate_all, get_strongest_benchmark
from appliance_energy.plotting import plot_forecasts, plot_forecast_with_ci
from appliance_energy.models.sarimax import (
    fit_sarimax,
    forecast_sarimax,
    select_order_by_aic,
    build_full_pdq_grid,
    residual_diagnostics,
)


def main():
    # --------------------------------------------------------
    # 1. Load data and split (same split as Part 2)
    # --------------------------------------------------------

    hourly = load_hourly_data()
    y = hourly[TARGET]

    train, test = train_test_split_series(y, TEST_STEPS)
    horizon = min(FORECAST_HORIZON, len(test))
    test = test.iloc[:horizon]

    print("Train period:", train.index.min(), "to", train.index.max())
    print("Test period: ", test.index.min(), "to", test.index.max())

    # --------------------------------------------------------
    # 2. Order selection
    #
    # Per the assignment spec (Part 4): "loop over all possible
    # parameter combinations for p=[0,6], d=[0,2] and q=[0,6]" and
    # select by AIC. The seasonal order is held fixed at (1,1,1,24),
    # justified by the Part 1 finding that the ACF shows slow,
    # periodic decay with spikes at lags 24/48/72 -> D=1, s=24
    # removes the daily periodicity; searching the seasonal order
    # too would multiply the grid size well beyond what is tractable
    # on an hourly series with a 24-period season.
    #
    # This is a 7 x 3 x 7 = 147-model grid search, so it is run with
    # a reduced maxiter for speed (see select_order_by_aic); the
    # winning order is then refit below with the full maxiter=200
    # for the actual forecast.
    # --------------------------------------------------------

    seasonal_order = (1, 1, 1, DAILY_PERIOD)

    order_path = METRICS_DIR / "order_selection.csv"

    if order_path.exists():
        # Resume from a previous run: the 147-model grid search is by
        # far the slowest step (hours), so if it already completed and
        # was saved, reuse it rather than repeating it.
        print(f"\nFound existing {order_path} - reusing it instead of "
              f"re-running the grid search. Delete this file first if "
              f"you want to force a fresh search.")
        order_results = pd.read_csv(order_path)
        # 'order' was saved as a string like "(6, 0, 0)" - eval it back
        # to a tuple so it can be passed straight to fit_sarimax.
        order_results["order"] = order_results["order"].apply(literal_eval)
    else:
        candidate_orders = build_full_pdq_grid(
            p_range=range(0, 7), d_range=range(0, 3), q_range=range(0, 7)
        )
        print(f"\nOrder selection (AIC) - full grid search over "
              f"{len(candidate_orders)} (p, d, q) combinations "
              f"(p in [0,6], d in [0,2], q in [0,6]), "
              f"seasonal_order={seasonal_order} fixed...")

        order_results = select_order_by_aic(
            train, candidate_orders, seasonal_order=seasonal_order,
            checkpoint_path=order_path,
        )
        order_results.to_csv(order_path, index=False)

    best_order = order_results.iloc[0]["order"]
    print(f"\nSelected order by AIC: {best_order} "
          f"(AIC={order_results.iloc[0]['AIC']:.1f})")
    print("\nTop 10 candidate orders by AIC:")
    print(order_results.head(10).to_string(index=False))

    # --------------------------------------------------------
    # 3. Target-only SARIMA
    # --------------------------------------------------------

    print("\nFitting target-only SARIMA...")
    sarima_fit = fit_sarimax(
        y_train=train,
        exog_train=None,
        order=best_order,
        seasonal_order=seasonal_order,
    )
    print(sarima_fit.summary().tables[0])

    sarima_forecast, sarima_conf_int = forecast_sarimax(
        sarima_fit, horizon, test.index, name="sarima_target_only"
    )

    diag_target_only = residual_diagnostics(
        sarima_fit, name="SARIMA (target-only)", fname_prefix="sarima_target_only"
    )

    # --------------------------------------------------------
    # 4. SARIMAX with exogenous weather variables
    # --------------------------------------------------------

    exog_cols = [c for c in EXOG_COLS if c in hourly.columns]
    print(f"\nSARIMAX exogenous columns: {exog_cols}")

    exog = hourly[exog_cols]
    exog_train, exog_holdout = train_test_split_series(exog, TEST_STEPS)
    # The common holdout is 14 days (336 hours), but the assignment
    # evaluates a 24-hour forecast. Keep the full holdout for data
    # bookkeeping, then pass only the exact forecast horizon to
    # statsmodels. Timestamp alignment is also performed inside
    # forecast_sarimax().
    exog_test = exog_holdout.iloc[:horizon].copy()

    # NOTE on data leakage / forecast realism (see README):
    # this uses the *realised* test-period weather values as
    # exogenous inputs. That is not available at forecast origin
    # in a real deployment, so this result should be read as a
    # conditional forecast (an upper bound on what weather-aware
    # SARIMAX could achieve), not an operational forecast. A
    # deployable version would need forecast weather values.
    print("Fitting SARIMAX with weather exogenous variables...")
    sarimax_fit = fit_sarimax(
        y_train=train,
        exog_train=exog_train,
        order=best_order,
        seasonal_order=seasonal_order,
    )

    sarimax_forecast, sarimax_conf_int = forecast_sarimax(
        sarimax_fit, horizon, test.index, exog_test=exog_test, name="sarimax_exog"
    )

    diag_exog = residual_diagnostics(
        sarimax_fit, name="SARIMAX (with weather exog)", fname_prefix="sarimax_exog"
    )

    diagnostics_df = pd.DataFrame([diag_target_only, diag_exog])
    diagnostics_df.to_csv(
        METRICS_DIR / "sarimax_residual_diagnostics.csv", index=False
    )

    # --------------------------------------------------------
    # 5. Evaluate against the strongest benchmark
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

    sarimax_forecasts = {
        "sarima_target_only": sarima_forecast,
        "sarimax_exog": sarimax_forecast,
    }

    sarimax_results = evaluate_all(sarimax_forecasts, test, train, seasonality=DAILY_PERIOD)

    combined_results = pd.concat(
        [sarimax_results, benchmark_results[benchmark_results["model"] == strongest_name]],
        ignore_index=True,
    ).sort_values("MASE").reset_index(drop=True)

    print("\nSARIMA/SARIMAX vs. strongest benchmark:")
    print(combined_results.round(3).to_string(index=False))

    combined_results.to_csv(METRICS_DIR / "sarimax_comparison.csv", index=False)

    # --------------------------------------------------------
    # 6. Save forecasts and plot
    # --------------------------------------------------------

    forecast_df = pd.DataFrame({"actual": test})
    forecast_df["sarima_target_only"] = sarima_forecast
    forecast_df["sarimax_exog"] = sarimax_forecast
    forecast_df[strongest_name] = pd.read_csv(
        FORECAST_DIR / "benchmark_forecasts.csv", index_col=0, parse_dates=True
    )[strongest_name].reindex(test.index)

    forecast_df.to_csv(FORECAST_DIR / "sarimax_forecasts.csv")

    plot_forecasts(
        train=train,
        test=test,
        forecast_df=forecast_df,
        title="SARIMA/SARIMAX forecasts vs. strongest benchmark",
        fname="sarimax_forecast_comparison.png",
    )

    # --------------------------------------------------------
    # 7. Confidence intervals (assignment Part 4: "Add confidence
    #    intervals on forecasts"). Saved as CSVs and plotted for
    #    the target-only model (the one recommended in the report).
    # --------------------------------------------------------

    conf_int_df = sarima_conf_int.join(sarimax_conf_int)
    conf_int_df.to_csv(FORECAST_DIR / "sarimax_confidence_intervals.csv")

    plot_forecast_with_ci(
        train=train,
        test=test,
        forecast=sarima_forecast,
        conf_int=sarima_conf_int,
        title="SARIMA (target-only) forecast with 95% confidence interval",
        fname="sarima_target_only_forecast_ci.png",
    )

    print("\nSaved:")
    print(" ", METRICS_DIR / "order_selection.csv")
    print(" ", METRICS_DIR / "sarimax_residual_diagnostics.csv")
    print(" ", METRICS_DIR / "sarimax_comparison.csv")
    print(" ", FORECAST_DIR / "sarimax_forecasts.csv")
    print(" ", FORECAST_DIR / "sarimax_confidence_intervals.csv")
    print(" ", "outputs/figures/sarimax_forecast_comparison.png")
    print(" ", "outputs/figures/sarima_target_only_forecast_ci.png")

    return combined_results, forecast_df


if __name__ == "__main__":
    main()