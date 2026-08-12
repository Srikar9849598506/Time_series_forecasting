# src/appliance_energy/stationarity.py

# ============================================================
# Stationarity testing for the appliance energy series.
#
# Tests:
#   * Augmented Dickey-Fuller (ADF)  - H0: unit root
#   * KPSS                           - H0: stationary
#   * ACF / PACF plots
#   * First and seasonal (24 h) differencing
#
# ADF and KPSS have opposite null hypotheses, so using both
# gives a more robust conclusion than either test alone.
# ============================================================

import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from appliance_energy.config import FIGURE_DIR, DAILY_PERIOD


def adf_test(series, name=""):
    """
    Augmented Dickey-Fuller test.

    H0: the series has a unit root (non-stationary).
    p < 0.05 -> reject H0 -> evidence of stationarity.
    """

    stat, pvalue, usedlag, nobs, crit, _ = adfuller(
        series.dropna(), autolag="AIC"
    )

    conclusion = (
        "stationary (reject H0)" if pvalue < 0.05
        else "non-stationary (fail to reject H0)"
    )

    print(f"\nADF test - {name}")
    print(f"  statistic = {stat:.4f}")
    print(f"  p-value   = {pvalue:.4g}")
    print(f"  lags used = {usedlag}, n = {nobs}")
    for key, value in crit.items():
        print(f"  critical value {key}: {value:.4f}")
    print(f"  => {conclusion}")

    return {"test": "ADF", "series": name, "statistic": stat,
            "p_value": pvalue, "conclusion": conclusion}


def kpss_test(series, name=""):
    """
    KPSS level-stationarity test (complements ADF).

    H0: the series is stationary.
    p < 0.05 -> reject H0 -> evidence of NON-stationarity.
    """

    stat, pvalue, lags, crit = kpss(
        series.dropna(), regression="c", nlags="auto"
    )

    conclusion = (
        "non-stationary (reject H0)" if pvalue < 0.05
        else "stationary (fail to reject H0)"
    )

    print(f"\nKPSS test - {name}")
    print(f"  statistic = {stat:.4f}")
    print(f"  p-value   = {pvalue:.4g}")
    print(f"  => {conclusion}")

    return {"test": "KPSS", "series": name, "statistic": stat,
            "p_value": pvalue, "conclusion": conclusion}


def plot_acf_pacf(series, name, lags=72, fname=None):
    """
    ACF and PACF out to three days of lags so daily
    seasonality (spikes at lags 24, 48, 72) is visible.
    """

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF - {name}")
    axes[0].set_xlabel("Lag (hours)")

    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF - {name}")
    axes[1].set_xlabel("Lag (hours)")

    fig.tight_layout()

    if fname is not None:
        fig.savefig(FIGURE_DIR / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def run_stationarity_analysis(y, seasonality=DAILY_PERIOD):
    """
    Full stationarity workflow on the target series:

      1. Raw series:            ADF + KPSS + ACF/PACF
      2. First difference:      ADF + KPSS + ACF/PACF
      3. Seasonal difference:   ADF + KPSS + ACF/PACF

    Returns a tidy DataFrame of test results, saved for the report.
    """

    results = []

    # --- 1. Raw series
    results.append(adf_test(y, "raw series"))
    results.append(kpss_test(y, "raw series"))
    plot_acf_pacf(y, "raw hourly series", fname="acf_pacf_raw.png")

    # --- 2. First difference
    y_diff = y.diff()
    results.append(adf_test(y_diff, "first difference"))
    results.append(kpss_test(y_diff, "first difference"))
    plot_acf_pacf(y_diff, "first difference", fname="acf_pacf_diff1.png")

    # --- 3. Seasonal difference (lag 24)
    y_sdiff = y.diff(seasonality)
    results.append(adf_test(y_sdiff, f"seasonal difference ({seasonality}h)"))
    results.append(kpss_test(y_sdiff, f"seasonal difference ({seasonality}h)"))
    plot_acf_pacf(
        y_sdiff,
        f"seasonal difference ({seasonality}h)",
        fname="acf_pacf_sdiff24.png",
    )

    # --- Side-by-side plot of the three series
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(y, lw=0.6)
    axes[0].set_title("Raw hourly series")
    axes[0].set_ylabel("Wh")

    axes[1].plot(y_diff, lw=0.6)
    axes[1].set_title("First difference")
    axes[1].set_ylabel("Wh")

    axes[2].plot(y_sdiff, lw=0.6)
    axes[2].set_title(f"Seasonal difference ({seasonality}h)")
    axes[2].set_ylabel("Wh")
    axes[2].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "differencing_comparison.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(results)
