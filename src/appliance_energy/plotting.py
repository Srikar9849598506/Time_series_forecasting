# src/appliance_energy/plotting.py

# ============================================================
# Exploratory, diagnostic, and forecast plots.
#
# Every figure is saved to outputs/figures/ at 300 dpi with
# labelled axes so it can be dropped straight into the report.
# ============================================================

import matplotlib.pyplot as plt

from appliance_energy.config import FIGURE_DIR, TARGET, DAILY_PERIOD


def plot_series_overview(hourly, target=TARGET):
    """
    Three-panel overview of the target:
      1. Full hourly series
      2. One-week zoom (exposes the daily cycle)
      3. Histogram (exposes the right skew)
    """

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    axes[0].plot(hourly.index, hourly[target], lw=0.6)
    axes[0].set_title("Hourly appliance energy use - full series")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Energy use (Wh)")

    week = hourly[target].iloc[: DAILY_PERIOD * 7]
    axes[1].plot(week.index, week.values)
    axes[1].set_title("First week - daily cycle")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Energy use (Wh)")

    axes[2].hist(hourly[target], bins=60, edgecolor="k", alpha=0.7)
    axes[2].set_title("Distribution of hourly appliance energy use")
    axes[2].set_xlabel("Energy use (Wh)")
    axes[2].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_overview.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    return fig


def plot_seasonal_profiles(hourly, target=TARGET):
    """
    Mean usage by hour-of-day and by day-of-week.
    These profiles show the two seasonal components directly.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    by_hour = hourly.groupby(hourly.index.hour)[target].mean()
    by_hour.plot(ax=axes[0], marker="o")
    axes[0].set_title("Mean usage by hour of day")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Mean energy use (Wh)")

    by_dow = hourly.groupby(hourly.index.dayofweek)[target].mean()
    by_dow.plot(ax=axes[1], marker="o")
    axes[1].set_title("Mean usage by day of week (0 = Monday)")
    axes[1].set_xlabel("Day of week")
    axes[1].set_ylabel("Mean energy use (Wh)")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_seasonal_profiles.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    return fig


def plot_decomposition(hourly, target=TARGET, period=DAILY_PERIOD):
    """
    Classical additive decomposition with a daily period,
    separating trend, seasonal and residual components.
    """

    # Imported here so the other plotting functions do not require
    # statsmodels to be installed.
    from statsmodels.tsa.seasonal import seasonal_decompose

    decomp = seasonal_decompose(
        hourly[target], model="additive", period=period
    )

    fig = decomp.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle(f"Additive decomposition (period = {period} h)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_decomposition.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    return decomp


def plot_forecasts(train, test, forecast_df, title="Appliance energy forecasting",
                    context_steps=14 * DAILY_PERIOD, fname=None):
    """
    Plot recent training data, the test period actuals, and one or
    more forecasts overlaid, for visual comparison of model fit.

    forecast_df: DataFrame indexed like `test`, one column per model
    (an 'actual' column, if present, is skipped - `test` is plotted
    instead so the actual line is always styled consistently).
    """

    fig, ax = plt.subplots(figsize=(14, 7))

    train.tail(context_steps).plot(
        ax=ax, label="Training data", linewidth=1.5, color="tab:gray"
    )

    test.plot(ax=ax, label="Test data (actual)", linewidth=2.0, color="black")

    for col in forecast_df.columns:
        if col == "actual":
            continue
        forecast_df[col].plot(ax=ax, label=col, alpha=0.85)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliance energy use (Wh)")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()

    if fname is not None:
        fig.savefig(FIGURE_DIR / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_forecast_with_ci(train, test, forecast, conf_int,
                           title="SARIMAX forecast with 95% confidence interval",
                           context_steps=14 * DAILY_PERIOD, fname=None):
    """
    Plot a single point forecast together with its shaded confidence
    interval band, against recent training context and the actual test
    data. Satisfies the assignment's Part 4 requirement to show
    confidence intervals on the SARIMAX forecast.

    conf_int: two-column DataFrame (lower, upper), same index as forecast.
    """

    fig, ax = plt.subplots(figsize=(14, 7))

    train.tail(context_steps).plot(
        ax=ax, label="Training data", linewidth=1.2, color="tab:gray", alpha=0.7
    )
    test.plot(ax=ax, label="Test data (actual)", linewidth=2.0, color="black")
    forecast.plot(ax=ax, label=forecast.name, color="tab:red", linewidth=1.6)

    lower_col, upper_col = conf_int.columns
    ax.fill_between(
        conf_int.index, conf_int[lower_col], conf_int[upper_col],
        color="tab:red", alpha=0.2, label="95% confidence interval",
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliance energy use (Wh)")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()

    if fname is not None:
        fig.savefig(FIGURE_DIR / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_forecast_comparison_split(train, test, forecast_df,
                                    weak_models=None, strong_models=None,
                                    context_steps=14 * DAILY_PERIOD,
                                    fname="forecast_comparison_split.png"):
    """
    Two-panel version of the full model comparison, split into:
      - top panel:    weak/simple benchmarks (mean, naive, drift,
                       seasonal naive) vs. actual
      - bottom panel: the stronger, dataset-aware/foundation models
                       (SARIMA/SARIMAX, feature_model, foundation_model)
                       vs. actual

    Splitting avoids the 9-line, similar-colour legend of the single
    combined plot, which is hard to read at report/print size once
    every model is overlaid together.
    """

    if weak_models is None:
        weak_models = [
            m for m in ["mean", "naive", "drift",
                        "seasonal_naive_daily", "seasonal_naive_weekly"]
            if m in forecast_df.columns
        ]

    if strong_models is None:
        strong_models = [
            m for m in ["sarima_target_only", "sarimax_exog",
                        "feature_model", "foundation_model"]
            if m in forecast_df.columns
        ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    for ax, models, subtitle in [
        (axes[0], weak_models, "Simple benchmarks"),
        (axes[1], strong_models, "SARIMA/SARIMAX, feature-based, and foundation models"),
    ]:
        train.tail(context_steps).plot(
            ax=ax, label="Training data", linewidth=1.2, color="tab:gray", alpha=0.6
        )
        test.plot(ax=ax, label="Test data (actual)", linewidth=2.2, color="black")

        for col in models:
            forecast_df[col].plot(ax=ax, label=col, alpha=0.85, linewidth=1.4)

        ax.set_title(subtitle, fontsize=12)
        ax.set_ylabel("Appliance energy use (Wh)")
        ax.legend(loc="upper left", fontsize=9, ncol=2)

    axes[1].set_xlabel("Date")
    fig.suptitle("Appliance energy forecasting - all models (split view)", y=1.0, fontsize=14)
    fig.tight_layout()

    if fname is not None:
        fig.savefig(FIGURE_DIR / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig