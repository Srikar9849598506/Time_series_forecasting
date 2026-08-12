# src/appliance_energy/models/benchmarks.py

# ============================================================
# Simple benchmark forecasting models.
#
# Every advanced model (SARIMAX, feature-based, foundation) is
# judged against the strongest of these benchmarks, not just
# against each other - a model that cannot beat seasonal naive
# forecasting is not adding real value.
# ============================================================

import pandas as pd


def mean_forecast(y_train, horizon, index):
    """
    Forecast every future step as the training mean.
    The weakest benchmark; establishes a floor on performance.
    """
    return pd.Series(y_train.mean(), index=index, name="mean")


def naive_forecast(y_train, horizon, index):
    """
    Forecast every future step as the last observed value.
    Strong for series with little seasonality; weak here because
    appliance use has a pronounced daily cycle.
    """
    return pd.Series(y_train.iloc[-1], index=index, name="naive")


def seasonal_naive_forecast(y_train, horizon, index, seasonality):
    """
    Recursive seasonal naive forecast.

    seasonality=24  -> same hour yesterday
    seasonality=168 -> same hour last week

    Recursive because, beyond the first `seasonality` steps, the
    forecast needs values that are themselves forecasts (there is
    no more real history that far ahead within the horizon).
    """

    values = []
    history = list(y_train.values)

    for _ in range(horizon):
        values.append(history[-seasonality])
        history.append(values[-1])

    name = "seasonal_naive_daily" if seasonality == 24 else "seasonal_naive_weekly"

    return pd.Series(values, index=index, name=name)


def drift_forecast(y_train, horizon, index):
    """
    Naive forecast plus a linear trend estimated from the average
    step-to-step change over the whole training set.
    """

    slope = (y_train.iloc[-1] - y_train.iloc[0]) / (len(y_train) - 1)

    values = [
        y_train.iloc[-1] + slope * step
        for step in range(1, horizon + 1)
    ]

    return pd.Series(values, index=index, name="drift")


def generate_all_benchmarks(train, horizon, index, daily_period=24, weekly_period=168):
    """
    Convenience function: generate every benchmark forecast at once,
    returned as a dict keyed by model name (matches the schema
    expected in outputs/forecasts/all_forecasts.csv).
    """

    forecasts = {
        "mean": mean_forecast(train, horizon, index),
        "naive": naive_forecast(train, horizon, index),
        "seasonal_naive_daily": seasonal_naive_forecast(
            train, horizon, index, seasonality=daily_period
        ),
        "seasonal_naive_weekly": seasonal_naive_forecast(
            train, horizon, index, seasonality=weekly_period
        ),
        "drift": drift_forecast(train, horizon, index),
    }

    return forecasts
