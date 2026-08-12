# tests/test_benchmarks.py

# ============================================================
# Tests for src/appliance_energy/models/benchmarks.py
#
# Covers the README-promised "forecast lengths match the test
# period" check, plus one behavioural test per benchmark so a
# broken implementation (e.g. an off-by-one in the seasonal-naive
# recursion) would be caught rather than silently changing report
# numbers.
# ============================================================

import numpy as np
import pandas as pd
import pytest

from appliance_energy.models.benchmarks import (
    mean_forecast,
    naive_forecast,
    seasonal_naive_forecast,
    drift_forecast,
    generate_all_benchmarks,
)


@pytest.fixture
def train_and_index():
    # long enough to cover generate_all_benchmarks' default weekly
    # seasonality (168h) with room to spare, so seasonal_naive_weekly
    # doesn't run out of history on its very first lookup
    rng_train = pd.date_range("2024-01-01", periods=24 * 10, freq="h")
    # distinct, easy-to-reason-about values: hour-of-day pattern repeated
    values = np.tile(np.arange(24, dtype=float), 10)
    train = pd.Series(values, index=rng_train, name="Appliances")

    horizon = 24
    test_index = pd.date_range(
        train.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )
    return train, horizon, test_index


@pytest.mark.parametrize(
    "forecast_fn, kwargs",
    [
        (mean_forecast, {}),
        (naive_forecast, {}),
        (drift_forecast, {}),
    ],
)
def test_forecast_length_matches_horizon(train_and_index, forecast_fn, kwargs):
    """README-promised test: every forecast's length (and index) must
    exactly match the requested test period."""
    train, horizon, test_index = train_and_index

    result = forecast_fn(train, horizon, test_index, **kwargs)

    assert len(result) == horizon
    assert (result.index == test_index).all()


def test_seasonal_naive_forecast_length_matches_horizon(train_and_index):
    train, horizon, test_index = train_and_index

    result = seasonal_naive_forecast(train, horizon, test_index, seasonality=24)

    assert len(result) == horizon
    assert (result.index == test_index).all()


def test_mean_forecast_is_constant_training_mean(train_and_index):
    train, horizon, test_index = train_and_index
    result = mean_forecast(train, horizon, test_index)

    assert (result == train.mean()).all()


def test_naive_forecast_repeats_last_observed_value(train_and_index):
    train, horizon, test_index = train_and_index
    result = naive_forecast(train, horizon, test_index)

    assert (result == train.iloc[-1]).all()


def test_seasonal_naive_first_steps_match_lagged_history(train_and_index):
    """The first `seasonality` steps of a seasonal-naive forecast should
    exactly repeat the last `seasonality` training observations (same
    hour, one cycle back), before the forecast has to start recursing
    on its own predictions."""
    train, horizon, test_index = train_and_index
    seasonality = 24

    result = seasonal_naive_forecast(train, horizon, test_index, seasonality=seasonality)

    expected_first_cycle = train.iloc[-seasonality:].values
    np.testing.assert_array_equal(result.values[:seasonality], expected_first_cycle)


def test_drift_forecast_is_monotonic_for_increasing_series():
    """With a strictly increasing training series, the drift forecast's
    slope should be positive, so the forecast itself is monotonically
    increasing."""
    rng = pd.date_range("2024-01-01", periods=48, freq="h")
    train = pd.Series(np.arange(48, dtype=float), index=rng)
    test_index = pd.date_range(
        train.index[-1] + pd.Timedelta(hours=1), periods=10, freq="h"
    )

    result = drift_forecast(train, 10, test_index)

    assert (np.diff(result.values) > 0).all()


def test_generate_all_benchmarks_returns_five_named_series(train_and_index):
    train, horizon, test_index = train_and_index

    forecasts = generate_all_benchmarks(train, horizon, test_index)

    expected_names = {
        "mean", "naive", "seasonal_naive_daily",
        "seasonal_naive_weekly", "drift",
    }
    assert set(forecasts.keys()) == expected_names
    for name, series in forecasts.items():
        assert len(series) == horizon
        assert series.name == name
