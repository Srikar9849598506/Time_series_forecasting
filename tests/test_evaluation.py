# tests/test_evaluation.py

# ============================================================
# Tests for src/appliance_energy/evaluation.py
#
# Covers the two examples explicitly promised in the README's
# "Tests" section:
#   - MASE is zero for a perfect forecast
#   - forecast lengths / alignment behave correctly through
#     evaluate_all
# plus basic sanity checks on MAE, RMSE, and Bias sign.
# ============================================================

import numpy as np
import pandas as pd
import pytest

from appliance_energy.evaluation import (
    rmse,
    mase,
    bias,
    evaluate_forecast,
    evaluate_all,
)


@pytest.fixture
def sample_series():
    """A small synthetic hourly series with an obvious daily (24h) cycle
    plus a tiny linear drift, long enough to compute a lag-24
    seasonal-naive scale. The drift matters: a perfectly periodic series
    would make the in-sample seasonal-naive scale exactly zero (every
    lag-24 difference would be 0), which is a genuine edge case handled
    separately in test_mase_returns_nan_for_constant_training_series -
    here we want the normal, non-degenerate case."""
    rng = pd.date_range("2024-01-01", periods=24 * 10, freq="h")
    hours = rng.hour.values
    day_index = np.arange(len(rng)) // 24
    values = 100 + 50 * np.sin(2 * np.pi * hours / 24) + 0.5 * day_index
    y = pd.Series(values, index=rng, name="Appliances")

    train = y.iloc[:-24]
    test = y.iloc[-24:]
    return train, test


def test_mase_zero_for_perfect_forecast(sample_series):
    """A forecast that exactly matches the actuals must have MASE 0."""
    train, test = sample_series
    perfect_pred = test.copy()

    result = mase(test, perfect_pred, train, seasonality=24)

    assert result == pytest.approx(0.0, abs=1e-9)


def test_mase_below_one_when_better_than_seasonal_naive(sample_series):
    """A near-perfect forecast should score well below 1.0 (i.e. clearly
    beat a lag-24 seasonal-naive forecast on its own training-set scale)."""
    train, test = sample_series
    # forecast is the truth plus a very small constant offset
    near_perfect_pred = test + 0.01

    result = mase(test, near_perfect_pred, train, seasonality=24)

    assert result < 1.0


def test_mase_returns_nan_for_constant_training_series():
    """If the training series is perfectly constant, the seasonal-naive
    scale is zero, and MASE is undefined (should return NaN, not raise
    or divide by zero silently)."""
    rng = pd.date_range("2024-01-01", periods=48, freq="h")
    constant_train = pd.Series(42.0, index=rng)
    test_index = pd.date_range("2024-01-03", periods=24, freq="h")
    test = pd.Series(42.0, index=test_index)
    pred = pd.Series(50.0, index=test_index)

    result = mase(test, pred, constant_train, seasonality=24)

    assert np.isnan(result)


def test_bias_sign_over_and_under_forecast():
    """Bias should be positive when the model over-forecasts on average,
    and negative when it under-forecasts on average."""
    y_true = pd.Series([100.0, 100.0, 100.0])

    over_forecast = pd.Series([110.0, 110.0, 110.0])
    under_forecast = pd.Series([90.0, 90.0, 90.0])

    assert bias(y_true, over_forecast) == pytest.approx(10.0)
    assert bias(y_true, under_forecast) == pytest.approx(-10.0)


def test_rmse_matches_hand_calculation():
    y_true = pd.Series([0.0, 0.0, 0.0, 0.0])
    y_pred = pd.Series([3.0, 4.0, 0.0, 0.0])
    # errors: 3, 4, 0, 0 -> squared: 9, 16, 0, 0 -> mean 6.25 -> sqrt 2.5
    assert rmse(y_true, y_pred) == pytest.approx(2.5)


def test_evaluate_forecast_returns_all_required_metrics(sample_series):
    """Assignment spec requires MAE, RMSE, MASE, and Bias for every model."""
    train, test = sample_series
    pred = test + 1.0

    result = evaluate_forecast("dummy_model", test, pred, train, seasonality=24)

    assert set(result.keys()) == {"model", "MAE", "RMSE", "MASE", "Bias"}
    assert result["model"] == "dummy_model"
    assert result["MAE"] == pytest.approx(1.0)
    assert result["Bias"] == pytest.approx(1.0)


def test_evaluate_all_sorts_by_mase_and_handles_misaligned_forecasts(sample_series):
    """evaluate_all should rank models by MASE (best first) and should not
    error out when a forecast Series starts later than the test window
    (e.g. a feature-based model that lost rows to lag/rolling windows)."""
    train, test = sample_series

    good_forecast = test.copy()  # perfect -> MASE 0
    bad_forecast = pd.Series(0.0, index=test.index)  # far off -> high MASE

    # a forecast missing the first two timestamps of the test window,
    # simulating a model whose feature table dropped leading rows
    short_forecast = test.copy()
    short_forecast = short_forecast.iloc[2:]

    forecasts = {
        "bad": bad_forecast,
        "good": good_forecast,
        "partial": short_forecast,
    }

    results = evaluate_all(forecasts, test, train, seasonality=24)

    assert list(results.columns) == ["model", "MAE", "RMSE", "MASE", "Bias"]
    # best (lowest MASE) model should be first after sorting
    assert results.iloc[0]["model"] == "good"
    assert results.iloc[0]["MASE"] == pytest.approx(0.0, abs=1e-9)
    # the partial forecast should still evaluate successfully (no crash),
    # using only its overlapping timestamps
    assert "partial" in results["model"].values
