# tests/test_features.py

# ============================================================
# Tests for src/appliance_energy/features.py
#
# The single most important property of this module is that lag
# and rolling features never use the current or a future value of
# the target - these tests check that directly rather than just
# trusting the docstring.
# ============================================================

import numpy as np
import pandas as pd
import pytest

from appliance_energy.features import (
    add_time_features,
    add_lag_and_rolling_features,
    make_feature_table,
    get_feature_columns,
)


@pytest.fixture
def toy_df():
    """A small target series with a distinctive, strictly increasing
    pattern, so leakage (a feature that 'sees' y[t] or later) is easy
    to detect: any leaking feature would equal or exceed y[t]."""
    rng = pd.date_range("2024-01-01", periods=200, freq="h")
    y = pd.Series(np.arange(200, dtype=float), index=rng, name="Appliances")
    return pd.DataFrame({"Appliances": y})


def test_time_features_known_in_advance(toy_df):
    """Calendar features must be exact, deterministic functions of the
    timestamp - they should never depend on the target column at all."""
    out = add_time_features(toy_df)

    assert (out["hour"] == out.index.hour).all()
    assert (out["dayofweek"] == out.index.dayofweek).all()
    assert set(out["is_weekend"].unique()) <= {0, 1}
    # cyclical encodings stay within [-1, 1]
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert out[col].between(-1.0, 1.0).all()


def test_lag_features_do_not_use_future_target_values(toy_df):
    """lag_k at time t must equal y[t-k], never y[t] or any value at or
    after t."""
    out = add_lag_and_rolling_features(toy_df, target="Appliances", lags=[1, 24])

    y = toy_df["Appliances"]
    shifted_1 = y.shift(1)
    shifted_24 = y.shift(24)

    pd.testing.assert_series_equal(out["lag_1"], shifted_1, check_names=False)
    pd.testing.assert_series_equal(out["lag_24"], shifted_24, check_names=False)

    # explicit leakage check: lag_1 at row i must never equal or exceed
    # y at row i (it should always be one step behind)
    valid = out["lag_1"].notna()
    assert (out.loc[valid, "lag_1"].values < y.loc[valid].values).all()


def test_rolling_features_are_shifted_before_windowing(toy_df):
    """roll_mean_k / roll_std_k at time t must be computed only from
    y[t-k .. t-1] - i.e. the target is shifted by 1 *before* the
    rolling window is applied, so the window never includes y[t]."""
    out = add_lag_and_rolling_features(toy_df, target="Appliances", windows=[3])

    y = toy_df["Appliances"]
    expected_mean = y.shift(1).rolling(3).mean()

    pd.testing.assert_series_equal(
        out["roll_mean_3"], expected_mean, check_names=False
    )

    # since y is strictly increasing, a non-leaking 3-step trailing mean
    # ending at t-1 must always be strictly less than y[t]
    valid = out["roll_mean_3"].notna()
    assert (out.loc[valid, "roll_mean_3"].values < y.loc[valid].values).all()


def test_make_feature_table_drops_incomplete_leading_rows(toy_df):
    """After adding an 8-step-max lag/rolling feature set, the leading
    rows that can't have a full window must be dropped (no NaNs left in
    the feature columns used for modelling)."""
    table = make_feature_table(toy_df, target="Appliances")
    feature_cols = get_feature_columns(table, target="Appliances")

    assert len(table) < len(toy_df)
    assert not table[feature_cols].isna().any().any()


def test_get_feature_columns_excludes_target(toy_df):
    table = make_feature_table(toy_df, target="Appliances")
    feature_cols = get_feature_columns(table, target="Appliances")

    assert "Appliances" not in feature_cols
    assert len(feature_cols) > 0
