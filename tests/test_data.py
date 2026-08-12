# tests/test_data.py

# ============================================================
# Tests for src/appliance_energy/data.py
#
# Uses small synthetic data rather than the real downloaded
# dataset, so these tests run fast and fully offline from a
# fresh clone (no dependency on data/raw/energydata_complete.csv
# being present, per the README's "reproducible from a fresh
# clone" requirement).
# ============================================================

import numpy as np
import pandas as pd

from appliance_energy.data import train_test_split_series


def test_train_test_split_is_chronological_and_non_overlapping():
    rng = pd.date_range("2024-01-01", periods=100, freq="h")
    y = pd.Series(np.arange(100, dtype=float), index=rng)

    test_steps = 24
    train, test = train_test_split_series(y, test_steps)

    # sizes
    assert len(test) == test_steps
    assert len(train) == len(y) - test_steps

    # no overlap, and every training timestamp precedes every test
    # timestamp (this is the leakage safeguard for a time series split)
    assert train.index.max() < test.index.min()

    # nothing lost or duplicated across the split
    assert len(train) + len(test) == len(y)
    assert not train.index.isin(test.index).any()


def test_train_test_split_preserves_values():
    """The split must not reorder or alter any values - it should be a
    pure positional slice."""
    rng = pd.date_range("2024-01-01", periods=50, freq="h")
    y = pd.Series(np.arange(50, dtype=float), index=rng)

    train, test = train_test_split_series(y, 10)

    pd.testing.assert_series_equal(train, y.iloc[:-10])
    pd.testing.assert_series_equal(test, y.iloc[-10:])


def test_resampled_hourly_series_has_no_missing_target_values(tmp_path):
    """README-promised test: 'the processed dataset has no missing
    target values'. Builds a small synthetic 10-minute frame (with one
    deliberate short gap, mimicking real sensor dropout) and checks
    that resample_to_hourly's interpolate+dropna step leaves no NaNs."""
    from appliance_energy.data import resample_to_hourly

    rng = pd.date_range("2024-01-01", periods=6 * 24, freq="10min")  # 1 day
    values = np.linspace(50, 150, len(rng))
    df = pd.DataFrame({"Appliances": values}, index=rng)

    # simulate a short sensor dropout (a few missing 10-minute readings)
    df.iloc[10:13] = np.nan

    hourly = resample_to_hourly(df, save_path=tmp_path / "hourly.csv")

    assert hourly["Appliances"].isna().sum() == 0
    assert len(hourly) > 0
