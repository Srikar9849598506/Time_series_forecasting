# src/appliance_energy/data.py

# ============================================================
# Data loading and preparation.
#
# Steps:
#   1. Download (or load a cached copy of) the raw
#      10-minute UCI Appliances Energy Prediction data.
#   2. Parse the timestamp and coerce columns to numeric.
#   3. Resample to hourly means so that SARIMAX with a
#      24-period seasonal component is tractable.
# ============================================================

import pandas as pd

from appliance_energy.config import (
    DATA_URL,
    RAW_FILE,
    HOURLY_FILE,
    TARGET,
)


def download_raw_data(url=DATA_URL, cache_path=RAW_FILE, force=False):
    """
    Download the raw CSV from UCI and cache it locally.

    A cached copy makes the pipeline reproducible offline and
    avoids re-downloading ~12 MB on every run.
    """

    if cache_path.exists() and not force:
        print(f"Using cached raw data: {cache_path}")
        return cache_path

    print(f"Downloading raw data from {url}")
    df = pd.read_csv(url)
    df.to_csv(cache_path, index=False)
    print(f"Saved raw data to {cache_path}")

    return cache_path


def load_raw_data(path=RAW_FILE):
    """
    Load the raw 10-minute data with a DatetimeIndex.
    """

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # The raw CSV stores numbers as quoted strings with padding,
    # so coerce every column to numeric.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])

    return df


def report_missing_and_gaps(df, freq="10min"):
    """
    Report missing values and gaps in the sampling grid.

    Returns the missing timestamps so they can be inspected.
    """

    print("\nMissing values per column (non-zero only):")
    missing = df.isna().sum()
    print(missing[missing > 0] if missing.any() else "  none")

    expected = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    missing_stamps = expected.difference(df.index)

    print(f"\nExpected {freq} timestamps: {len(expected)}")
    print(f"Actual rows:               {len(df)}")
    print(f"Missing timestamps:        {len(missing_stamps)}")

    return missing_stamps


def resample_to_hourly(df, save_path=HOURLY_FILE):
    """
    Aggregate the 10-minute data to hourly means.

    Hourly resolution keeps the daily usage cycle while
    reducing the series length by a factor of six, which makes
    SARIMAX estimation with seasonal period 24 practical.
    Small gaps are filled by time interpolation.
    """

    hourly = df.resample("h").mean()
    hourly = hourly.interpolate("time")
    hourly = hourly.dropna()

    hourly.to_csv(save_path)

    print(f"\nHourly data shape: {hourly.shape}")
    print(f"Period: {hourly.index.min()} to {hourly.index.max()}")

    return hourly


def load_hourly_data(path=HOURLY_FILE):
    """
    Load the processed hourly dataset (building it if absent).
    """

    if not path.exists():
        download_raw_data()
        raw = load_raw_data()
        return resample_to_hourly(raw)

    hourly = pd.read_csv(path, index_col="date", parse_dates=True)
    hourly = hourly.asfreq("h")  # sets explicit frequency, avoids
                                  # statsmodels frequency-inference warnings

    return hourly


def train_test_split_series(y, test_steps):
    """
    Chronological train/test split - the last `test_steps`
    observations become the test set. Never shuffle time series
    data, since that would leak future information into training.
    """

    train = y.iloc[:-test_steps]
    test = y.iloc[-test_steps:]

    return train, test
