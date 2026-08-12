# src/appliance_energy/features.py

# ============================================================
# Feature engineering for the feature-based (XGBoost) model.
#
# Three feature families, per the assignment brief:
#   1. Original measured variables (indoor/outdoor sensors)
#   2. Time-based features (hour, day-of-week, cyclical encodings)
#   3. Lag and rolling features of the target
#
# Leakage safeguard: every lag/rolling feature is built from
# target values shifted by at least 1 step, so no feature can
# see the current or a future value of the target it is meant
# to help predict.
# ============================================================

import numpy as np

from appliance_energy.config import TARGET

LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [3, 6, 12, 24, 168]


def add_time_features(df):
    """
    Add calendar-derived features. These are known in advance for
    any future timestamp, so using them at the forecast origin is
    always legitimate (unlike future sensor/weather values).
    """

    out = df.copy()

    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    # Cyclical encoding so the model sees hour 23 and hour 0 as
    # close together, rather than as maximally different integers.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    return out


def add_lag_and_rolling_features(df, target=TARGET, lags=LAGS,
                                  windows=ROLLING_WINDOWS):
    """
    Add lagged and rolling-window features of the target.

    Every rolling feature is computed on target.shift(1) first -
    this is the critical leakage safeguard: without the shift,
    a rolling window ending at time t would include y[t] itself,
    letting the model "predict" using the answer.
    """

    out = df.copy()

    for lag in lags:
        out[f"lag_{lag}"] = out[target].shift(lag)

    shifted_target = out[target].shift(1)

    for window in windows:
        out[f"roll_mean_{window}"] = shifted_target.rolling(window).mean()
        out[f"roll_std_{window}"] = shifted_target.rolling(window).std()

    return out


def make_feature_table(df, target=TARGET, lags=LAGS, windows=ROLLING_WINDOWS):
    """
    Build the full supervised-learning feature table:
    original sensor/weather variables + time features +
    lag/rolling features of the target, with incomplete
    leading rows (from lags/rolling windows) dropped.
    """

    out = add_time_features(df)
    out = add_lag_and_rolling_features(out, target=target, lags=lags, windows=windows)
    out = out.dropna()

    return out


def get_feature_columns(feature_table, target=TARGET):
    """
    All columns except the target are candidate model features.
    """
    return [c for c in feature_table.columns if c != target]
