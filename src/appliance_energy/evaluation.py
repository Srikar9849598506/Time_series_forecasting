# src/appliance_energy/evaluation.py

# ============================================================
# Forecast evaluation metrics.
#
# Required metrics (per assignment spec):
#   MAE   - Mean Absolute Error
#   RMSE  - Root Mean Squared Error
#   MASE  - Mean Absolute Scaled Error
#   Bias  - Mean signed error (systematic over/under-forecasting)
#
# MASE is the headline metric: it scales the forecast's MAE by
# the in-sample MAE of a seasonal naive forecast, so a value
# below 1.0 means "better than seasonally-naive guessing" in a
# way that is comparable across series and units.
# ============================================================

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from appliance_energy.config import DAILY_PERIOD


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mase(y_true, y_pred, y_train, seasonality=DAILY_PERIOD):
    """
    Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Scale = mean absolute error of the in-sample seasonal naive
    forecast (lag = seasonality). A MASE < 1 means the model
    beats seasonal-naive guessing on the training data's own scale.
    """

    y_train = pd.Series(y_train).astype(float)

    seasonal_errors = np.abs(
        y_train.iloc[seasonality:].values
        - y_train.iloc[:-seasonality].values
    )

    scale = seasonal_errors.mean()

    if scale == 0 or np.isnan(scale):
        return np.nan

    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale


def bias(y_true, y_pred):
    """
    Mean signed error. Positive = model over-forecasts on average,
    negative = model under-forecasts on average.
    """
    return np.mean(np.asarray(y_pred) - np.asarray(y_true))


def evaluate_forecast(name, y_true, y_pred, y_train, seasonality=DAILY_PERIOD):
    """
    Compute all required metrics for one forecast and return
    them as a dict, ready to be collected into a results table.
    """

    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred, index=y_true.index).astype(float)

    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, seasonality=seasonality),
        "Bias": bias(y_true, y_pred),
    }


def evaluate_all(forecasts, test, train, seasonality=DAILY_PERIOD):
    """
    Evaluate a dict of {model_name: forecast_series} against a
    common test set, returning a results DataFrame sorted by MASE
    (the primary ranking metric, since it is scale-independent).
    """

    results = []

    for name, pred in forecasts.items():
        pred = pred.reindex(test.index)

        # Guard against any forecasts that start slightly later
        # (e.g. feature-based models needing lag history).
        valid = pred.notna() & test.notna()

        results.append(
            evaluate_forecast(
                name=name,
                y_true=test.loc[valid],
                y_pred=pred.loc[valid],
                y_train=train,
                seasonality=seasonality,
            )
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values("MASE")
        .reset_index(drop=True)
    )

    return results_df


def get_strongest_benchmark(benchmark_results, exclude=None):
    """
    Return the (model_name, row) of the best-performing benchmark
    by MASE. Advanced models should be compared against this,
    not against each other - which specific benchmark wins can
    vary by dataset/test window, so this is computed dynamically
    rather than assumed.
    """

    df = benchmark_results.copy()

    if exclude is not None:
        df = df[~df["model"].isin(exclude)]

    best_row = df.sort_values("MASE").iloc[0]

    return best_row["model"], best_row
