# src/appliance_energy/models/sarimax.py

# ============================================================
# SARIMAX modelling for the appliance energy series.
#
# Order justified by the Part 1 stationarity analysis:
#   - ADF rejects a stochastic trend on the raw series -> d = 0
#   - ACF shows slowly-decaying periodic structure at lags
#     24, 48, 72 -> seasonal differencing needed -> D = 1, s = 24
#   - ACF/PACF of the seasonally-differenced series cut off
#     quickly at low lags -> small non-seasonal (p, q)
#
# Starting point (per assignment spec):
#   order          = (1, 0, 1)
#   seasonal_order = (1, 1, 1, 24)
# ============================================================

import matplotlib.pyplot as plt
from ast import literal_eval
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

from appliance_energy.config import FIGURE_DIR


def fit_sarimax(y_train, exog_train=None, order=(1, 0, 1),
                 seasonal_order=(1, 1, 1, 24), trend="c", maxiter=200):
    """
    Fit a SARIMAX model.

    exog_train=None fits a target-only SARIMA. Passing a DataFrame
    of exogenous regressors fits SARIMAX with those covariates.

    enforce_stationarity/invertibility are relaxed because with a
    seasonal period of 24 the parameter space is large and strict
    enforcement can prevent convergence; this is standard practice
    for exploratory SARIMAX fitting.
    """

    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    # maxiter raised above the statsmodels default (50) because the
    # exogenous fit with a 24-period season occasionally needs more
    # iterations to converge cleanly. Grid search calls this with a
    # lower maxiter (see select_order_by_aic) purely for speed across
    # ~150 candidate fits; the final chosen order is always refit with
    # the full maxiter=200 for the actual forecast.
    fit = model.fit(disp=False, maxiter=maxiter)

    return fit


def forecast_sarimax(fit, horizon, index, exog_test=None, name="sarimax", alpha=0.05):
    """
    Produce a point forecast Series over the test index, along with its
    (1 - alpha) confidence interval (assignment Part 4: "add confidence
    intervals on forecasts"). alpha=0.05 -> a 95% interval.

    Returns
    -------
    mean : pd.Series
        Point forecast (predicted mean), named `name`.
    conf_int : pd.DataFrame
        Two columns, f"{name}_lower" and f"{name}_upper".
    """

    # Statsmodels requires exactly `horizon` rows of future exogenous
    # values. The project keeps a 14-day holdout (336 hours) for the
    # overall evaluation split, while the assignment forecast horizon
    # is 24 hours. Passing all 336 rows causes the classic
    # "Required (24, 5), got (336, 5)" error.
    if exog_test is not None:
        exog_test = exog_test.copy()
        if isinstance(exog_test, pd.DataFrame):
            # Prefer exact timestamp alignment when the forecast index
            # is available. This avoids accidentally using the wrong
            # 24 rows if the holdout is longer than the forecast horizon.
            if hasattr(index, "__len__") and len(index) == horizon:
                aligned = exog_test.reindex(index)
                if not aligned.isna().any().any():
                    exog_test = aligned
                else:
                    exog_test = exog_test.iloc[:horizon].copy()
            else:
                exog_test = exog_test.iloc[:horizon].copy()

        if len(exog_test) != horizon:
            raise ValueError(
                f"SARIMAX future exogenous data must have exactly {horizon} rows; "
                f"got {len(exog_test)} rows."
            )

    fc = fit.get_forecast(steps=horizon, exog=exog_test)

    mean = fc.predicted_mean
    mean.index = index
    mean.name = name

    conf_int = fc.conf_int(alpha=alpha)
    conf_int.index = index
    conf_int.columns = [f"{name}_lower", f"{name}_upper"]

    return mean, conf_int


def select_order_by_aic(y_train, candidate_orders, seasonal_order=(1, 1, 1, 24),
                         maxiter=50, verbose=True, checkpoint_path=None):
    """
    Fit a grid of candidate (p, d, q) orders and report AIC/BIC for each,
    used to select the non-seasonal order by minimum AIC.

    Per the assignment spec, the full grid searched is every combination
    of p in [0,6], d in [0,2], q in [0,6] (see
    `build_full_pdq_grid` / scripts/run_sarimax.py) - 147 candidates in
    total. That is expensive on an hourly series with a 24-period season,
    so this search phase uses a lower maxiter (50, the statsmodels
    default) purely for speed; the winning order is refit with the full
    maxiter=200 by the caller before it is used for forecasting.

    Some higher-order combinations (in particular d=2, or large p/q
    combined with a seasonal order) may fail to converge or raise a
    LinAlgError from an over-differenced/near-singular series - these are
    caught and skipped rather than allowed to crash the whole search.
    """

    rows = []
    n_failed = 0
    completed = set()

    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            existing = pd.read_csv(checkpoint_path)
            for _, row in existing.iterrows():
                order = tuple(literal_eval(row["order"])) if isinstance(row["order"], str) else tuple(row["order"])
                rows.append({
                    "order": order,
                    "seasonal_order": tuple(literal_eval(row["seasonal_order"])) if isinstance(row["seasonal_order"], str) else tuple(row["seasonal_order"]),
                    "AIC": float(row["AIC"]),
                    "BIC": float(row["BIC"]),
                    "converged": str(row["converged"]).strip().lower() == "true",
                })
                completed.add(order)
            print(f"Resuming SARIMAX search: {len(completed)} completed candidates loaded from {checkpoint_path}")
        except Exception as exc:
            print(f"Could not read checkpoint {checkpoint_path}: {exc}. Starting fresh.")
            rows = []
            completed = set()

    def save_checkpoint():
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).sort_values("AIC").to_csv(checkpoint_path, index=False)

    for i, order in enumerate(candidate_orders):
        if order in completed:
            if verbose:
                print(f"[{i + 1}/{len(candidate_orders)}] order={order} already checkpointed - skipped")
            continue
        try:
            fit = fit_sarimax(
                y_train, order=order, seasonal_order=seasonal_order,
                maxiter=maxiter,
            )
            row = {
                "order": order,
                "seasonal_order": seasonal_order,
                "AIC": fit.aic,
                "BIC": fit.bic,
                "converged": bool(fit.mle_retvals.get("converged", True))
                if hasattr(fit, "mle_retvals") else True,
            }
            rows.append(row)
            completed.add(order)
            save_checkpoint()
            if verbose:
                print(f"[{i + 1}/{len(candidate_orders)}] order={order} AIC={fit.aic:.1f} BIC={fit.bic:.1f}")
        except Exception as exc:
            n_failed += 1
            if verbose:
                print(f"[{i + 1}/{len(candidate_orders)}] order={order} failed to converge: {type(exc).__name__}: {exc}")

    print(f"\nOrder search complete: {len(rows)} converged, "
          f"{n_failed} failed/skipped, out of {len(candidate_orders)} candidates.")

    if not rows:
        raise RuntimeError("No candidate (p, d, q) order converged.")

    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def build_full_pdq_grid(p_range=range(0, 7), d_range=range(0, 3), q_range=range(0, 7)):
    """
    Build the full (p, d, q) candidate grid required by the assignment
    spec: p in [0,6], d in [0,2], q in [0,6] -> 7 * 3 * 7 = 147 orders.
    """
    return [
        (p, d, q)
        for d in d_range
        for p in p_range
        for q in q_range
    ]


def residual_diagnostics(fit, name="sarimax", lags=48, fname_prefix="sarimax"):
    """
    Standard SARIMAX residual diagnostics:
      - statsmodels' built-in 4-panel diagnostic plot
        (residuals, histogram + KDE vs normal, Q-Q plot, ACF)
      - Ljung-Box test for residual autocorrelation
        (H0: residuals are independently distributed - a good
        model should FAIL to reject this, i.e. p > 0.05)
    """

    fig = fit.plot_diagnostics(figsize=(14, 10))
    fig.suptitle(f"Residual diagnostics - {name}", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{fname_prefix}_residual_diagnostics.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    lb = acorr_ljungbox(fit.resid.dropna(), lags=[lags], return_df=True)
    lb_stat = lb["lb_stat"].iloc[0]
    lb_pvalue = lb["lb_pvalue"].iloc[0]

    conclusion = (
        "no significant residual autocorrelation (good fit)"
        if lb_pvalue > 0.05
        else "significant residual autocorrelation remains (model may be under-specified)"
    )

    print(f"\nLjung-Box test - {name} (lag={lags})")
    print(f"  statistic = {lb_stat:.4f}, p-value = {lb_pvalue:.4g}")
    print(f"  => {conclusion}")

    # Residual ACF on its own, easier to read than the 4-panel plot
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(fit.resid.dropna(), lags=lags, ax=ax)
    ax.set_title(f"Residual ACF - {name}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{fname_prefix}_residual_acf.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "model": name,
        "ljung_box_stat": lb_stat,
        "ljung_box_pvalue": lb_pvalue,
        "conclusion": conclusion,
        "AIC": fit.aic,
        "BIC": fit.bic,
    }