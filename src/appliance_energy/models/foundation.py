"""Real Chronos foundation-model forecasting.

Chronos is used zero-shot and target-only: only the historical
Appliances series is supplied to the pretrained model.  There is
intentionally NO fallback forecast.  If Chronos is unavailable,
the pipeline fails early rather than producing a result that could
be incorrectly reported as a foundation-model forecast.
"""

import numpy as np
import pandas as pd

from appliance_energy.config import RANDOM_STATE

CHRONOS_MODEL_NAME = "amazon/chronos-t5-small"


def check_chronos_available(model_name=CHRONOS_MODEL_NAME):
    """Import Chronos and load the pretrained model to fail early.

    Returns the loaded pipeline.  This function is called by the
    preflight stage before the expensive SARIMAX grid search.
    """
    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Chronos is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    try:
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Chronos model '{model_name}' could not be loaded. "
            "Check internet access/model cache and try again."
        ) from exc

    return pipeline


def forecast_chronos(y_train, horizon, index, pipeline=None,
                     model_name=CHRONOS_MODEL_NAME):
    """Forecast exactly ``horizon`` steps using real Chronos.

    The assignment asks for the next 24 hours.  The model therefore
    receives the full historical target context and makes one direct
    probabilistic forecast for the requested horizon; no test targets
    or future covariates are supplied.
    """
    if pipeline is None:
        pipeline = check_chronos_available(model_name)

    import torch

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    context = torch.tensor(y_train.to_numpy(dtype=float), dtype=torch.float32)
    try:
        forecast = pipeline.predict(context, prediction_length=horizon)
    except TypeError:
        forecast = pipeline.predict(context, horizon)

    samples = forecast[0].detach().cpu().numpy()
    median = np.median(samples, axis=0)

    return pd.Series(median[:horizon], index=index, name="foundation_model")
