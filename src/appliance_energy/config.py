# src/appliance_energy/config.py

# ============================================================
# Central configuration for the appliance energy
# forecasting project.
#
# All paths and modelling constants are defined here so that
# every script and notebook uses the same settings.
# ============================================================

from pathlib import Path

import numpy as np

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

# config.py lives at src/appliance_energy/config.py,
# so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "model_objects"

for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
             FIGURE_DIR, FORECAST_DIR, METRICS_DIR, MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00374/energydata_complete.csv"
)

RAW_FILE = RAW_DIR / "energydata_complete.csv"
HOURLY_FILE = PROCESSED_DIR / "appliance_hourly.csv"

TARGET = "Appliances"

# ------------------------------------------------------------
# Time series constants (hourly resolution)
# ------------------------------------------------------------

DAILY_PERIOD = 24        # observations per day
WEEKLY_PERIOD = 168      # observations per week

# Final 14 days held out as the test set
TEST_STEPS = 14 * DAILY_PERIOD

# Assignment forecast horizon: next 24 hours.
FORECAST_HORIZON = 24

# Candidate exogenous weather variables for SARIMAX
EXOG_COLS = [
    "T_out",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
]
