"""Preflight checks for the complete assignment pipeline.

This stage deliberately runs BEFORE the expensive 147-model SARIMAX
search. It verifies the Python dependencies and loads the real Chronos
model so a missing package/model is discovered immediately.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    required = [
        "numpy", "pandas", "matplotlib", "sklearn", "statsmodels",
        "xgboost", "torch", "chronos",
    ]
    missing = []
    for name in required:
        try:
            __import__(name)
        except Exception as exc:
            missing.append(f"{name}: {exc}")

    if missing:
        print("\nPREFLIGHT FAILED - install requirements before starting the long run:\n")
        for item in missing:
            print("  -", item)
        print("\nRun: python -m pip install -r requirements.txt")
        raise SystemExit(1)

    from appliance_energy.models.foundation import check_chronos_available, CHRONOS_MODEL_NAME
    print("All required Python packages are installed.")
    print(f"Checking real foundation model: {CHRONOS_MODEL_NAME}")
    check_chronos_available()
    print("Chronos model loaded successfully. Foundation stage is ready.")
    print("\nPREFLIGHT PASSED - the pipeline can now start the SARIMAX grid search.\n")


if __name__ == "__main__":
    main()
