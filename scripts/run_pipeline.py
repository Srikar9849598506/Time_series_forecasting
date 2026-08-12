# scripts/run_pipeline.py

# ============================================================
# Main pipeline entry point (assignment Part 11: "a runnable
# pipeline script"; README "Running the pipeline" section).
#
# Runs every stage of the analysis, in the order each stage
# depends on the last, from a single command:
#
#   python scripts/run_pipeline.py
#
# Equivalent to running, in order:
#   1. run_eda_stationarity.py   (data prep, EDA, stationarity)
#   2. run_benchmarks.py         (Part 2: benchmark forecasts)
#   3. run_sarimax.py            (Part 3: SARIMA/SARIMAX)
#   4. run_feature_model.py      (Part 4: feature-based model)
#   5. run_foundation_model.py   (Part 5: Chronos foundation model)
#   6. run_final_comparison.py   (Part 6: combined comparison)
#
# NOTE on run time: run_sarimax.py includes the assignment-required
# 147-combination AIC grid search the first time it runs. Results are
# checkpointed to outputs/metrics/order_selection.csv after each
# successful candidate, so an interrupted run can resume without
# discarding completed fits.
#
# The preflight stage runs first and loads the real Chronos model,
# preventing a long SARIMAX run from ending later because a required
# foundation-model dependency is missing.
#
# Each stage is run as a separate subprocess (not just an imported
# function call) so that a failure in one stage prints a normal
# Python traceback and stops the pipeline with a non-zero exit code,
# rather than silently continuing with partial/stale outputs.
# ============================================================

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

STAGES = [
        ("Preflight: dependency + Chronos check", "preflight.py"),
    ("Part 1: Data prep, EDA, stationarity", "run_eda_stationarity.py"),
    ("Part 2: Benchmark models", "run_benchmarks.py"),
    ("Part 3: SARIMA / SARIMAX", "run_sarimax.py"),
    ("Part 4: Feature-based model (XGBoost)", "run_feature_model.py"),
    ("Part 5: Foundation model (Chronos)", "run_foundation_model.py"),
    ("Part 6: Final combined comparison", "run_final_comparison.py"),
]


def main():
    start = time.time()

    for i, (description, script_name) in enumerate(STAGES, start=1):
        script_path = SCRIPTS_DIR / script_name
        print("\n" + "=" * 70)
        print(f"STAGE {i}/{len(STAGES)}: {description}")
        print(f"  running {script_path.relative_to(SCRIPTS_DIR.parent)}")
        print("=" * 70)

        stage_start = time.time()
        result = subprocess.run([sys.executable, str(script_path)])
        stage_elapsed = time.time() - stage_start

        if result.returncode != 0:
            print(
                f"\nPipeline stopped: stage {i} ({script_name}) exited "
                f"with code {result.returncode}."
            )
            sys.exit(result.returncode)

        print(f"\nStage {i} finished in {stage_elapsed / 60:.1f} min.")

    total_elapsed = time.time() - start
    print("\n" + "=" * 70)
    print(f"Pipeline complete in {total_elapsed / 60:.1f} min.")
    print("Final outputs:")
    print("  outputs/forecasts/all_forecasts.csv")
    print("  outputs/metrics/model_comparison.csv")
    print("  outputs/figures/forecast_comparison.png")
    print("  outputs/figures/forecast_comparison_split.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
