# tests/conftest.py

# ============================================================
# Makes `src/appliance_energy` importable when running `pytest`
# from the project root, without needing the package installed
# (`pip install -e .`) or a src-layout build step. Mirrors the
# sys.path.insert(...) pattern already used at the top of every
# script in scripts/ and every notebook in notebooks/.
# ============================================================

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
