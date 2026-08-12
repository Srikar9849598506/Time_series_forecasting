# tests/test_sarimax.py

# ============================================================
# Tests for src/appliance_energy/models/sarimax.py
#
# These deliberately do NOT fit any SARIMAX models (that needs
# statsmodels and is slow/expensive - covered by the notebook and
# scripts/run_sarimax.py instead). They check the one thing that is
# cheap, fast, and easy to silently get wrong: that the candidate
# order grid actually matches the assignment's explicit spec
# (p in [0,6], d in [0,2], q in [0,6], looped over every
# combination - 147 candidates total).
# ============================================================

from appliance_energy.models.sarimax import build_full_pdq_grid


def test_full_pdq_grid_has_147_combinations():
    grid = build_full_pdq_grid()
    assert len(grid) == 147


def test_full_pdq_grid_covers_every_required_combination():
    grid = build_full_pdq_grid()
    grid_set = set(grid)

    expected = {
        (p, d, q)
        for p in range(0, 7)
        for d in range(0, 3)
        for q in range(0, 7)
    }

    assert grid_set == expected


def test_full_pdq_grid_has_no_duplicates():
    grid = build_full_pdq_grid()
    assert len(grid) == len(set(grid))


def test_full_pdq_grid_respects_custom_ranges():
    """The function should still work correctly for a smaller/custom
    range (used, e.g., for quick local testing without waiting for the
    full 147-model search)."""
    grid = build_full_pdq_grid(p_range=range(0, 2), d_range=range(0, 1), q_range=range(0, 2))

    assert len(grid) == 2 * 1 * 2
    assert set(grid) == {(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)}
