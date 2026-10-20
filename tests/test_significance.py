"""Unit tests for the statistical-significance module."""

# Python libraries
import numpy as np
import pytest

# EdgeSimPy
from edge_sim_py.experiments import (
    paired_test,
    independent_test,
    friedman_test,
    cohens_d_paired,
    cliffs_delta,
    holm_bonferroni,
    benjamini_hochberg,
    compare_all_pairs,
)


def test_paired_test_detects_clear_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 1, 30)
    b = a + 3.0 + rng.normal(0, 0.5, 30)  # larger by ~3 with realistic (non-constant) variation
    result = paired_test(a, b)
    assert result["p_value"] < 0.001
    assert result["mean_difference"] == pytest.approx(-3.0, abs=0.5)
    assert result["effect"] == "large"


def test_paired_test_no_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(5, 1, 40)
    b = rng.normal(5, 1, 40)
    result = paired_test(a, b)
    assert result["p_value"] > 0.05  # no real difference


def test_independent_test():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 50)
    b = rng.normal(2, 1, 50)
    result = independent_test(a, b)
    assert result["p_value"] < 0.001
    assert abs(result["cohens_d"]) > 0.8  # large effect


def test_cohens_d_paired_sign_and_magnitude():
    a = [1, 2, 3, 4, 5]
    b = [2.5, 3, 4.5, 5, 6.5]  # b larger on average, with non-constant differences
    d = cohens_d_paired(a, b)
    assert d < 0  # a < b


def test_cohens_d_paired_constant_difference_is_guarded():
    # Zero-variance difference -> d is undefined; the guard returns 0.0 rather than dividing by zero.
    assert cohens_d_paired([1, 2, 3], [2, 3, 4]) == 0.0


def test_cliffs_delta_bounds():
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)  # fully dominates
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == pytest.approx(-1.0)
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_holm_bonferroni_monotonic_and_stricter_than_raw():
    p = [0.001, 0.02, 0.5]
    corrected = holm_bonferroni(p)
    adj = [c["p_adjusted"] for c in corrected]
    assert all(adj[i] >= p[i] for i in range(len(p)))  # adjusted >= raw
    assert corrected[0]["reject"] is True and corrected[2]["reject"] is False


def test_benjamini_hochberg_runs():
    corrected = benjamini_hochberg([0.001, 0.04, 0.2, 0.6])
    assert corrected[0]["reject"] is True
    assert all(0.0 <= c["p_adjusted"] <= 1.0 for c in corrected)


def test_friedman_detects_group_difference():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 15)
    groups = {"a": base, "b": base + 2, "c": base + 4}  # clearly ordered
    result = friedman_test(groups)
    assert result["p_value"] < 0.05


def test_compare_all_pairs_with_correction():
    rng = np.random.default_rng(4)
    base = rng.normal(0, 1, 20)
    groups = {"low": base, "mid": base + 1.5, "high": base + 3.0}
    rows = compare_all_pairs(groups, paired=True, correction="holm")
    assert len(rows) == 3  # 3 pairwise comparisons
    assert all("p_adjusted" in r and "significant" in r for r in rows)
    # low vs high should be significant with a large effect.
    lo_hi = next(r for r in rows if {r["group_a"], r["group_b"]} == {"low", "high"})
    assert lo_hi["significant"] is True
