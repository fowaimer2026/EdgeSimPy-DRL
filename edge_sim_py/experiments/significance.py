"""Hypothesis testing, effect sizes and multiple-comparison correction for experiments.

Complements the confidence intervals in ``statistics.py`` with the significance machinery a
reviewer expects. Because every study runs all configurations across the *same* seed set, the
default comparison is **paired** (paired t-test / Wilcoxon signed-rank); independent-group tests
(Welch / Mann-Whitney) and omnibus tests (Friedman) are provided for the cases that need them.
Every comparison is reported with an **effect size** (Cohen's d, Cliff's delta) so significance is
never claimed without magnitude, and p-values from families of comparisons can be corrected
(Holm-Bonferroni, Benjamini-Hochberg).

SciPy is used when available (already a dependency of the CI code); NumPy-only fallbacks keep the
core usable without it.
"""

# Python libraries
import itertools

import numpy as np

try:
    from scipy import stats as _sp
except ImportError:  # pragma: no cover
    _sp = None


def _as_arrays(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a, b


def cohens_d_paired(a, b) -> float:
    """Cohen's d for paired samples (mean difference / std of differences)."""
    a, b = _as_arrays(a, b)
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def cohens_d_pooled(a, b) -> float:
    """Cohen's d for two independent samples (pooled standard deviation)."""
    a, b = _as_arrays(a, b)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(na + nb - 2, 1))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def cliffs_delta(a, b) -> float:
    """Cliff's delta, a non-parametric effect size in [-1, 1]."""
    a, b = _as_arrays(a, b)
    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    n = len(a) * len(b)
    return float((greater - less) / n) if n else 0.0


def _interpret_d(d: float) -> str:
    ad = abs(d)
    return "negligible" if ad < 0.2 else "small" if ad < 0.5 else "medium" if ad < 0.8 else "large"


def paired_test(a, b, parametric: bool = True) -> dict:
    """Paired comparison of two configurations measured on the same seeds.

    Args:
        a, b (list): Per-seed metric values (same length, same seed order).
        parametric (bool): Paired t-test if True, Wilcoxon signed-rank if False.

    Returns:
        result (dict): test name, statistic, p-value, mean difference, Cohen's d (paired),
            Cliff's delta and a plain-language effect-size label.
    """
    a, b = _as_arrays(a, b)
    if _sp is not None:
        if parametric:
            stat, p = _sp.ttest_rel(a, b)
            name = "paired t-test"
        else:
            try:
                stat, p = _sp.wilcoxon(a, b)
            except ValueError:  # all differences zero
                stat, p = 0.0, 1.0
            name = "Wilcoxon signed-rank"
    else:  # pragma: no cover - normal-approx fallback
        diff = a - b
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
        stat = diff.mean() / se if se > 0 else 0.0
        p = 2 * (1 - _normal_cdf(abs(stat)))
        name = "paired t-test (normal approx)"

    d = cohens_d_paired(a, b)
    return {
        "test": name,
        "statistic": float(stat),
        "p_value": float(p),
        "mean_difference": float(a.mean() - b.mean()),
        "cohens_d": d,
        "cliffs_delta": cliffs_delta(a, b),
        "effect": _interpret_d(d),
        "n": len(a),
    }


def independent_test(a, b, parametric: bool = True) -> dict:
    """Two-independent-group comparison (Welch's t-test or Mann-Whitney U)."""
    a, b = _as_arrays(a, b)
    if _sp is not None:
        if parametric:
            stat, p = _sp.ttest_ind(a, b, equal_var=False)
            name = "Welch's t-test"
        else:
            stat, p = _sp.mannwhitneyu(a, b, alternative="two-sided")
            name = "Mann-Whitney U"
    else:  # pragma: no cover
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        stat = (a.mean() - b.mean()) / se if se > 0 else 0.0
        p = 2 * (1 - _normal_cdf(abs(stat)))
        name = "Welch's t-test (normal approx)"

    d = cohens_d_pooled(a, b)
    return {
        "test": name,
        "statistic": float(stat),
        "p_value": float(p),
        "mean_difference": float(a.mean() - b.mean()),
        "cohens_d": d,
        "cliffs_delta": cliffs_delta(a, b),
        "effect": _interpret_d(d),
        "n_a": len(a),
        "n_b": len(b),
    }


def friedman_test(groups: dict) -> dict:
    """Omnibus non-parametric test for >2 paired groups (e.g. routing policies over seeds).

    Args:
        groups (dict): Maps a group label to its per-seed metric list (all same length).

    Returns:
        result (dict): test name, statistic, p-value and Kendall's W effect size.
    """
    labels = list(groups)
    matrix = np.array([groups[label] for label in labels], dtype=float)  # (k groups, n seeds)
    k, n = matrix.shape
    if _sp is not None and k >= 3:
        stat, p = _sp.friedmanchisquare(*[matrix[i] for i in range(k)])
    else:  # pragma: no cover
        stat, p = float("nan"), float("nan")
    kendalls_w = float(stat / (n * (k - 1))) if not np.isnan(stat) and n * (k - 1) > 0 else float("nan")
    return {"test": "Friedman", "statistic": float(stat), "p_value": float(p), "kendalls_w": kendalls_w, "k": k, "n": n}


def holm_bonferroni(p_values: list, alpha: float = 0.05) -> list:
    """Holm-Bonferroni step-down correction. Returns per-test dicts with adjusted p + reject flag."""
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    m = len(p_values)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)  # enforce monotonicity
        adjusted[idx] = running_max
    return [{"index": i, "p_raw": p_values[i], "p_adjusted": adjusted[i], "reject": adjusted[i] < alpha} for i in range(m)]


def benjamini_hochberg(p_values: list, alpha: float = 0.05) -> list:
    """Benjamini-Hochberg FDR correction. Returns per-test dicts with adjusted p + reject flag."""
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    m = len(p_values)
    adjusted = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx, p = indexed[rank]
        adj = min(prev, p * m / (rank + 1))
        adjusted[idx] = adj
        prev = adj
    return [{"index": i, "p_raw": p_values[i], "p_adjusted": adjusted[i], "reject": adjusted[i] < alpha} for i in range(m)]


def compare_all_pairs(groups: dict, paired: bool = True, correction: str = "holm", alpha: float = 0.05) -> list:
    """Runs all pairwise comparisons among groups with multiple-comparison correction.

    Args:
        groups (dict): Maps label to per-seed metric list.
        paired (bool): Use paired tests (same seeds) if True, else independent tests.
        correction (str): "holm" (Holm-Bonferroni) or "bh" (Benjamini-Hochberg).
        alpha (float): Significance level.

    Returns:
        rows (list): One dict per pair with labels, test result and corrected p / reject flag.
    """
    labels = list(groups)
    pairs = list(itertools.combinations(labels, 2))
    results = [paired_test(groups[a], groups[b]) if paired else independent_test(groups[a], groups[b]) for a, b in pairs]
    corrected = (benjamini_hochberg if correction == "bh" else holm_bonferroni)([r["p_value"] for r in results], alpha=alpha)
    rows = []
    for (a, b), res, corr in zip(pairs, results, corrected):
        rows.append({"group_a": a, "group_b": b, **res, "p_adjusted": corr["p_adjusted"], "significant": corr["reject"]})
    return rows


def _normal_cdf(x: float) -> float:  # pragma: no cover - only used without SciPy
    """Standard-normal CDF via the error function."""
    import math

    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
