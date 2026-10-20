"""Statistical aggregation for multi-seed simulation experiments.

Turns a list of per-seed metric dictionaries into mean / standard deviation / standard
error / confidence interval summaries -- the rigor a single simulation run cannot provide.
Uses NumPy only; the Student-t critical value is taken from SciPy when available, falling
back to a normal approximation otherwise.
"""

# Python libraries
import numpy as np

try:  # SciPy gives exact small-sample t critical values; optional.
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - exercised only when SciPy is absent
    _scipy_stats = None


def _t_critical(confidence: float, dof: int) -> float:
    """Returns the two-sided t critical value (normal approximation if SciPy is missing)."""
    if dof <= 0:
        return float("nan")
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(0.5 + confidence / 2.0, dof))
    # Normal approximation fallback for the common 95% case.
    return 1.96 if abs(confidence - 0.95) < 1e-9 else float(np.sqrt(2) * _erfinv(confidence))


def _erfinv(x: float) -> float:
    """Crude inverse error function (only used in the SciPy-absent fallback path)."""
    a = 0.147
    ln = np.log(1 - x * x)
    term = 2 / (np.pi * a) + ln / 2
    return np.sign(x) * np.sqrt(np.sqrt(term * term - ln / a) - term)


def confidence_interval(values: list, confidence: float = 0.95) -> dict:
    """Computes the mean and a two-sided confidence interval for a sample.

    Args:
        values (list): Sample values.
        confidence (float, optional): Confidence level. Defaults to 0.95.

    Returns:
        summary (dict): mean, std (sample), sem, ci_low, ci_high, half_width and n.
    """
    array = np.asarray(values, dtype=float)
    n = array.size
    mean = float(array.mean()) if n else 0.0
    if n < 2:
        return {"mean": mean, "std": 0.0, "sem": 0.0, "ci_low": mean, "ci_high": mean, "half_width": 0.0, "n": n}

    std = float(array.std(ddof=1))
    sem = std / np.sqrt(n)
    half_width = _t_critical(confidence, n - 1) * sem
    return {
        "mean": mean,
        "std": std,
        "sem": float(sem),
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "half_width": float(half_width),
        "n": n,
    }


def aggregate(results: list, metrics: list = None, confidence: float = 0.95) -> dict:
    """Aggregates a list of per-seed metric dicts into per-metric CI summaries.

    Args:
        results (list): List of dicts (one per seed) sharing numeric metric keys.
        metrics (list, optional): Metric keys to aggregate. Defaults to all numeric keys
            present in the first result.
        confidence (float, optional): Confidence level. Defaults to 0.95.

    Returns:
        summary (dict): Maps each metric name to its ``confidence_interval`` summary.
    """
    if not results:
        return {}
    if metrics is None:
        metrics = [key for key, value in results[0].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]

    summary = {}
    for metric in metrics:
        values = [result[metric] for result in results if metric in result]
        summary[metric] = confidence_interval(values, confidence=confidence)
    return summary
