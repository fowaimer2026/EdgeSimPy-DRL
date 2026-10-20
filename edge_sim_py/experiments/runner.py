"""Reproducible multi-seed experiment runner and parameter sweeps.

A *trial function* maps a seed (and optional parameters) to a dict of scalar metrics.
These helpers run it across many seeds, aggregate the results with confidence intervals,
and sweep a parameter -- producing the kind of statistically-grounded comparison tables
that a single simulation run cannot.
"""

# EdgeSimPy experiments
from edge_sim_py.experiments.statistics import aggregate, confidence_interval


def run_trials(trial_fn, seeds: list, **params) -> list:
    """Runs a trial function once per seed.

    Args:
        trial_fn (callable): ``trial_fn(seed=..., **params) -> dict`` of scalar metrics.
        seeds (list): Seeds to run (defines the number of repetitions).
        **params: Fixed parameters forwarded to every trial.

    Returns:
        results (list): One metric dict per seed.
    """
    return [trial_fn(seed=seed, **params) for seed in seeds]


def run_experiment(trial_fn, seeds: list, metrics: list = None, confidence: float = 0.95, **params) -> dict:
    """Runs a trial across seeds and returns both raw results and aggregated CIs.

    Args:
        trial_fn (callable): Trial function (see :func:`run_trials`).
        seeds (list): Seeds to run.
        metrics (list, optional): Metric keys to aggregate. Defaults to all numeric keys.
        confidence (float, optional): Confidence level. Defaults to 0.95.
        **params: Fixed parameters forwarded to every trial.

    Returns:
        experiment (dict): ``{"raw": [...], "summary": {metric: ci_summary}}``.
    """
    results = run_trials(trial_fn, seeds, **params)
    return {"raw": results, "summary": aggregate(results, metrics=metrics, confidence=confidence)}


def parameter_sweep(trial_fn, param_name: str, param_values: list, seeds: list, metrics: list = None, confidence: float = 0.95, **fixed) -> list:
    """Sweeps one parameter, running a multi-seed experiment at each value.

    Args:
        trial_fn (callable): Trial function accepting ``seed`` and ``param_name``.
        param_name (str): Name of the swept parameter.
        param_values (list): Values to sweep.
        seeds (list): Seeds per value.
        metrics (list, optional): Metrics to aggregate. Defaults to all numeric keys.
        confidence (float, optional): Confidence level. Defaults to 0.95.
        **fixed: Other fixed parameters forwarded to the trial.

    Returns:
        rows (list): One flattened row per swept value, with ``<metric>_mean`` and
            ``<metric>_ci`` columns suitable for CSV export or plotting.
    """
    rows = []
    for value in param_values:
        experiment = run_experiment(trial_fn, seeds, metrics=metrics, confidence=confidence, **{param_name: value}, **fixed)
        row = {param_name: value}
        for metric, summary in experiment["summary"].items():
            row[f"{metric}_mean"] = summary["mean"]
            row[f"{metric}_ci"] = summary["half_width"]
            row[f"{metric}_std"] = summary["std"]
        rows.append(row)
    return rows
