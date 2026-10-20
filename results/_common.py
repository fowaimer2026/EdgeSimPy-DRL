"""Shared helpers for the results experiment suite.

Provides a common output directory, a consistent matplotlib style, a grouped bar-chart
helper with 95% CI error bars, and a policy-evaluation loop over the offloading
environment. Kept torch-free so the offloading/routing/execution studies run without
PyTorch installed.
"""

# Python libraries
import os

import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def output_path(name: str) -> str:
    """Returns an absolute path inside the results/output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, name)


def _matplotlib():
    """Imports matplotlib with a headless backend (lazy)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def grouped_bar_chart(categories, series, title, ylabel, path, errors=None):
    """Draws a grouped bar chart with optional CI error bars.

    Args:
        categories (list): X-axis group labels (e.g. policy names).
        series (dict): Maps a series label to its per-category values.
        title (str): Figure title.
        ylabel (str): Y-axis label.
        path (str): Output file path.
        errors (dict, optional): Maps a series label to per-category error-bar half-widths.
    """
    plt = _matplotlib()
    n_groups = len(categories)
    n_series = len(series)
    width = 0.8 / max(n_series, 1)
    x = np.arange(n_groups)

    figure, axis = plt.subplots(figsize=(7, 4.2))
    for index, (label, values) in enumerate(series.items()):
        offset = (index - (n_series - 1) / 2) * width
        err = errors.get(label) if errors else None
        axis.bar(x + offset, values, width, label=label, yerr=err, capsize=3)

    axis.set_xticks(x)
    axis.set_xticklabels(categories)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if n_series > 1:
        axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    figure.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(figure)
    return path


def line_chart(x, series, title, xlabel, ylabel, path, errors=None):
    """Draws a line chart (optionally with shaded CI bands) for one or more series."""
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(7, 4.2))
    for label, values in series.items():
        axis.plot(x, values, marker="o", label=label)
        if errors and label in errors:
            lower = np.array(values) - np.array(errors[label])
            upper = np.array(values) + np.array(errors[label])
            axis.fill_between(x, lower, upper, alpha=0.2)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if len(series) > 1:
        axis.legend()
    axis.grid(True, alpha=0.3)
    figure.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(figure)
    return path


def evaluate_offloading_policy(env, action_fn) -> dict:
    """Runs one episode of an offloading environment under a given action function.

    The environment's reward is the negative balanced weighted cost, so this measures
    every policy under the *same* cost metric regardless of how it chooses actions.

    Args:
        env (OffloadingEnv): The offloading environment.
        action_fn (callable): ``action_fn(env) -> int`` choosing a server for the current
            service (using ``env.pointer``, ``env.services``, ``env.servers``, ``env.action_mask``).

    Returns:
        metrics (dict): ``{"total_cost", "placement_rate"}`` for the episode.
    """
    env.reset()
    total_reward = 0.0
    done = False
    info = {"placed": 0}
    while not done:
        action = action_fn(env)
        _, reward, done, info = env.step(action)
        total_reward += reward
    return {"total_cost": -total_reward, "placement_rate": info["placed"] / len(env.services)}
