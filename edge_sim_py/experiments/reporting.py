"""CSV export and plotting helpers for experiment results.

Kept dependency-light: CSV export uses the standard library, while pandas and matplotlib
are imported lazily so the core experiment harness never requires them.
"""

# Python libraries
import csv


def results_to_csv(rows: list, path: str) -> str:
    """Writes a list of flat metric dicts to a CSV file.

    Args:
        rows (list): List of dictionaries sharing the same keys.
        path (str): Output file path.

    Returns:
        path (str): The path written to.
    """
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def results_to_dataframe(rows: list):
    """Returns the rows as a pandas DataFrame (pandas imported lazily)."""
    import pandas as pd

    return pd.DataFrame(rows)


def plot_sweep(rows: list, x: str, y: str, path: str = None, show: bool = False):
    """Plots a swept metric with confidence-interval error bars (matplotlib lazy import).

    Args:
        rows (list): Sweep rows from :func:`parameter_sweep` (expects ``y_mean``/``y_ci``).
        x (str): Column for the x-axis (the swept parameter name).
        y (str): Metric base name; uses ``<y>_mean`` and ``<y>_ci`` columns.
        path (str, optional): If given, saves the figure to this path. Defaults to None.
        show (bool, optional): If True, displays the figure. Defaults to False.

    Returns:
        path (str | None): The path saved to, if any.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")  # headless backend for file output
    import matplotlib.pyplot as plt

    xs = [row[x] for row in rows]
    means = [row[f"{y}_mean"] for row in rows]
    errors = [row.get(f"{y}_ci", 0.0) for row in rows]

    figure, axis = plt.subplots()
    axis.errorbar(xs, means, yerr=errors, marker="o", capsize=4)
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    axis.set_title(f"{y} vs {x} (95% CI)")
    axis.grid(True, alpha=0.3)

    if path:
        figure.savefig(path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close(figure)
    return path
