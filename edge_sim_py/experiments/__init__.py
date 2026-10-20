"""Reproducible experiment and statistics harness for EdgeSimPy.

Run a trial across many seeds, aggregate with confidence intervals, sweep parameters, and
export the results to CSV or plots -- turning single simulation runs into statistically
grounded, publishable evidence.
"""

__version__ = "1.1.0"

from .statistics import confidence_interval, aggregate
from .runner import run_trials, run_experiment, parameter_sweep
from .reporting import results_to_csv, results_to_dataframe, plot_sweep
from .significance import (
    paired_test,
    independent_test,
    friedman_test,
    cohens_d_paired,
    cohens_d_pooled,
    cliffs_delta,
    holm_bonferroni,
    benjamini_hochberg,
    compare_all_pairs,
)
