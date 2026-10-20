"""Automatic Python configuration file."""

__version__ = "1.1.0"

# Reference computation-offloading policies
from .min_cost_offloading import min_cost_offloading

# Baseline policies for benchmarking
from .baseline_policies import latency_greedy_offloading, random_offloading

# Offloading metrics aggregation
from .metrics import collect_cost_metrics, collect_sla_metrics, collect_offloading_metrics
