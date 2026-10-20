"""Workload execution subsystem for EdgeSimPy.

Adds the missing task-execution dynamics: discrete tasks that queue for CPU, execute for a
workload-dependent duration, and finish subject to deadlines. The execution engine can
accumulate energy (``edge_sim_py.energy``) and inject failures (``edge_sim_py.reliability``)
to produce ground-truth response-time, energy and reliability metrics.
"""

__version__ = "1.1.0"

from .task import Task, WAITING, RUNNING, FINISHED, FAILED
from .execution_engine import ExecutionEngine
from .placement import place_least_loaded, place_random, place_round_robin
from .metrics import collect_workload_metrics
from .scenario import random_workload, reset_workload_registries
