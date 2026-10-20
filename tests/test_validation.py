"""Validation tests: the contributed models must match their analytical references.

These encode the validation study as regression tests -- each model's empirical behavior is
checked against an independent closed-form reference using the experiment harness.
"""

# Python libraries
import math

import numpy as np
import pytest

# EdgeSimPy
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.reliability import FailureModel
from edge_sim_py.network import StochasticDelayModel
from edge_sim_py.workload import Task, ExecutionEngine, reset_workload_registries
from edge_sim_py.experiments import run_experiment


def test_failure_model_matches_steady_state_availability():
    def trial(seed):
        reset_workload_registries()
        server = EdgeServer(cpu=8, memory=4096, disk=1000)
        model = FailureModel(mtbf=50.0, mttr=10.0, seed=seed)
        available = sum(1 for step in range(10_000) if (model.step([server], step) or True) and server.available)
        return {"availability": available / 10_000}

    summary = run_experiment(trial, seeds=range(20))["summary"]["availability"]
    theoretical = FailureModel.steady_state_availability(50.0, 10.0)
    # The theoretical availability should fall within the empirical 95% CI.
    assert summary["ci_low"] <= theoretical <= summary["ci_high"]


def test_stochastic_delay_matches_expectation():
    def trial(seed):
        model = StochasticDelayModel(jitter_std=2.0, packet_loss=0.2, seed=seed)
        return {"mean_delay": float(np.mean([model.sample(10.0) for _ in range(10_000)]))}

    summary = run_experiment(trial, seeds=range(20))["summary"]["mean_delay"]
    analytical = StochasticDelayModel(jitter_std=2.0, packet_loss=0.2).expected_delay(10.0)
    assert summary["ci_low"] <= analytical <= summary["ci_high"]


def test_execution_time_matches_ceiling_model():
    reset_workload_registries()
    server = EdgeServer(cpu=64, memory=4096, disk=1000)
    tasks = [Task(cpu_cycles=cycles, cpu_demand=demand, arrival_step=0, server=server) for cycles in (10, 24, 50) for demand in (1, 2, 4)]
    ExecutionEngine([server], tasks).run()

    for task in tasks:
        analytical = math.ceil(task.cpu_cycles / task.cpu_demand)
        assert task.execution_time == analytical
