"""Validation study: checking the simulator's models against ground truth.

Rather than adding features, this script demonstrates *methodology* -- it uses the
experiment/statistics harness to validate three of the contributed models against
independent references, reporting each with a 95% confidence interval:

    1. Reliability  -- empirical server availability vs. the analytical steady-state
       availability MTBF / (MTBF + MTTR).
    2. Network      -- empirical mean stochastic delay vs. the closed-form expected delay.
    3. Execution    -- measured task execution time vs. the analytical compute time
       cpu_cycles / (cpu_demand * processing_rate).

A model is considered validated when its analytical reference falls within (or extremely
close to) the empirical confidence interval. Results are also written to CSV.

Run with:  python examples/validation_study.py
"""

# Python libraries
import math

import numpy as np

# EdgeSimPy
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.reliability import FailureModel
from edge_sim_py.network import StochasticDelayModel
from edge_sim_py.workload import Task, ExecutionEngine, reset_workload_registries
from edge_sim_py.experiments import run_experiment, results_to_csv

# ---------------------------------------------------------------- 1. reliability


def availability_trial(seed, mtbf=50.0, mttr=10.0, horizon=20_000):
    """Measures empirical availability of one server over a long failure process."""
    reset_workload_registries()
    server = EdgeServer(cpu=8, memory=4096, disk=1000)
    model = FailureModel(mtbf=mtbf, mttr=mttr, seed=seed)
    available_ticks = 0
    for step in range(horizon):
        model.step([server], current_step=step)
        available_ticks += 1 if server.available else 0
    return {"availability": available_ticks / horizon}


# ---------------------------------------------------------------- 2. network


def delay_trial(seed, base_delay=10.0, jitter_std=2.0, packet_loss=0.2, samples=20_000):
    """Measures the empirical mean of the stochastic delay sampler."""
    model = StochasticDelayModel(jitter_std=jitter_std, packet_loss=packet_loss, seed=seed)
    values = [model.sample(base_delay) for _ in range(samples)]
    return {"mean_delay": float(np.mean(values))}


# ---------------------------------------------------------------- 3. execution


def execution_model_error():
    """Returns the max relative error between measured and analytical execution times."""
    reset_workload_registries()
    server = EdgeServer(cpu=64, memory=4096, disk=1000)
    tasks = [Task(cpu_cycles=cycles, cpu_demand=demand, arrival_step=0, server=server) for cycles in (10, 24, 50) for demand in (1, 2, 4)]
    ExecutionEngine([server], tasks).run()

    max_error = 0.0
    for task in tasks:
        # Integer-step, space-sharing model: whole ticks, so the reference is the ceiling
        # of cpu_cycles / (cpu_demand * processing_rate) (processing_rate = 1 here).
        analytical = math.ceil(task.cpu_cycles / task.cpu_demand)
        max_error = max(max_error, abs(task.execution_time - analytical) / analytical)
    return max_error


def report(name, summary, analytical):
    mean, lo, hi = summary["mean"], summary["ci_low"], summary["ci_high"]
    within = lo - 1e-9 <= analytical <= hi + 1e-9
    verdict = "PASS" if within else "review"
    print(f"  {name:<28} empirical={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  analytical={analytical:.4f}  -> {verdict}")
    return {"model": name, "empirical": mean, "ci_low": lo, "ci_high": hi, "analytical": analytical, "within_ci": within}


def main():
    seeds = range(30)
    print("Validation study (30 seeds, 95% confidence intervals)\n")
    rows = []

    print("1. Reliability - availability vs MTBF/(MTBF+MTTR):")
    exp = run_experiment(availability_trial, seeds=seeds, mtbf=50.0, mttr=10.0)
    theoretical = FailureModel.steady_state_availability(50.0, 10.0)
    rows.append(report("server_availability", exp["summary"]["availability"], theoretical))

    print("\n2. Network - mean delay vs closed-form expectation:")
    exp = run_experiment(delay_trial, seeds=seeds, base_delay=10.0, jitter_std=2.0, packet_loss=0.2)
    analytical_delay = StochasticDelayModel(jitter_std=2.0, packet_loss=0.2).expected_delay(10.0)
    rows.append(report("stochastic_delay", exp["summary"]["mean_delay"], analytical_delay))

    print("\n3. Execution - measured vs analytical compute time:")
    error = execution_model_error()
    print(f"  {'execution_time':<28} max relative error = {error:.4%}  -> {'PASS' if error < 1e-9 else 'review'}")
    rows.append({"model": "execution_time", "empirical": error, "ci_low": 0, "ci_high": 0, "analytical": 0.0, "within_ci": error < 1e-9})

    path = results_to_csv(rows, "validation_results.csv")
    print(f"\nValidation results written to {path}")


if __name__ == "__main__":
    main()
