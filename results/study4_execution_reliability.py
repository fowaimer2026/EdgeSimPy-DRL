"""Study 4 - Execution dynamics and reliability.

Three experiments on the execution engine, each across seeds with 95% CIs:
    A. Load sweep      -- response time / deadline violations / energy vs. number of tasks.
    B. Failure sweep   -- response time / deadline violations / completion vs. MTBF.
    C. Validation      -- each model vs. its analytical reference (availability, delay, execution).
Emits CSV tables and figures.
"""

# EdgeSimPy
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.workload import (
    Task,
    ExecutionEngine,
    random_workload,
    place_least_loaded,
    collect_workload_metrics,
    reset_workload_registries,
)
from edge_sim_py.energy import EnergyMeter
from edge_sim_py.reliability import FailureModel
from edge_sim_py.network import StochasticDelayModel
from edge_sim_py.experiments import run_experiment, parameter_sweep, confidence_interval, results_to_csv

# Python libraries
import math

import numpy as np

from _common import output_path, line_chart, grouped_bar_chart

NUM_SERVERS = 4
SEEDS = range(30)


# ------------------------------------------------------------------ A. load sweep


def load_trial(seed, num_tasks=40):
    servers, tasks = random_workload(num_tasks=num_tasks, num_servers=NUM_SERVERS, seed=seed)
    place_least_loaded(tasks, servers)
    meter = EnergyMeter()
    engine = ExecutionEngine(servers, tasks, energy_meter=meter)
    engine.run()
    return collect_workload_metrics(tasks, engine)


# ------------------------------------------------------------------ B. failure sweep


def failure_trial(seed, mtbf=80.0):
    servers, tasks = random_workload(num_tasks=40, num_servers=NUM_SERVERS, seed=seed)
    place_least_loaded(tasks, servers)
    model = FailureModel(mtbf=mtbf, mttr=10.0, seed=seed)
    engine = ExecutionEngine(servers, tasks, failure_model=model)
    engine.run()
    return collect_workload_metrics(tasks, engine)


# ------------------------------------------------------------------ C. validation


def availability_trial(seed):
    reset_workload_registries()
    server = EdgeServer(cpu=8, memory=4096, disk=1000)
    model = FailureModel(mtbf=50.0, mttr=10.0, seed=seed)
    available = sum(1 for step in range(10_000) if (model.step([server], step) or True) and server.available)
    return {"availability": available / 10_000}


def delay_trial(seed):
    model = StochasticDelayModel(jitter_std=2.0, packet_loss=0.2, seed=seed)
    return {"mean_delay": float(np.mean([model.sample(10.0) for _ in range(10_000)]))}


def main():
    print(f"Study 4 - Execution & reliability ({len(list(SEEDS))} seeds, 95% CI)\n")

    # --- A. Load sweep ---
    print("A. Load sweep (tasks on 4 servers):")
    task_counts = [10, 20, 40, 60, 80]
    load = parameter_sweep(load_trial, "num_tasks", task_counts, seeds=SEEDS)
    for row in load:
        print(
            f"  tasks={row['num_tasks']:>3}: response={row['mean_response_time_mean']:.2f}"
            f"+/-{row['mean_response_time_ci']:.2f}  deadline_viol={row['deadline_violation_rate_mean']:.3f}"
            f"  energy={row['total_energy_mean']:.0f}"
        )
    results_to_csv(load, output_path("study4_load_sweep.csv"))
    load_fig = line_chart(
        x=task_counts,
        series={"response time": [r["mean_response_time_mean"] for r in load], "waiting time": [r["mean_waiting_time_mean"] for r in load]},
        errors={"response time": [r["mean_response_time_ci"] for r in load], "waiting time": [r["mean_waiting_time_ci"] for r in load]},
        title="Execution: response/waiting time vs. load",
        xlabel="Number of tasks (on 4 servers)",
        ylabel="Steps",
        path=output_path("study4_load_sweep.png"),
    )

    # --- B. Failure sweep ---
    print("\nB. Failure sweep (MTBF, MTTR=10):")
    mtbf_values = [20, 40, 80, 160, 100_000]  # last ~ no failures
    failure = parameter_sweep(failure_trial, "mtbf", mtbf_values, seeds=SEEDS)
    for row in failure:
        label = "none" if row["mtbf"] >= 100_000 else str(row["mtbf"])
        print(
            f"  MTBF={label:>6}: response={row['mean_response_time_mean']:.2f}"
            f"  deadline_viol={row['deadline_violation_rate_mean']:.3f}"
            f"  completion={row['completion_rate_mean']:.3f}"
        )
    results_to_csv(failure, output_path("study4_failure_sweep.csv"))
    labels = [("none" if v >= 100_000 else str(v)) for v in mtbf_values]
    fail_fig = grouped_bar_chart(
        categories=labels,
        series={
            "response time": [r["mean_response_time_mean"] for r in failure],
            "deadline viol. x100": [r["deadline_violation_rate_mean"] * 100 for r in failure],
        },
        errors={
            "response time": [r["mean_response_time_ci"] for r in failure],
            "deadline viol. x100": [r["deadline_violation_rate_ci"] * 100 for r in failure],
        },
        title="Reliability: failure impact (MTTR=10)",
        ylabel="steps  /  violation% ",
        path=output_path("study4_failure_impact.png"),
    )

    # --- C. Validation ---
    print("\nC. Model validation (empirical vs. analytical, within 95% CI):")
    val_rows = []
    avail = run_experiment(availability_trial, seeds=SEEDS)["summary"]["availability"]
    val_rows.append(("server_availability", avail, FailureModel.steady_state_availability(50.0, 10.0)))
    delay = run_experiment(delay_trial, seeds=SEEDS)["summary"]["mean_delay"]
    val_rows.append(("stochastic_delay", delay, StochasticDelayModel(jitter_std=2.0, packet_loss=0.2).expected_delay(10.0)))

    csv_rows = []
    for name, summary, analytical in val_rows:
        within = summary["ci_low"] <= analytical <= summary["ci_high"]
        print(
            f"  {name:<22} empirical={summary['mean']:.4f} CI=[{summary['ci_low']:.4f},{summary['ci_high']:.4f}]"
            f" analytical={analytical:.4f} -> {'PASS' if within else 'review'}"
        )
        csv_rows.append(
            {
                "model": name,
                "empirical": summary["mean"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "analytical": analytical,
                "within_ci": within,
            }
        )
    # Execution time is deterministic: exact ceiling model.
    reset_workload_registries()
    server = EdgeServer(cpu=64, memory=4096, disk=1000)
    tasks = [Task(cpu_cycles=c, cpu_demand=d, arrival_step=0, server=server) for c in (10, 24, 50) for d in (1, 2, 4)]
    ExecutionEngine([server], tasks).run()
    exec_ok = all(t.execution_time == math.ceil(t.cpu_cycles / t.cpu_demand) for t in tasks)
    print(f"  {'execution_time':<22} measured == ceil(cycles/demand) for all tasks -> {'PASS' if exec_ok else 'review'}")
    csv_rows.append({"model": "execution_time", "empirical": 0.0, "ci_low": 0.0, "ci_high": 0.0, "analytical": 0.0, "within_ci": exec_ok})
    results_to_csv(csv_rows, output_path("study4_validation.csv"))

    print(f"\nWrote CSV + figures to {output_path('')}")
    print(f"       {load_fig}\n       {fail_fig}")


if __name__ == "__main__":
    main()
