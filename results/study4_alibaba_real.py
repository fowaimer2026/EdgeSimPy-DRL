"""Study 4 (real data) - Real Alibaba workload on the real EUA infrastructure.

Runs a burst of real Alibaba cluster-trace tasks on the real EUA (Melbourne) edge servers.
Because the trace's tasks are small/short relative to 125 edge servers (so the full
infrastructure is lightly loaded), two sweeps expose the interesting behaviour with 95% CIs:

    A. Capacity sweep -- shrink the number of available edge servers; queueing, response time
       and deadline violations rise as capacity falls.
    B. Failure sweep  -- inject MTBF/MTTR server failures on the real workload.

Each seed selects a different real burst (via the trace window offset) for natural variation.
Point --eua-servers / --alibaba at the real files (defaults assume dataset/).
"""

# EdgeSimPy
from edge_sim_py.datasets.eua import load_eua_servers
from edge_sim_py.datasets.alibaba import load_alibaba_tasks
from edge_sim_py.workload import ExecutionEngine, place_least_loaded, collect_workload_metrics
from edge_sim_py.workload.scenario import reset_workload_registries
from edge_sim_py.energy import EnergyMeter
from edge_sim_py.reliability import FailureModel
from edge_sim_py.experiments import parameter_sweep, results_to_csv

# Python libraries
import argparse
import random

from _common import output_path, line_chart, grouped_bar_chart

NUM_TASKS = 1000
SEEDS = range(5)
MAX_STEPS = 40_000  # safety guard so no single run can run away
_PATHS = {}


def _load(seed):
    """Loads a fresh EUA infrastructure and a distinct real Alibaba burst for this seed."""
    reset_workload_registries()
    servers = load_eua_servers(_PATHS["servers"])
    tasks = load_alibaba_tasks(_PATHS["alibaba"], sort_by_arrival=True, limit=NUM_TASKS, window_offset=seed * NUM_TASKS, scan_limit=200_000)
    return servers, tasks


def capacity_trial(seed, num_servers=125):
    servers, tasks = _load(seed)
    servers = random.Random(seed).sample(servers, min(num_servers, len(servers)))
    place_least_loaded(tasks, servers)
    meter = EnergyMeter()
    engine = ExecutionEngine(servers, tasks, energy_meter=meter)
    engine.run(max_steps=MAX_STEPS)
    return collect_workload_metrics(tasks, engine)


def failure_trial(seed, mtbf=400.0):
    servers, tasks = _load(seed)
    servers = random.Random(seed).sample(servers, 30)  # constrained capacity so failures bite
    place_least_loaded(tasks, servers)
    # MTBF is kept well above the ~90-step task durations: with restart-on-failure, an MTBF
    # below the task duration would make tasks restart faster than they finish (pathological).
    model = FailureModel(mtbf=mtbf, mttr=20.0, seed=seed)
    engine = ExecutionEngine(servers, tasks, failure_model=model)
    engine.run(max_steps=MAX_STEPS)
    return collect_workload_metrics(tasks, engine)


def main():
    parser = argparse.ArgumentParser(description="Real Alibaba workload on real EUA servers.")
    parser.add_argument("--eua-servers", default="dataset/eua-dataset/edge-servers/site-optus-melbCBD.csv")
    parser.add_argument("--alibaba", default="dataset/alibaba-clusterdata/batch_task.csv")
    args = parser.parse_args()
    _PATHS["servers"], _PATHS["alibaba"] = args.eua_servers, args.alibaba

    print(f"Study 4 (REAL Alibaba on REAL EUA) - {NUM_TASKS} tasks/burst, {len(list(SEEDS))} seeds, 95% CI\n")

    # --- A. Capacity sweep ---
    print("A. Capacity sweep (available edge servers):")
    server_counts = [125, 60, 30, 15]
    cap = parameter_sweep(capacity_trial, "num_servers", server_counts, seeds=SEEDS)
    for row in cap:
        print(
            f"  servers={row['num_servers']:>3}: response={row['mean_response_time_mean']:.1f}"
            f"  waiting={row['mean_waiting_time_mean']:.1f}  deadline_viol={row['deadline_violation_rate_mean']:.3f}"
            f"  completion={row['completion_rate_mean']:.3f}"
        )
    results_to_csv(cap, output_path("study4_alibaba_capacity.csv"))
    cap_fig = line_chart(
        x=server_counts,
        series={"response time": [r["mean_response_time_mean"] for r in cap], "waiting time": [r["mean_waiting_time_mean"] for r in cap]},
        errors={"response time": [r["mean_response_time_ci"] for r in cap], "waiting time": [r["mean_waiting_time_ci"] for r in cap]},
        title="Real Alibaba workload: response/waiting vs. edge capacity",
        xlabel="Number of edge servers",
        ylabel="Steps",
        path=output_path("study4_alibaba_capacity.png"),
    )

    # --- B. Failure sweep (on constrained 30-server capacity) ---
    print("\nB. Failure sweep (30 servers, MTTR=20):")
    mtbf_values = [200, 400, 800, 100_000]
    fail = parameter_sweep(failure_trial, "mtbf", mtbf_values, seeds=SEEDS)
    for row in fail:
        label = "none" if row["mtbf"] >= 100_000 else str(row["mtbf"])
        print(
            f"  MTBF={label:>6}: response={row['mean_response_time_mean']:.1f}"
            f"  deadline_viol={row['deadline_violation_rate_mean']:.3f}"
            f"  completion={row['completion_rate_mean']:.3f}"
        )
    results_to_csv(fail, output_path("study4_alibaba_failure.csv"))
    labels = [("none" if v >= 100_000 else str(v)) for v in mtbf_values]
    fail_fig = grouped_bar_chart(
        categories=labels,
        series={"response time": [r["mean_response_time_mean"] for r in fail], "deadline viol. x100": [r["deadline_violation_rate_mean"] * 100 for r in fail]},
        errors={"response time": [r["mean_response_time_ci"] for r in fail], "deadline viol. x100": [r["deadline_violation_rate_ci"] * 100 for r in fail]},
        title="Real Alibaba workload: failure impact (30 servers)",
        ylabel="steps  /  violation%",
        path=output_path("study4_alibaba_failure.png"),
    )
    print(f"\nWrote CSVs + figures to {output_path('')}\n       {cap_fig}\n       {fail_fig}")


if __name__ == "__main__":
    main()
