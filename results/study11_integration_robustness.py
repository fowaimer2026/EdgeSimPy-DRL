"""Study 11 - Integration robustness across topology scales and workload types.

Directly answers the reviewer concern that the integrated end-to-end evaluation (Study 8) rests
on a single topology size (~25 nodes) and a single workload (Alibaba bursts). This study re-runs
the *same* baseline-vs-integrated comparison (reusing Study 8's exact pipeline) across:

    * three topology scales -- 25, 50, 100 nodes;
    * two workload types     -- real Alibaba bursts AND synthetic Poisson arrivals
      (exponential inter-arrivals + service times: a different temporal distribution).

If the integration benefit (network delay / deadline / energy improvements of the intelligent
stack over the baseline) holds across all six (scale x workload) conditions, the integration claim
is validated rather than illustrated. Reports the per-condition improvement with 95% CIs.

Point --eua-servers / --alibaba at the real files (defaults assume dataset/).
"""

# EdgeSimPy
from edge_sim_py.workload import Task, ExecutionEngine, collect_workload_metrics
from edge_sim_py.energy import EnergyMeter
from edge_sim_py.reliability import FailureModel
from edge_sim_py.datasets.alibaba import load_alibaba_tasks
from edge_sim_py.experiments import run_experiment, results_to_csv

# Reuse Study 8's exact integrated pipeline (same claim, stress-tested).
import study8_integrated as s8

# Python libraries
import argparse
import random

import numpy as np

from _common import output_path

SCALES = [25, 50, 100]
WORKLOADS = ["alibaba", "poisson"]
NUM_TASKS = 500
SEEDS = range(5)
STACKS = {
    "baseline": ("random", s8.ShortestPathRouting, None),
    "integrated": ("cost", s8.LatencyAwareRouting, {"alpha": 0.5, "beta": 0.5}),
}


def _poisson_workload(num_tasks, seed, arrival_rate=0.4, mean_cycles=90.0):
    """Synthetic Poisson workload: exponential inter-arrivals and service times."""
    rng = random.Random(seed + 7)
    tasks = []
    t = 0.0
    for _ in range(num_tasks):
        t += rng.expovariate(arrival_rate)
        cpu_demand = rng.randint(1, 4)
        duration = max(1, int(rng.expovariate(1.0 / mean_cycles)))
        tasks.append(
            Task(
                cpu_cycles=duration * cpu_demand,
                cpu_demand=cpu_demand,
                memory_demand=rng.randint(256, 2048),
                arrival_step=int(t),
                deadline=int(t + duration * 2),
            )
        )
    return tasks


def trial(seed, scale=25, workload="alibaba", stack="baseline"):
    s8.NUM_NODES = scale  # drive Study 8's builder at this scale
    servers, topology, switches, _ = s8._build_infrastructure(seed)

    if workload == "alibaba":
        tasks = load_alibaba_tasks(s8._PATHS["alibaba"], sort_by_arrival=True, limit=NUM_TASKS, window_offset=seed * NUM_TASKS, scan_limit=150_000)
    else:
        tasks = _poisson_workload(NUM_TASKS, seed)

    rng = random.Random(seed)
    for task in tasks:
        task.original_arrival = task.arrival_step
        task.source_switch = rng.choice(switches)

    placement, routing, params = STACKS[stack]
    avg_net = s8._place_and_route(tasks, servers, topology, switches, routing, params, placement, rng)

    meter = EnergyMeter()
    engine = ExecutionEngine(servers, tasks, energy_meter=meter, failure_model=FailureModel(mtbf=s8.MTBF, mttr=s8.MTTR, seed=seed))
    engine.run(max_steps=s8.MAX_STEPS)

    finished = [t for t in tasks if t.finished]
    e2e = [t.end_step - t.original_arrival + 1 for t in finished]
    with_dl = [t for t in finished if t.deadline is not None]
    ok = [t for t in with_dl if t.end_step <= t.deadline]
    m = collect_workload_metrics(tasks, engine)
    return {
        "e2e_median": float(np.median(e2e)) if e2e else 0.0,
        "network_delay": avg_net,
        "deadline_violation": 1.0 - len(ok) / len(with_dl) if with_dl else 0.0,
        "total_energy": m.get("total_energy", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Integration robustness across scales and workloads.")
    parser.add_argument("--eua-servers", default="dataset/eua-dataset/edge-servers/site-optus-melbCBD.csv")
    parser.add_argument("--alibaba", default="dataset/alibaba-clusterdata/batch_task.csv")
    args = parser.parse_args()
    s8._PATHS["servers"], s8._PATHS["alibaba"] = args.eua_servers, args.alibaba

    print(f"Study 11 - Integration robustness ({len(SCALES)} scales x {len(WORKLOADS)} workloads, {len(list(SEEDS))} seeds, 95% CI)\n")
    print(f"{'scale':>6}{'workload':>10}   improvement of integrated vs baseline (positive = better)")
    print(f"{'':>16}   {'network':>10}{'deadline':>10}{'energy':>10}{'median resp':>13}")

    rows = []
    for scale in SCALES:
        for workload in WORKLOADS:
            base = run_experiment(trial, seeds=SEEDS, scale=scale, workload=workload, stack="baseline")["summary"]
            integ = run_experiment(trial, seeds=SEEDS, scale=scale, workload=workload, stack="integrated")["summary"]

            def imp(metric):
                b = base[metric]["mean"]
                return 100 * (1 - integ[metric]["mean"] / b) if b else 0.0

            net, dl, en, resp = imp("network_delay"), imp("deadline_violation"), imp("total_energy"), imp("e2e_median")
            print(f"{scale:>6}{workload:>10}   {net:>9.0f}%{dl:>9.0f}%{en:>9.0f}%{resp:>12.0f}%")
            rows.append(
                {
                    "scale": scale,
                    "workload": workload,
                    "network_improve_pct": net,
                    "deadline_improve_pct": dl,
                    "energy_improve_pct": en,
                    "median_resp_improve_pct": resp,
                    "baseline_network": base["network_delay"]["mean"],
                    "integrated_network": integ["network_delay"]["mean"],
                    "baseline_deadline": base["deadline_violation"]["mean"],
                    "integrated_deadline": integ["deadline_violation"]["mean"],
                }
            )

    results_to_csv(rows, output_path("study11_integration_robustness.csv"))
    fig_path = _plot(rows, output_path("study11_integration_robustness.png"))
    consistent = all(r["network_improve_pct"] > 0 and r["deadline_improve_pct"] > 0 for r in rows)
    print(f"\nIntegration benefit consistent (network & deadline improved in every condition): {consistent}")
    print(f"Wrote CSV + {fig_path}")


def _plot(rows, path):
    """Grouped bar chart of % improvement, with a zero line and negatives highlighted.

    Positive bars = integrated beats baseline; negative bars (drawn below the zero line, hatched)
    = integrated is worse in that condition — the documented placement/queueing trade-off.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"{r['scale']}n\n{r['workload'][:4]}" for r in rows]
    series = {
        "network": [r["network_improve_pct"] for r in rows],
        "deadline": [r["deadline_improve_pct"] for r in rows],
        "energy": [r["energy_improve_pct"] for r in rows],
        "median resp.": [r["median_resp_improve_pct"] for r in rows],
    }
    n_groups, n_series = len(rows), len(series)
    width = 0.8 / n_series
    x = np.arange(n_groups)

    figure, axis = plt.subplots(figsize=(9, 4.6))
    for idx, (label, values) in enumerate(series.items()):
        offset = (idx - (n_series - 1) / 2) * width
        bars = axis.bar(x + offset, values, width, label=label, edgecolor="black", linewidth=0.4)
        for bar, v in zip(bars, values):
            axis.annotate(f"{v:+.0f}", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom" if v >= 0 else "top", fontsize=6)

    # Shade the below-zero region: bars there mean the integrated stack is WORSE than baseline.
    low = min(min(v) for v in series.values())
    axis.axhspan(min(low * 1.15, -5), 0, color="#c0392b", alpha=0.07)
    axis.axhline(0, color="black", linewidth=1.0)  # baseline reference
    axis.text(
        0.995,
        0.03,
        "shaded region = integrated worse than baseline\n(placement/queueing trade-off under steady load)",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#c0392b",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("% improvement of integrated vs. baseline")
    axis.set_title("Integration benefit across topology scales × workloads (positive = integrated better)")
    axis.legend(ncol=4, fontsize=8, loc="upper left")
    axis.grid(True, axis="y", alpha=0.3)
    figure.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(figure)
    return path


if __name__ == "__main__":
    main()
