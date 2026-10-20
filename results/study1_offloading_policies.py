"""Study 1 - Offloading policy comparison.

Compares three service-placement policies under the *same* balanced weighted-cost metric
(latency 0.5 / energy 0.3 / money 0.2):

    * cost-driven   -- picks the minimum weighted-cost server (the contributed policy);
    * latency-greedy -- picks the minimum-latency server;
    * random        -- picks a random capable server.

Reports mean cost and placement rate across 30 seeds with 95% confidence intervals, plus a
sweep over the number of edge servers. Emits a CSV table and two figures.
"""

# EdgeSimPy (torch-free imports)
from edge_sim_py.drl.offloading_env import OffloadingEnv
from edge_sim_py.drl.scenario import random_scenario
from edge_sim_py.components.cost_models.latency_cost_model import LatencyCostModel
from edge_sim_py.experiments import run_experiment, parameter_sweep, results_to_csv

# Python libraries
import random

import numpy as np

from _common import output_path, grouped_bar_chart, line_chart, evaluate_offloading_policy

COST_PARAMS = {"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}}
NUM_SERVICES = 12
SEEDS = range(30)


def _cost_driven(env):
    return env.greedy_action()


def _random(env):
    return env.random_action()


def _latency_greedy(env):
    service = env.services[env.pointer]
    costs = [LatencyCostModel.get_cost(service, server, None, COST_PARAMS) for server in env.servers]
    masked = [cost if valid else np.inf for cost, valid in zip(costs, env.action_mask)]
    return int(np.argmin(masked))


POLICIES = {"cost_driven": _cost_driven, "latency_greedy": _latency_greedy, "random": _random}


def trial(seed, num_servers=4):
    """Evaluates every policy on the same scenario, returning per-policy cost and placement."""
    services, servers = random_scenario(num_services=NUM_SERVICES, num_servers=num_servers, seed=seed)

    # Heterogeneous pricing across servers (e.g. community vs. premium operators), so the
    # multi-objective cost-driven policy differs from a latency-only greedy policy.
    rng = random.Random(seed)
    for server in servers:
        server.cost_model_parameters = {"price_per_cpu_unit_time": rng.choice([0.02, 0.05, 0.2])}

    env = OffloadingEnv(services, servers, cost_parameters=COST_PARAMS)
    result = {}
    for name, action_fn in POLICIES.items():
        metrics = evaluate_offloading_policy(env, action_fn)
        result[f"{name}_cost"] = metrics["total_cost"]
        result[f"{name}_placement"] = metrics["placement_rate"]
    return result


def main():
    print("Study 1 - Offloading policies (30 seeds, 95% CI)\n")

    # Fixed-setting comparison table.
    summary = run_experiment(trial, seeds=SEEDS, num_servers=4)["summary"]
    rows = []
    print(f"{'Policy':<16}{'Mean cost':>14}{'95% CI':>12}{'Placement':>12}")
    for name in POLICIES:
        cost = summary[f"{name}_cost"]
        placement = summary[f"{name}_placement"]
        print(f"{name:<16}{cost['mean']:>14.4f}{cost['half_width']:>12.4f}{placement['mean']:>12.3f}")
        rows.append(
            {
                "policy": name,
                "mean_cost": cost["mean"],
                "cost_ci": cost["half_width"],
                "mean_placement": placement["mean"],
                "placement_ci": placement["half_width"],
            }
        )
    csv_path = results_to_csv(rows, output_path("study1_offloading_policies.csv"))

    # Bar chart of mean cost with CI error bars.
    bar_path = grouped_bar_chart(
        categories=list(POLICIES.keys()),
        series={"mean cost": [row["mean_cost"] for row in rows]},
        errors={"mean cost": [row["cost_ci"] for row in rows]},
        title="Offloading policy comparison (lower is better)",
        ylabel="Mean weighted cost",
        path=output_path("study1_offloading_cost.png"),
    )

    # Sweep over number of servers.
    server_counts = [2, 3, 4, 6, 8]
    sweep = parameter_sweep(trial, "num_servers", server_counts, seeds=SEEDS)
    line_path = line_chart(
        x=server_counts,
        series={name: [row[f"{name}_cost_mean"] for row in sweep] for name in POLICIES},
        errors={name: [row[f"{name}_cost_ci"] for row in sweep] for name in POLICIES},
        title="Offloading cost vs. number of edge servers",
        xlabel="Number of edge servers",
        ylabel="Mean weighted cost",
        path=output_path("study1_offloading_sweep.png"),
    )
    results_to_csv(sweep, output_path("study1_offloading_sweep.csv"))

    print(f"\nWrote: {csv_path}\n       {bar_path}\n       {line_path}")


if __name__ == "__main__":
    main()
