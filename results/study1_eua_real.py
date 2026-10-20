"""Study 1 (real data) - Offloading policies on the real EUA (Melbourne) infrastructure.

Repeats the offloading-policy comparison of ``study1_offloading_policies.py`` but on the
**real EUA dataset** (125 Melbourne CBD base stations). To keep the infrastructure at a
meaningful load (the full 816-user trace is heavily over-subscribed on these servers), each
seed subsamples a fixed number of users and samples their resource demands. Reports
cost-driven vs. latency-greedy vs. random with 95% CIs; emits a CSV and a figure.

Point --eua-servers / --eua-users at the real files (defaults assume dataset/eua-dataset/).
"""

# EdgeSimPy
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.service import Service
from edge_sim_py.components.cost_models.latency_cost_model import LatencyCostModel
from edge_sim_py.components.cost_models.weighted_cost_model import WeightedCostModel
from edge_sim_py.datasets.eua import load_eua_servers, _read_coordinates
from edge_sim_py.drl.offloading_env import OffloadingEnv
from edge_sim_py.experiments import run_experiment, results_to_csv

# Python libraries
import argparse
import random

import numpy as np

from _common import output_path, grouped_bar_chart

COST_PARAMS = {"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}}
NUM_USERS = 150  # subsample so the 125-server CBD infrastructure is loaded but not saturated
SEEDS = range(20)

# Paths are filled in by main() from CLI args so the trial closure can see them.
_PATHS = {}


def _cost_driven(env):
    return env.greedy_action()


def _random(env):
    return env.random_action()


def _latency_greedy(env):
    service = env.services[env.pointer]
    costs = [LatencyCostModel.get_cost(service, s, None, COST_PARAMS) for s in env.servers]
    masked = [c if valid else np.inf for c, valid in zip(costs, env.action_mask)]
    return int(np.argmin(masked))


POLICIES = {"cost_driven": _cost_driven, "latency_greedy": _latency_greedy, "random": _random}


def evaluate_absolute(env, action_fn):
    """Runs a policy and measures ABSOLUTE per-objective cost at *decision time*.

    The environment's reward is a per-step *relative* score (fine for training, not additive
    across load trajectories). We instead record each service's raw latency / energy /
    monetary cost on its chosen server at the moment it is placed -- i.e. while that server
    still has free capacity, so the processing term is well-defined (evaluating it post-hoc,
    once the server is saturated, would divide by ~0 and explode). Reports the mean per placed
    service plus placement rate.
    """
    env.reset()
    totals = {"latency": 0.0, "energy": 0.0, "monetary": 0.0}
    placed = 0
    done, info = False, {"placed": 0}
    while not done:
        action = action_fn(env)
        service = env.services[env.pointer]
        server = env.servers[action]
        if env._has_capacity(server, service):
            objectives = WeightedCostModel.get_objectives(service, server, None, COST_PARAMS)
            for key in totals:
                totals[key] += objectives[key]
            placed += 1
        _, _, done, info = env.step(action)

    n = max(placed, 1)
    return {**{key: value / n for key, value in totals.items()}, "placement": info["placed"] / len(env.services)}


def trial(seed):
    # Fresh registries each seed so repeated loads don't accumulate.
    for cls in (EdgeServer, Service):
        cls._instances = []
        cls._object_count = 0

    servers = load_eua_servers(_PATHS["servers"])
    rng = random.Random(seed)
    for server in servers:
        server.cost_model_parameters = {"price_per_cpu_unit_time": rng.choice([0.02, 0.05, 0.2])}

    # Subsample users and build Service objects with sampled demands.
    coordinates = _read_coordinates(_PATHS["users"], "latitude", "longitude")
    chosen = rng.sample(coordinates, min(NUM_USERS, len(coordinates)))
    services = []
    for lat, lon in chosen:
        service = Service(cpu_demand=rng.randint(1, 4), memory_demand=rng.randint(256, 2048), state=rng.randint(1, 100))
        service.coordinates = (lat, lon)
        services.append(service)

    env = OffloadingEnv(services, servers, cost_parameters=COST_PARAMS)
    result = {}
    for name, action_fn in POLICIES.items():
        metrics = evaluate_absolute(env, action_fn)
        for key in ("latency", "energy", "monetary", "placement"):
            result[f"{name}_{key}"] = metrics[key]
    return result


def main():
    parser = argparse.ArgumentParser(description="Offloading policies on the real EUA dataset.")
    parser.add_argument("--eua-servers", default="dataset/eua-dataset/edge-servers/site-optus-melbCBD.csv")
    parser.add_argument("--eua-users", default="dataset/eua-dataset/users/users-melbcbd-generated.csv")
    args = parser.parse_args()
    _PATHS["servers"], _PATHS["users"] = args.eua_servers, args.eua_users

    servers = load_eua_servers(_PATHS["servers"])
    print(f"Study 1 (REAL EUA) - {len(servers)} base stations, {NUM_USERS} users/seed, {len(list(SEEDS))} seeds, 95% CI\n")

    summary = run_experiment(trial, seeds=SEEDS)["summary"]
    print(f"{'Policy':<16}{'Latency':>12}{'Energy':>12}{'Monetary':>12}{'Placement':>12}")
    rows = []
    series = {"latency": [], "energy_/10": [], "monetary": []}
    errors = {"latency": [], "energy_/10": [], "monetary": []}
    for name in POLICIES:
        lat, ene, mon, plc = (summary[f"{name}_{k}"] for k in ("latency", "energy", "monetary", "placement"))
        print(f"{name:<16}{lat['mean']:>12.3f}{ene['mean']:>12.2f}{mon['mean']:>12.3f}{plc['mean']:>12.3f}")
        rows.append(
            {
                "policy": name,
                "mean_latency": lat["mean"],
                "latency_ci": lat["half_width"],
                "mean_energy": ene["mean"],
                "energy_ci": ene["half_width"],
                "mean_monetary": mon["mean"],
                "monetary_ci": mon["half_width"],
                "mean_placement": plc["mean"],
            }
        )
        series["latency"].append(lat["mean"])
        errors["latency"].append(lat["half_width"])
        series["energy_/10"].append(ene["mean"] / 10)
        errors["energy_/10"].append(ene["half_width"] / 10)
        series["monetary"].append(mon["mean"])
        errors["monetary"].append(mon["half_width"])

    csv_path = results_to_csv(rows, output_path("study1_eua_real.csv"))
    fig_path = grouped_bar_chart(
        categories=list(POLICIES.keys()),
        series=series,
        errors=errors,
        title="Offloading policies on real EUA infrastructure (per-service cost, lower is better)",
        ylabel="mean cost per service",
        path=output_path("study1_eua_real.png"),
    )
    print(f"\nWrote: {csv_path}\n       {fig_path}")


if __name__ == "__main__":
    main()
