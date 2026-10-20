"""Study 1c (real data) - Offloading policies on the real Shanghai Telecom infrastructure.

The direct analogue of ``study1_eua_real.py``, but on a **second, independent real
infrastructure**: the Shanghai Telecom dataset (real base-station coordinates + user access
sessions). This provides real-data cross-validation of the offloading result on a different
city and a much larger base-station set.

The (large) Excel file is read once and its base-station / session coordinates are cached;
each seed then subsamples a fixed number of servers and users, samples demands, and compares
cost-driven vs. latency-greedy vs. random by decision-time absolute per-objective cost.

Point --shanghai at a real .xlsx (default assumes dataset/Shanghai/).
"""

# EdgeSimPy
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.service import Service
from edge_sim_py.components.power_models.servers.linear_server_power_model import LinearServerPowerModel
from edge_sim_py.components.cost_models.latency_cost_model import LatencyCostModel
from edge_sim_py.components.cost_models.weighted_cost_model import WeightedCostModel
from edge_sim_py.datasets import load_shanghai
from edge_sim_py.datasets.profiles import profile_for_index
from edge_sim_py.drl.offloading_env import OffloadingEnv
from edge_sim_py.experiments import run_experiment, results_to_csv

# Python libraries
import argparse
import random

import numpy as np

from _common import output_path, grouped_bar_chart

COST_PARAMS = {"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}}
NUM_SERVERS = 125
NUM_USERS = 150
MAX_SESSIONS = 15_000
SEEDS = range(20)

_CACHE = {}


def _pools(path):
    """Reads the Shanghai file once and caches base-station + session coordinates."""
    if "servers" not in _CACHE:
        servers, services = load_shanghai(path, seed=0, max_sessions=MAX_SESSIONS)
        _CACHE["servers"] = [s.coordinates for s in servers]
        _CACHE["sessions"] = [s.coordinates for s in services]
    return _CACHE["servers"], _CACHE["sessions"]


def _build_server(coord, index, rng):
    profile = profile_for_index(index)
    server = EdgeServer(model_name=profile["model_name"], cpu=profile["cpu"], memory=profile["memory"], disk=profile["disk"])
    server.coordinates = coord
    server.power_model = LinearServerPowerModel
    server.power_model_parameters = {"max_power_consumption": profile["max_power"], "static_power_percentage": profile["static"]}
    server.cost_model_parameters = {"price_per_cpu_unit_time": rng.choice([0.02, 0.05, 0.2])}
    return server


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
    """Mean raw latency/energy/monetary cost measured at decision time (see study1_eua_real)."""
    env.reset()
    totals = {"latency": 0.0, "energy": 0.0, "monetary": 0.0}
    placed = 0
    done, info = False, {"placed": 0}
    while not done:
        action = action_fn(env)
        service, server = env.services[env.pointer], env.servers[action]
        if env._has_capacity(server, service):
            objectives = WeightedCostModel.get_objectives(service, server, None, COST_PARAMS)
            for key in totals:
                totals[key] += objectives[key]
            placed += 1
        _, _, done, info = env.step(action)
    n = max(placed, 1)
    return {**{k: v / n for k, v in totals.items()}, "placement": info["placed"] / len(env.services)}


def trial(seed):
    server_coords, session_coords = _pools(_PATH["shanghai"])
    for cls in (EdgeServer, Service):
        cls._instances = []
        cls._object_count = 0

    rng = random.Random(seed)
    chosen_servers = rng.sample(server_coords, min(NUM_SERVERS, len(server_coords)))
    servers = [_build_server(coord, i, rng) for i, coord in enumerate(chosen_servers)]

    chosen_sessions = rng.sample(session_coords, min(NUM_USERS, len(session_coords)))
    services = []
    for coord in chosen_sessions:
        service = Service(cpu_demand=rng.randint(1, 4), memory_demand=rng.randint(256, 2048), state=rng.randint(1, 100))
        service.coordinates = coord
        services.append(service)

    env = OffloadingEnv(services, servers, cost_parameters=COST_PARAMS)
    result = {}
    for name, action_fn in POLICIES.items():
        metrics = evaluate_absolute(env, action_fn)
        for key in ("latency", "energy", "monetary", "placement"):
            result[f"{name}_{key}"] = metrics[key]
    return result


_PATH = {}


def main():
    parser = argparse.ArgumentParser(description="Offloading policies on the real Shanghai Telecom dataset.")
    parser.add_argument("--shanghai", default="dataset/Shanghai/data_6.16.15.xlsx")
    args = parser.parse_args()
    _PATH["shanghai"] = args.shanghai

    server_coords, _ = _pools(_PATH["shanghai"])
    print(
        f"Study 1c (REAL Shanghai) - {len(server_coords)} base stations available, "
        f"{NUM_SERVERS} servers/{NUM_USERS} users per seed, {len(list(SEEDS))} seeds, 95% CI\n"
    )

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

    csv_path = results_to_csv(rows, output_path("study1_shanghai_real.csv"))
    fig_path = grouped_bar_chart(
        categories=list(POLICIES.keys()),
        series=series,
        errors=errors,
        title="Offloading policies on real Shanghai Telecom infrastructure (per-service cost)",
        ylabel="mean cost per service",
        path=output_path("study1_shanghai_real.png"),
    )
    print(f"\nWrote: {csv_path}\n       {fig_path}")


if __name__ == "__main__":
    main()
