"""Study 5 - Cost-model weight sensitivity (real EUA infrastructure).

Addresses the reviewer request for an actual weight-sensitivity analysis rather than merely
disclosing the single weight vector used. Runs the cost-driven placement policy on the real
EUA (Melbourne) infrastructure under several latency/energy/monetary weight vectors and
measures the achieved per-service objectives (decision-time absolute costs), across seeds
with 95% CIs.

If the cost model behaves correctly, tilting the weights toward an objective should reduce
that objective in the resulting allocation (e.g. energy-heavy weights -> lowest energy).
"""

# EdgeSimPy
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.service import Service
from edge_sim_py.components.power_models.servers.linear_server_power_model import LinearServerPowerModel
from edge_sim_py.components.cost_models.weighted_cost_model import WeightedCostModel
from edge_sim_py.datasets.eua import _read_coordinates
from edge_sim_py.datasets.profiles import profile_for_index
from edge_sim_py.drl.offloading_env import OffloadingEnv
from edge_sim_py.experiments import run_experiment, results_to_csv

# Python libraries
import argparse
import random

from _common import output_path, grouped_bar_chart

NUM_SERVERS = 125
NUM_USERS = 150
SEEDS = range(20)

# Weight vectors to sweep: (latency, energy, monetary).
WEIGHT_VECTORS = {
    "balanced": {"latency": 0.5, "energy": 0.3, "monetary": 0.2},
    "equal": {"latency": 1 / 3, "energy": 1 / 3, "monetary": 1 / 3},
    "latency-heavy": {"latency": 0.8, "energy": 0.1, "monetary": 0.1},
    "energy-heavy": {"latency": 0.1, "energy": 0.8, "monetary": 0.1},
    "money-heavy": {"latency": 0.1, "energy": 0.1, "monetary": 0.8},
}
OBJECTIVES = ("latency", "energy", "monetary")

_CACHE = {}
_PATHS = {}


def _coords():
    if "servers" not in _CACHE:
        _CACHE["servers"] = _read_coordinates(_PATHS["servers"], "latitude", "longitude")
        _CACHE["users"] = _read_coordinates(_PATHS["users"], "latitude", "longitude")
    return _CACHE["servers"], _CACHE["users"]


def _build_scenario(seed):
    server_coords, user_coords = _coords()
    for cls in (EdgeServer, Service):
        cls._instances = []
        cls._object_count = 0
    rng = random.Random(seed)
    servers = []
    for index, coord in enumerate(rng.sample(server_coords, min(NUM_SERVERS, len(server_coords)))):
        profile = profile_for_index(index)
        server = EdgeServer(model_name=profile["model_name"], cpu=profile["cpu"], memory=profile["memory"], disk=profile["disk"])
        server.coordinates = coord
        server.power_model = LinearServerPowerModel
        server.power_model_parameters = {"max_power_consumption": profile["max_power"], "static_power_percentage": profile["static"]}
        server.cost_model_parameters = {"price_per_cpu_unit_time": rng.choice([0.02, 0.05, 0.2])}
        servers.append(server)
    services = []
    for coord in rng.sample(user_coords, min(NUM_USERS, len(user_coords))):
        service = Service(cpu_demand=rng.randint(1, 4), memory_demand=rng.randint(256, 2048), state=rng.randint(1, 100))
        service.coordinates = coord
        services.append(service)
    return servers, services


def _cost_driven_absolute(env):
    """Cost-driven placement; returns mean decision-time raw objectives of the allocation."""
    env.reset()
    totals = {obj: 0.0 for obj in OBJECTIVES}
    placed = 0
    done = False
    while not done:
        action = env.greedy_action()
        service, server = env.services[env.pointer], env.servers[action]
        if env._has_capacity(server, service):
            objectives = WeightedCostModel.get_objectives(service, server, None, env.cost_parameters)
            for obj in OBJECTIVES:
                totals[obj] += objectives[obj]
            placed += 1
        _, _, done, _ = env.step(action)
    n = max(placed, 1)
    return {obj: totals[obj] / n for obj in OBJECTIVES}


def trial(seed):
    servers, services = _build_scenario(seed)
    result = {}
    for name, weights in WEIGHT_VECTORS.items():
        env = OffloadingEnv(services, servers, cost_parameters={"weights": weights})
        metrics = _cost_driven_absolute(env)
        for obj in OBJECTIVES:
            result[f"{name}_{obj}"] = metrics[obj]
    return result


def main():
    parser = argparse.ArgumentParser(description="Cost-model weight sensitivity on real EUA.")
    parser.add_argument("--eua-servers", default="dataset/eua-dataset/edge-servers/site-optus-melbCBD.csv")
    parser.add_argument("--eua-users", default="dataset/eua-dataset/users/users-melbcbd-generated.csv")
    args = parser.parse_args()
    _PATHS["servers"], _PATHS["users"] = args.eua_servers, args.eua_users

    print(f"Study 5 - Cost-model weight sensitivity (real EUA, {NUM_SERVERS} servers/{NUM_USERS} users, {len(list(SEEDS))} seeds, 95% CI)\n")
    summary = run_experiment(trial, seeds=SEEDS)["summary"]

    print(f"{'Weights':<15}{'Latency':>12}{'Energy':>12}{'Monetary':>12}")
    rows = []
    series = {obj: [] for obj in OBJECTIVES}
    errors = {obj: [] for obj in OBJECTIVES}
    for name in WEIGHT_VECTORS:
        vals = {obj: summary[f"{name}_{obj}"] for obj in OBJECTIVES}
        print(f"{name:<15}{vals['latency']['mean']:>12.3f}{vals['energy']['mean']:>12.2f}{vals['monetary']['mean']:>12.3f}")
        row = {"weights": name}
        for obj in OBJECTIVES:
            row[f"mean_{obj}"] = vals[obj]["mean"]
            row[f"{obj}_ci"] = vals[obj]["half_width"]
            series[obj].append(vals[obj]["mean"])
            errors[obj].append(vals[obj]["half_width"])
        rows.append(row)

    csv_path = results_to_csv(rows, output_path("study5_weight_sensitivity.csv"))
    # Normalize each objective to its max across weight vectors so the shifts are visible on one axis.
    norm_series, norm_errors = {}, {}
    for obj in OBJECTIVES:
        peak = max(series[obj]) or 1.0
        norm_series[obj] = [v / peak for v in series[obj]]
        norm_errors[obj] = [e / peak for e in errors[obj]]
    fig_path = grouped_bar_chart(
        categories=list(WEIGHT_VECTORS.keys()),
        series=norm_series,
        errors=norm_errors,
        title="Cost-model weight sensitivity on real EUA (objectives normalized per column)",
        ylabel="achieved objective (normalized)",
        path=output_path("study5_weight_sensitivity.png"),
    )
    print(f"\nWrote: {csv_path}\n       {fig_path}")


if __name__ == "__main__":
    main()
