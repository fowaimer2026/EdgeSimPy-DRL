"""Study 2b (real data) - DRL agents on the real EUA (Melbourne) infrastructure.

The real-data analogue of ``study2_drl_vs_heuristics.py``: the three DRL agents (DQN, PPO,
A2C) are trained on the offloading environment built from **real EUA base-station
coordinates**, then compared to the min-cost greedy and random baselines across seeds with
95% CIs.

To keep the action space trainable, each seed subsamples a fixed number of real base stations
(the full 125-server action space is hard for DQN with limited training). Coordinates and
heterogeneous pricing are real/realistic; service demands are sampled per seed.

Requires PyTorch. Point --eua-servers / --eua-users at the real files (defaults assume
dataset/eua-dataset/).
"""

# EdgeSimPy
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.service import Service
from edge_sim_py.components.power_models.servers.linear_server_power_model import LinearServerPowerModel
from edge_sim_py.datasets.eua import _read_coordinates
from edge_sim_py.datasets.profiles import profile_for_index
from edge_sim_py.drl import OffloadingEnv, DQNAgent, PPOAgent, A2CAgent, evaluate_agent, evaluate_greedy, evaluate_random
from edge_sim_py.experiments import confidence_interval, results_to_csv

# Python libraries
import argparse
import random

import numpy as np

from _common import output_path, grouped_bar_chart, line_chart

COST_PARAMS = {"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}}
NUM_SERVERS = 20  # subsampled real base stations -> trainable action space
NUM_USERS = 30
EPISODES = 250
SEEDS = range(12)
AGENTS = {"DQN": DQNAgent, "PPO": PPOAgent, "A2C": A2CAgent}

_CACHE = {}
_PATHS = {}


def _coords():
    if "servers" not in _CACHE:
        _CACHE["servers"] = _read_coordinates(_PATHS["servers"], "latitude", "longitude")
        _CACHE["users"] = _read_coordinates(_PATHS["users"], "latitude", "longitude")
    return _CACHE["servers"], _CACHE["users"]


def make_env(seed):
    """Builds an offloading environment from a real-EUA subsample for this seed."""
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

    return OffloadingEnv(services, servers, cost_parameters=COST_PARAMS)


def main():
    parser = argparse.ArgumentParser(description="DRL agents on the real EUA dataset.")
    parser.add_argument("--eua-servers", default="dataset/eua-dataset/edge-servers/site-optus-melbCBD.csv")
    parser.add_argument("--eua-users", default="dataset/eua-dataset/users/users-melbcbd-generated.csv")
    args = parser.parse_args()
    _PATHS["servers"], _PATHS["users"] = args.eua_servers, args.eua_users

    server_coords, _ = _coords()
    print(
        f"Study 2b (REAL EUA DRL) - {len(server_coords)} base stations, {NUM_SERVERS} servers/{NUM_USERS} users, "
        f"{EPISODES} episodes, {len(list(SEEDS))} seeds\n"
    )

    histories = {name: [] for name in AGENTS}
    final_cost = {name: [] for name in AGENTS}
    greedy_cost, random_cost = [], []

    for seed in SEEDS:
        for name, agent_class in AGENTS.items():
            env = make_env(seed)
            agent = agent_class(env.state_dim, env.num_actions, seed=seed)
            histories[name].append(agent.train(env, num_episodes=EPISODES))
            final_cost[name].append(evaluate_agent(agent, env, num_episodes=20)["mean_total_cost"])
        env = make_env(seed)
        greedy_cost.append(evaluate_greedy(env, num_episodes=20)["mean_total_cost"])
        random_cost.append(evaluate_random(env, num_episodes=20)["mean_total_cost"])
        print(f"  seed {seed} done")

    # Learning curves (mean reward per episode across seeds, smoothed).
    def smooth(curve, window=10):
        return np.convolve(curve, np.ones(window) / window, mode="valid")

    curves = {name: np.mean([smooth(np.array(c)) for c in histories[name]], axis=0) for name in AGENTS}
    x = list(range(len(next(iter(curves.values())))))
    curve_path = line_chart(
        x=x,
        series=curves,
        title="DRL learning curves on real EUA infrastructure (10-ep smoothed)",
        xlabel="Training episode",
        ylabel="Episode reward (= -cost)",
        path=output_path("study2_eua_learning.png"),
    )

    # Final-cost comparison.
    print(f"\n{'Policy':<10}{'Mean cost':>12}{'95% CI':>10}")
    rows, values, errs = [], [], []
    labels = list(AGENTS.keys()) + ["Greedy", "Random"]
    datasets = [final_cost[n] for n in AGENTS] + [greedy_cost, random_cost]
    for label, data in zip(labels, datasets):
        ci = confidence_interval(data)
        print(f"{label:<10}{ci['mean']:>12.4f}{ci['half_width']:>10.4f}")
        rows.append({"policy": label, "mean_cost": ci["mean"], "cost_ci": ci["half_width"]})
        values.append(ci["mean"])
        errs.append(ci["half_width"])

    csv_path = results_to_csv(rows, output_path("study2_eua_real.csv"))
    bar_path = grouped_bar_chart(
        categories=labels,
        series={"mean cost": values},
        errors={"mean cost": errs},
        title="DRL agents vs. heuristics on real EUA infrastructure (lower is better)",
        ylabel="Mean weighted cost",
        path=output_path("study2_eua_real.png"),
    )
    print(f"\nWrote: {csv_path}\n       {curve_path}\n       {bar_path}")


if __name__ == "__main__":
    main()
