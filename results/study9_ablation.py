"""Study 9 - Ablation study (Part 3, item 1).

Isolates each module's contribution by toggling placement and routing intelligence on/off in
the integrated end-to-end pipeline (real EUA mesh + real Alibaba tasks + execution + energy +
failures). Six configurations span the requested ablation cells:

    placement in {random, cost-greedy, DRL} x routing in {shortest-path, latency-aware}

    | config           | placement   | routing        | modules enabled            |
    | baseline         | random      | shortest-path  | none                       |
    | cost only        | cost-greedy | shortest-path  | cost model                 |
    | routing only     | random      | latency-aware  | routing                    |
    | DRL only         | DRL         | shortest-path  | DRL (cost as reward)       |
    | cost + routing   | cost-greedy | latency-aware  | cost + routing             |
    | full (DRL+route) | DRL         | latency-aware  | cost + routing + DRL       |

For each config we report end-to-end median response, network delay, deadline-violation rate
and energy, plus each module's marginal contribution. DRL placement trains a DQN per seed on
the config's server set and uses its greedy actions to place the real task stream.

Requires PyTorch. Point --eua-servers / --alibaba at the real files (defaults assume dataset/).
"""

# EdgeSimPy
from edge_sim_py.components.topology import Topology
from edge_sim_py.components.network_switch import NetworkSwitch
from edge_sim_py.components.network_link import NetworkLink
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.power_models.servers.linear_server_power_model import LinearServerPowerModel
from edge_sim_py.components.cost_models.weighted_cost_model import WeightedCostModel
from edge_sim_py.components.routing import ShortestPathRouting, LatencyAwareRouting
from edge_sim_py.datasets.eua import _read_coordinates
from edge_sim_py.datasets.alibaba import load_alibaba_tasks
from edge_sim_py.datasets.profiles import profile_for_index
from edge_sim_py.drl import OffloadingEnv, DQNAgent
from edge_sim_py.workload import ExecutionEngine, collect_workload_metrics
from edge_sim_py.workload.scenario import reset_workload_registries
from edge_sim_py.energy import EnergyMeter
from edge_sim_py.reliability import FailureModel
from edge_sim_py.experiments import run_experiment, results_to_csv

# Python libraries
import argparse
import math
import random

import numpy as np

from _common import output_path, grouped_bar_chart

NUM_NODES = 20
NUM_TASKS = 300
K_NEAREST = 4
DELAY_SCALE = 2000.0
DRL_TRAIN_SERVICES = 60
DRL_EPISODES = 120
COST_PARAMS = {"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}}
SEEDS = range(10)
MAX_STEPS = 40_000
MTBF, MTTR = 600.0, 30.0

CONFIGS = {
    "baseline": ("random", ShortestPathRouting, None),
    "cost_only": ("cost", ShortestPathRouting, None),
    "routing_only": ("random", LatencyAwareRouting, {"alpha": 0.5, "beta": 0.5}),
    "drl_only": ("drl", ShortestPathRouting, None),
    "cost+routing": ("cost", LatencyAwareRouting, {"alpha": 0.5, "beta": 0.5}),
    "full": ("drl", LatencyAwareRouting, {"alpha": 0.5, "beta": 0.5}),
}
_CACHE = {}
_PATHS = {}


def _coords():
    if "servers" not in _CACHE:
        _CACHE["servers"] = _read_coordinates(_PATHS["servers"], "latitude", "longitude")
    return _CACHE["servers"]


def _build_infrastructure(seed):
    import networkx as nx

    for cls in (Topology, NetworkSwitch, NetworkLink, EdgeServer):
        cls._instances = []
        cls._object_count = 0
    reset_workload_registries()

    rng = random.Random(seed)
    coords = rng.sample(_coords(), min(NUM_NODES, len(_coords())))
    n = len(coords)
    topology = Topology()
    switches = [NetworkSwitch() for _ in range(n)]
    servers = []
    for index, (sw, coord) in enumerate(zip(switches, coords)):
        sw.coordinates = coord
        topology.add_node(sw)
        profile = profile_for_index(index)
        server = EdgeServer(model_name=profile["model_name"], cpu=profile["cpu"], memory=profile["memory"], disk=profile["disk"])
        server.coordinates = coord
        server.network_switch = sw
        server.power_model = LinearServerPowerModel
        server.power_model_parameters = {"max_power_consumption": profile["max_power"], "static_power_percentage": profile["static"]}
        server.cost_model_parameters = {"price_per_cpu_unit_time": rng.choice([0.02, 0.05, 0.2])}
        servers.append(server)

    complete = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            complete.add_edge(i, j, weight=math.dist(coords[i], coords[j]))
    edges = set(tuple(sorted(e)) for e in nx.minimum_spanning_tree(complete).edges())
    for i in range(n):
        for j in sorted(range(n), key=lambda k: math.dist(coords[i], coords[k]))[1 : K_NEAREST + 1]:
            edges.add(tuple(sorted((i, j))))
    for i, j in edges:
        link = NetworkLink()
        link["topology"] = topology
        link["nodes"] = [switches[i], switches[j]]
        link["delay"] = max(1, int(math.dist(coords[i], coords[j]) * DELAY_SCALE))
        link["bandwidth"] = 100
        link["bandwidth_demand"] = rng.uniform(0.0, 0.9) * 100
        topology.add_edge(switches[i], switches[j], **link)
    return servers, topology, switches


def _routed_delay(topology, routing, params, source, target, cache):
    key = (id(source), id(target))
    if key not in cache:
        path = routing.get_path(topology, source, target, parameters=params)
        cache[key] = sum(topology[path[i]][path[i + 1]]["delay"] for i in range(len(path) - 1)) if len(path) > 1 else (0 if path else float("inf"))
    return cache[key]


def _train_placement_agent(servers, seed):
    """Trains a DQN placement agent on a contended OffloadingEnv over the same servers."""
    from edge_sim_py.components.service import Service

    rng = random.Random(seed + 999)
    services = [Service(cpu_demand=rng.randint(1, 4), memory_demand=rng.randint(256, 2048), state=rng.randint(1, 100)) for _ in range(DRL_TRAIN_SERVICES)]
    env = OffloadingEnv(services, servers, cost_parameters=COST_PARAMS)
    agent = DQNAgent(env.state_dim, env.num_actions, seed=seed)
    agent.train(env, num_episodes=DRL_EPISODES)
    # Clear the training reservation so the stream placement starts clean.
    for s in servers:
        s.cpu_demand = 0
        s.memory_demand = 0
    return agent, env


def _place(tasks, servers, topology, routing, params, placement, rng, agent=None, encoder_env=None):
    """Assigns each task a server (+ network transit); returns mean network delay."""
    weights = WeightedCostModel._normalized_weights({**WeightedCostModel.DEFAULT_PARAMETERS, **COST_PARAMS})
    cache = {}
    total_net = 0
    for task in tasks:
        source = task.source_switch
        capable = [s for s in servers if (s.cpu - s.cpu_demand) >= task.cpu_demand] or servers

        if placement == "random":
            server = rng.choice(capable)
        elif placement == "cost":
            best, best_cost = None, float("inf")
            for cand in capable:
                net = _routed_delay(topology, routing, params, source, cand.network_switch, cache)
                obj = WeightedCostModel.get_objectives(task, cand, None, COST_PARAMS)
                c = weights["latency"] * (obj["latency"] + net) + weights["energy"] * obj["energy"] + weights["monetary"] * obj["monetary"]
                if c < best_cost:
                    best, best_cost = cand, c
            server = best
        else:  # drl: encode the task against current server load and use the trained agent
            state = encoder_env._encode(task)
            mask = encoder_env.action_mask
            action = agent.act(state, mask, greedy=True)
            server = servers[action]

        net_delay = _routed_delay(topology, routing, params, source, server.network_switch, cache)
        net_delay = 0 if net_delay == float("inf") else int(net_delay)
        task.server = server
        server.cpu_demand += task.cpu_demand
        task.arrival_step = task.original_arrival + net_delay
        total_net += net_delay

    for s in servers:
        s.cpu_demand = 0
    return total_net / max(len(tasks), 1)


def trial(seed, config_name="baseline"):
    placement, routing, params = CONFIGS[config_name]
    servers, topology, switches = _build_infrastructure(seed)

    agent, encoder_env = (None, None)
    if placement == "drl":
        agent, encoder_env = _train_placement_agent(servers, seed)

    tasks = load_alibaba_tasks(_PATHS["alibaba"], sort_by_arrival=True, limit=NUM_TASKS, window_offset=seed * NUM_TASKS, scan_limit=150_000)
    rng = random.Random(seed)
    for task in tasks:
        task.original_arrival = task.arrival_step
        task.source_switch = rng.choice(switches)

    avg_net = _place(tasks, servers, topology, routing, params, placement, rng, agent, encoder_env)

    meter = EnergyMeter()
    engine = ExecutionEngine(servers, tasks, energy_meter=meter, failure_model=FailureModel(mtbf=MTBF, mttr=MTTR, seed=seed))
    engine.run(max_steps=MAX_STEPS)

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
        "completion_rate": m["completion_rate"],
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation study on real EUA + Alibaba.")
    parser.add_argument("--eua-servers", default="dataset/eua-dataset/edge-servers/site-optus-melbCBD.csv")
    parser.add_argument("--alibaba", default="dataset/alibaba-clusterdata/batch_task.csv")
    args = parser.parse_args()
    _PATHS["servers"], _PATHS["alibaba"] = args.eua_servers, args.alibaba

    print(f"Study 9 - Ablation ({NUM_NODES}-node mesh, {NUM_TASKS} real tasks, {len(list(SEEDS))} seeds, 95% CI)\n")
    metrics = ("e2e_median", "network_delay", "deadline_violation", "total_energy", "completion_rate")
    rows, summaries = [], {}
    print(f"{'Config':<15}{'E2E median':>12}{'Network':>10}{'DL viol.':>10}{'Energy':>12}")
    for name in CONFIGS:
        summary = run_experiment(trial, seeds=SEEDS, config_name=name)["summary"]
        summaries[name] = summary
        print(
            f"{name:<15}{summary['e2e_median']['mean']:>12.1f}{summary['network_delay']['mean']:>10.1f}"
            f"{summary['deadline_violation']['mean']:>10.3f}{summary['total_energy']['mean']:>12.0f}"
        )
        row = {"config": name}
        for m in metrics:
            row[f"{m}_mean"] = summary[m]["mean"]
            row[f"{m}_ci"] = summary[m]["half_width"]
        rows.append(row)

    results_to_csv(rows, output_path("study9_ablation.csv"))

    # Per-module contribution vs baseline (% improvement on each metric).
    base = summaries["baseline"]
    print("\nModule contribution vs baseline (positive = improvement):")
    for name in ("cost_only", "routing_only", "drl_only", "cost+routing", "full"):
        s = summaries[name]
        dl = 100 * (1 - s["deadline_violation"]["mean"] / base["deadline_violation"]["mean"]) if base["deadline_violation"]["mean"] else 0
        net = 100 * (1 - s["network_delay"]["mean"] / base["network_delay"]["mean"]) if base["network_delay"]["mean"] else 0
        en = 100 * (1 - s["total_energy"]["mean"] / base["total_energy"]["mean"]) if base["total_energy"]["mean"] else 0
        print(f"  {name:<15} network {net:+.0f}%   deadline {dl:+.0f}%   energy {en:+.0f}%")

    names = list(CONFIGS.keys())
    fig_path = grouped_bar_chart(
        categories=names,
        series={
            "network delay": [summaries[n]["network_delay"]["mean"] for n in names],
            "deadline viol. %": [summaries[n]["deadline_violation"]["mean"] * 100 for n in names],
        },
        errors={
            "network delay": [summaries[n]["network_delay"]["half_width"] for n in names],
            "deadline viol. %": [summaries[n]["deadline_violation"]["half_width"] * 100 for n in names],
        },
        title="Ablation: per-configuration end-to-end metrics (real EUA + Alibaba)",
        ylabel="steps  /  violation %",
        path=output_path("study9_ablation.png"),
    )
    print(f"\nWrote CSV + {fig_path}")


if __name__ == "__main__":
    main()
