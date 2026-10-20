""" Study 8 - Integrated end-to-end experiment (placement + routing + execution + energy + failures).

Addresses the reviewer's single biggest concern: the components must be shown working *together*
in one run, not just evaluated separately. This study wires them into a single pipeline on real
data:

    * Infrastructure -- real EUA base stations become edge servers sitting on a geographic mesh
      (real-distance link delays), so routing has a real topology to route over.
    * Workload -- real Alibaba tasks arrive over time, each from a random source base station.
    * Placement -- a policy chooses which edge server runs each task. The cost-driven policy's
      latency term *includes the routing path delay*, so placement and routing are coupled.
    * Routing -- the network delay from the task's source to its chosen server is computed by a
      routing policy under link congestion, and applied as transit time before execution.
    * Execution -- the task then queues and runs on the server (execution engine).
    * Energy + failures -- energy is metered throughout, and servers fail (MTBF/MTTR), interrupting
      and re-queuing tasks.

End-to-end response time therefore spans routing transit + queue wait + compute, and every layer
influences it. Two full stacks are compared across seeds with 95% CIs:

    baseline    = random placement      + shortest-path routing
    integrated  = cost-driven placement + latency-aware routing

Point --eua-servers / --alibaba at the real files (defaults assume dataset/).
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
from edge_sim_py.workload import Task, ExecutionEngine, collect_workload_metrics
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

NUM_NODES = 25
NUM_TASKS = 600
K_NEAREST = 4
DELAY_SCALE = 2000.0
COST_PARAMS = {"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}}
SEEDS = range(6)
MAX_STEPS = 40_000
MTBF, MTTR = 600.0, 30.0

STACKS = {
    "baseline (random + shortest-path)": {"placement": "random", "routing": ShortestPathRouting, "params": None},
    "integrated (cost-driven + latency-aware)": {"placement": "cost", "routing": LatencyAwareRouting, "params": {"alpha": 0.5, "beta": 0.5}},
}
_CACHE = {}
_PATHS = {}


def _coords():
    if "servers" not in _CACHE:
        _CACHE["servers"] = _read_coordinates(_PATHS["servers"], "latitude", "longitude")
    return _CACHE["servers"]


def _build_infrastructure(seed):
    """Builds edge servers on a geographic mesh of real base stations. Returns (servers, topology, switch_of_server)."""
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

    switch_of_server = {id(s): s.network_switch for s in servers}
    return servers, topology, switches, switch_of_server


def _routed_delay(topology, routing, params, source, target, cache):
    """Memoized network delay of the routing policy's path from source to target switch."""
    key = (id(source), id(target))
    if key not in cache:
        path = routing.get_path(topology, source, target, parameters=params)
        if len(path) < 1:
            cache[key] = float("inf")
        else:
            cache[key] = sum(topology[path[i]][path[i + 1]]["delay"] for i in range(len(path) - 1))
    return cache[key]


def _place_and_route(tasks, servers, topology, switches, routing, params, placement, rng):
    """Assigns each task a server + network transit time; returns total network delay applied."""
    delay_cache = {}
    total_network_delay = 0
    for task in tasks:
        source = task.source_switch
        capable = [s for s in servers if (s.cpu - s.cpu_demand) >= task.cpu_demand]
        if not capable:
            capable = servers  # will be a no-capacity placement; engine leaves it queued

        if placement == "random":
            server = rng.choice(capable)
        else:  # cost-driven: latency term includes the routing path delay -> couples placement & routing
            best, best_cost = None, float("inf")
            for candidate in capable:
                net = _routed_delay(topology, routing, params, source, candidate.network_switch, delay_cache)
                objectives = WeightedCostModel.get_objectives(task, candidate, None, COST_PARAMS)
                weights = WeightedCostModel._normalized_weights({**WeightedCostModel.DEFAULT_PARAMETERS, **COST_PARAMS})
                cost = weights["latency"] * (objectives["latency"] + net) + weights["energy"] * objectives["energy"] + weights["monetary"] * objectives["monetary"]
                if cost < best_cost:
                    best, best_cost = candidate, cost
            server = best

        net_delay = _routed_delay(topology, routing, params, source, server.network_switch, delay_cache)
        net_delay = 0 if net_delay == float("inf") else int(net_delay)
        task.server = server
        server.cpu_demand += task.cpu_demand  # reservation view for subsequent cost-driven decisions
        task.arrival_step = task.original_arrival + net_delay  # routing transit before execution
        total_network_delay += net_delay

    for s in servers:  # clear the reservation view; the engine re-derives demand from running tasks
        s.cpu_demand = 0
    return total_network_delay / max(len(tasks), 1)


def trial(seed, stack_name="baseline (random + shortest-path)"):
    stack = STACKS[stack_name]
    servers, topology, switches, _ = _build_infrastructure(seed)

    tasks = load_alibaba_tasks(_PATHS["alibaba"], sort_by_arrival=True, limit=NUM_TASKS, window_offset=seed * NUM_TASKS, scan_limit=150_000)
    rng = random.Random(seed)
    for task in tasks:
        task.original_arrival = task.arrival_step
        task.source_switch = rng.choice(switches)

    avg_net = _place_and_route(tasks, servers, topology, switches, stack["routing"], stack["params"], stack["placement"], rng)

    meter = EnergyMeter()
    engine = ExecutionEngine(servers, tasks, energy_meter=meter, failure_model=FailureModel(mtbf=MTBF, mttr=MTTR, seed=seed))
    engine.run(max_steps=MAX_STEPS)

    # End-to-end response = completion step - ORIGINAL arrival (spans routing transit + queue + compute).
    finished = [t for t in tasks if t.finished]
    e2e = [t.end_step - t.original_arrival + 1 for t in finished]
    deadline_ok = [t for t in finished if t.deadline is not None and t.end_step <= t.deadline]
    with_dl = [t for t in finished if t.deadline is not None]
    metrics = collect_workload_metrics(tasks, engine)
    return {
        # Mean response is dominated by failure-restart tails and cross-window workload variance;
        # the median is the robust "typical task" statistic and is reported alongside it.
        "e2e_response_mean": float(np.mean(e2e)) if e2e else 0.0,
        "e2e_response_median": float(np.median(e2e)) if e2e else 0.0,
        "network_delay": avg_net,
        "deadline_violation": 1.0 - len(deadline_ok) / len(with_dl) if with_dl else 0.0,
        "completion_rate": metrics["completion_rate"],
        "total_energy": metrics.get("total_energy", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Integrated end-to-end experiment on real data.")
    parser.add_argument("--eua-servers", default="dataset/eua-dataset/edge-servers/site-optus-melbCBD.csv")
    parser.add_argument("--alibaba", default="dataset/alibaba-clusterdata/batch_task.csv")
    args = parser.parse_args()
    _PATHS["servers"], _PATHS["alibaba"] = args.eua_servers, args.alibaba

    print(f"Study 8 - Integrated end-to-end ({NUM_NODES}-node mesh, {NUM_TASKS} real tasks, MTBF={MTBF}, {len(list(SEEDS))} seeds, 95% CI)\n")

    metrics = ("e2e_response_mean", "e2e_response_median", "network_delay", "deadline_violation", "completion_rate", "total_energy")
    rows = []
    summaries = {}
    for stack_name in STACKS:
        summary = run_experiment(trial, seeds=SEEDS, stack_name=stack_name)["summary"]
        summaries[stack_name] = summary
        print(f"{stack_name}")
        print(f"  e2e response: median {summary['e2e_response_median']['mean']:.1f} +/-{summary['e2e_response_median']['half_width']:.1f}"
              f"   mean {summary['e2e_response_mean']['mean']:.1f} +/-{summary['e2e_response_mean']['half_width']:.1f}"
              f"   (network transit {summary['network_delay']['mean']:.1f})")
        print(f"  deadline violations: {summary['deadline_violation']['mean']:.3f}   completion: {summary['completion_rate']['mean']:.3f}"
              f"   energy: {summary['total_energy']['mean']:.0f}\n")
        row = {"stack": stack_name}
        for m in metrics:
            row[f"{m}_mean"] = summary[m]["mean"]
            row[f"{m}_ci"] = summary[m]["half_width"]
        rows.append(row)

    results_to_csv(rows, output_path("study8_integrated.csv"))
    names = list(STACKS.keys())
    fig_path = grouped_bar_chart(
        categories=["baseline", "integrated"],
        series={"median response": [summaries[n]["e2e_response_median"]["mean"] for n in names],
                "network delay": [summaries[n]["network_delay"]["mean"] for n in names],
                "deadline viol. %": [summaries[n]["deadline_violation"]["mean"] * 100 for n in names]},
        errors={"median response": [summaries[n]["e2e_response_median"]["half_width"] for n in names],
                "network delay": [summaries[n]["network_delay"]["half_width"] for n in names],
                "deadline viol. %": [summaries[n]["deadline_violation"]["half_width"] * 100 for n in names]},
        title="Integrated stack vs. baseline (end-to-end, real EUA + Alibaba)",
        ylabel="steps  /  violation %",
        path=output_path("study8_integrated.png"),
    )

    base, integ = summaries[names[0]], summaries[names[1]]

    def pct(metric):
        b = base[metric]["mean"]
        return 100 * (1 - integ[metric]["mean"] / b) if b else 0.0

    print(f"Integrated vs baseline: median response {pct('e2e_response_median'):+.0f}%, "
          f"network delay {pct('network_delay'):+.0f}%, deadline violations {pct('deadline_violation'):+.0f}%, "
          f"energy {pct('total_energy'):+.0f}% (positive = improvement).")
    print(f"Wrote CSV + {fig_path}")


if __name__ == "__main__":
    main()
