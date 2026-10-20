"""Study 6 - Routing on a larger, geographically-grounded topology (real EUA coordinates).

Addresses the reviewer request for a larger/more realistic routing evaluation than the
synthetic 6-path micro-topology of Study 3. A connected mesh is built over real EUA
(Melbourne) base-station coordinates: link delays are the real geographic distances, and the
graph combines a Euclidean minimum spanning tree (guaranteeing connectivity) with k-nearest-
neighbour edges (giving multiple multi-hop paths). Per seed, link congestion is randomized
and many random source->destination flows are routed under each policy; we report the mean
end-to-end path delay and bottleneck utilization with 95% CIs.

"real coordinates + modeled links/congestion": the datasets carry no link or traffic data,
so edges and load are synthesized, but distances/topology derive from real base stations.
"""

# EdgeSimPy
from edge_sim_py.components.topology import Topology
from edge_sim_py.components.network_switch import NetworkSwitch
from edge_sim_py.components.network_link import NetworkLink
from edge_sim_py.components.routing import ShortestPathRouting, LeastCongestedRouting, LatencyAwareRouting, ECMPRouting
from edge_sim_py.components.routing.base_routing import link_utilization
from edge_sim_py.datasets.eua import _read_coordinates
from edge_sim_py.experiments import run_experiment, results_to_csv

# Python libraries
import argparse
import math
import random

import networkx as nx

from _common import output_path, grouped_bar_chart

NUM_NODES = 80  # subsampled real base stations forming the routing mesh
K_NEAREST = 4
FLOWS_PER_SEED = 200
DELAY_SCALE = 1000.0  # degrees -> integer-ish link delays
SEEDS = range(20)

POLICIES = {
    "shortest_path": (ShortestPathRouting, None),
    "least_congested": (LeastCongestedRouting, None),
    "latency_aware": (LatencyAwareRouting, {"alpha": 0.5, "beta": 0.5}),
    "ecmp": (ECMPRouting, {"weight": "delay"}),
}
_CACHE = {}
_PATHS = {}


def _distance(a, b):
    return math.dist(a, b)


def _reset():
    for cls in (Topology, NetworkSwitch, NetworkLink):
        cls._instances = []
        cls._object_count = 0


def _build_mesh(coords, rng):
    """Builds a connected mesh (Euclidean MST + k-NN) over the given coordinates."""
    n = len(coords)
    _reset()
    topology = Topology()
    switches = [NetworkSwitch() for _ in range(n)]
    for sw, coord in zip(switches, coords):
        sw.coordinates = coord
        topology.add_node(sw)

    # Candidate edges = MST (connectivity) + each node's k nearest neighbours (richness).
    complete = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            complete.add_edge(i, j, weight=_distance(coords[i], coords[j]))
    edges = set(tuple(sorted(e)) for e in nx.minimum_spanning_tree(complete).edges())
    for i in range(n):
        neighbours = sorted(range(n), key=lambda j: _distance(coords[i], coords[j]))[1 : K_NEAREST + 1]
        for j in neighbours:
            edges.add(tuple(sorted((i, j))))

    for i, j in edges:
        link = NetworkLink()
        link["topology"] = topology
        link["nodes"] = [switches[i], switches[j]]
        link["delay"] = max(1, int(_distance(coords[i], coords[j]) * DELAY_SCALE))
        link["bandwidth"] = 100
        link["bandwidth_demand"] = rng.uniform(0.0, 0.95) * 100  # randomized congestion per link
        topology.add_edge(switches[i], switches[j], **link)

    return topology, switches


def _path_metrics(topology, path):
    delay, max_util, hops = 0.0, 0.0, max(0, len(path) - 1)
    for i in range(len(path) - 1):
        link = topology[path[i]][path[i + 1]]
        delay += link["delay"]
        max_util = max(max_util, link_utilization(link))
    return delay, max_util, hops


def _coords():
    if "servers" not in _CACHE:
        _CACHE["servers"] = _read_coordinates(_PATHS["servers"], "latitude", "longitude")
    return _CACHE["servers"]


def trial(seed):
    rng = random.Random(seed)
    coords = rng.sample(_coords(), min(NUM_NODES, len(_coords())))
    topology, switches = _build_mesh(coords, rng)

    accum = {f"{name}_{m}": 0.0 for name in POLICIES for m in ("delay", "util", "hops")}
    reachable = 0
    for _ in range(FLOWS_PER_SEED):
        source, target = rng.sample(switches, 2)
        paths = {name: policy.get_path(topology, source, target, parameters=params) for name, (policy, params) in POLICIES.items()}
        if any(len(p) < 2 for p in paths.values()):
            continue
        reachable += 1
        for name, path in paths.items():
            delay, util, hops = _path_metrics(topology, path)
            accum[f"{name}_delay"] += delay
            accum[f"{name}_util"] += util
            accum[f"{name}_hops"] += hops

    n = max(reachable, 1)
    result = {key: value / n for key, value in accum.items()}
    result["avg_degree"] = 2 * topology.number_of_edges() / topology.number_of_nodes()
    return result


def main():
    parser = argparse.ArgumentParser(description="Routing on a real-coordinate geographic mesh.")
    parser.add_argument("--eua-servers", default="dataset/eua-dataset/edge-servers/site-optus-melbCBD.csv")
    args = parser.parse_args()
    _PATHS["servers"] = args.eua_servers

    print(f"Study 6 - Routing on real-coordinate mesh ({NUM_NODES} nodes, k={K_NEAREST}, " f"{FLOWS_PER_SEED} flows/seed, {len(list(SEEDS))} seeds, 95% CI)\n")
    summary = run_experiment(trial, seeds=SEEDS)["summary"]
    print(f"  mesh avg degree: {summary['avg_degree']['mean']:.2f}\n")

    print(f"{'Policy':<18}{'Path delay':>16}{'Bottleneck util':>18}{'Hops':>8}")
    rows, delays, delay_err, utils, util_err = [], [], [], [], []
    for name in POLICIES:
        d, u, h = (summary[f"{name}_{m}"] for m in ("delay", "util", "hops"))
        print(f"{name:<18}{d['mean']:>10.1f} +/-{d['half_width']:<5.1f}{u['mean']:>12.3f} +/-{u['half_width']:<5.3f}{h['mean']:>8.2f}")
        rows.append(
            {"policy": name, "mean_delay": d["mean"], "delay_ci": d["half_width"], "mean_util": u["mean"], "util_ci": u["half_width"], "mean_hops": h["mean"]}
        )
        delays.append(d["mean"])
        delay_err.append(d["half_width"])
        utils.append(u["mean"])
        util_err.append(u["half_width"])

    csv_path = results_to_csv(rows, output_path("study6_routing_topology.csv"))
    peak = max(delays) or 1.0
    fig_path = grouped_bar_chart(
        categories=list(POLICIES.keys()),
        series={"path delay (norm)": [d / peak for d in delays], "bottleneck utilization": utils},
        errors={"path delay (norm)": [e / peak for e in delay_err], "bottleneck utilization": util_err},
        title=f"Routing on a real-coordinate mesh ({NUM_NODES} EUA base stations)",
        ylabel="normalized delay  /  utilization",
        path=output_path("study6_routing_topology.png"),
    )
    print(f"\nWrote: {csv_path}\n       {fig_path}")


if __name__ == "__main__":
    main()
