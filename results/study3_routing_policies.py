"""Study 3 - Routing policy comparison under congestion.

Builds a source->sink topology with several parallel 2-hop paths whose per-path delay and
congestion are randomized each seed, then routes a flow under each policy and measures the
chosen path's total delay and bottleneck (max-link) utilization. Reports means with 95% CIs
across seeds, exposing the latency-vs-congestion trade-off. Emits a CSV and a figure.
"""

# EdgeSimPy
from edge_sim_py.components.topology import Topology
from edge_sim_py.components.network_switch import NetworkSwitch
from edge_sim_py.components.network_link import NetworkLink
from edge_sim_py.components.routing import (
    ShortestPathRouting,
    LeastCongestedRouting,
    LatencyAwareRouting,
    ECMPRouting,
)
from edge_sim_py.components.routing.base_routing import link_utilization
from edge_sim_py.experiments import run_experiment, results_to_csv

# Python libraries
import random

from _common import output_path, grouped_bar_chart

NUM_PATHS = 6
SEEDS = range(40)
# ECMP treats all equal-hop paths as equal-cost (weight=None -> hop count).
POLICIES = {
    "shortest_path": (ShortestPathRouting, None),
    "least_congested": (LeastCongestedRouting, None),
    "latency_aware": (LatencyAwareRouting, {"alpha": 0.5, "beta": 0.5}),
    "ecmp": (ECMPRouting, {"weight": None}),
}


def _reset():
    for cls in (Topology, NetworkSwitch, NetworkLink):
        cls._instances = []
        cls._object_count = 0


def _build_parallel_topology(rng):
    """Builds source -> {middle_i} -> sink with randomized delay and congestion per path."""
    topology = Topology()
    source = NetworkSwitch()
    sink = NetworkSwitch()
    topology.add_node(source)
    topology.add_node(sink)

    for _ in range(NUM_PATHS):
        middle = NetworkSwitch()
        topology.add_node(middle)
        utilization = rng.uniform(0.0, 0.95)  # shared congestion level for this path
        for a, b in ((source, middle), (middle, sink)):
            link = NetworkLink()
            link["topology"] = topology
            link["nodes"] = [a, b]
            link["delay"] = rng.randint(1, 10)
            link["bandwidth"] = 100
            link["bandwidth_demand"] = utilization * 100
            topology.add_edge(a, b, **link)

    return topology, source, sink


def _path_metrics(topology, path):
    """Returns (total_delay, max_utilization) for a path of switches."""
    total_delay = 0.0
    max_util = 0.0
    for i in range(len(path) - 1):
        link = topology[path[i]][path[i + 1]]
        total_delay += link["delay"]
        max_util = max(max_util, link_utilization(link))
    return total_delay, max_util


def trial(seed):
    rng = random.Random(seed)
    _reset()
    topology, source, sink = _build_parallel_topology(rng)

    result = {}
    for name, (policy, params) in POLICIES.items():
        path = policy.get_path(topology, source, sink, parameters=params)
        delay, util = _path_metrics(topology, path)
        result[f"{name}_delay"] = delay
        result[f"{name}_util"] = util
    return result


def main():
    print(f"Study 3 - Routing policies ({len(list(SEEDS))} seeds, 95% CI)\n")
    summary = run_experiment(trial, seeds=SEEDS)["summary"]

    print(f"{'Policy':<18}{'Path delay':>14}{'Bottleneck util':>18}")
    rows, delays, delay_err, utils, util_err = [], [], [], [], []
    for name in POLICIES:
        d = summary[f"{name}_delay"]
        u = summary[f"{name}_util"]
        print(f"{name:<18}{d['mean']:>8.2f} +/- {d['half_width']:<4.2f}{u['mean']:>12.3f} +/- {u['half_width']:<4.3f}")
        rows.append({"policy": name, "mean_delay": d["mean"], "delay_ci": d["half_width"], "mean_util": u["mean"], "util_ci": u["half_width"]})
        delays.append(d["mean"])
        delay_err.append(d["half_width"])
        utils.append(u["mean"])
        util_err.append(u["half_width"])

    csv_path = results_to_csv(rows, output_path("study3_routing_policies.csv"))
    fig_path = grouped_bar_chart(
        categories=list(POLICIES.keys()),
        series={"path delay": delays, "bottleneck utilization x10": [u * 10 for u in utils]},
        errors={"path delay": delay_err, "bottleneck utilization x10": [e * 10 for e in util_err]},
        title="Routing policies: latency vs. congestion trade-off",
        ylabel="delay  /  utilization x10",
        path=output_path("study3_routing_policies.png"),
    )
    print(f"\nWrote: {csv_path}\n       {fig_path}")


if __name__ == "__main__":
    main()
