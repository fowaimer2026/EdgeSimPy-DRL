"""Study 10 - Scalability & runtime overhead (Part 3, item 4).

Quantifies (a) the runtime overhead of each added layer relative to a bare execution run, and
(b) how wall-clock and peak memory scale with the number of servers and tasks. Synthetic so it
can be scaled freely; torch-free and fast.

Layers measured incrementally: bare execution engine -> + energy meter -> + failure model ->
+ cost-model placement decisions -> + routing path computations.
"""

# EdgeSimPy
from edge_sim_py.components.topology import Topology
from edge_sim_py.components.network_switch import NetworkSwitch
from edge_sim_py.components.network_link import NetworkLink
from edge_sim_py.components.cost_models.weighted_cost_model import WeightedCostModel
from edge_sim_py.components.routing import LatencyAwareRouting
from edge_sim_py.workload import random_workload, place_least_loaded, ExecutionEngine
from edge_sim_py.energy import EnergyMeter
from edge_sim_py.reliability import FailureModel
from edge_sim_py.experiments import results_to_csv

# Python libraries
import math
import random
import time
import tracemalloc

from _common import output_path, line_chart, grouped_bar_chart

SEEDS = range(5)


def _time_ms(fn):
    """Returns (result, elapsed_ms) for a zero-argument callable."""
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000.0


def _run_engine(num_tasks, num_servers, seed, energy=False, failures=False):
    servers, tasks = random_workload(num_tasks=num_tasks, num_servers=num_servers, seed=seed)
    place_least_loaded(tasks, servers)
    meter = EnergyMeter() if energy else None
    fmodel = FailureModel(mtbf=400.0, mttr=20.0, seed=seed) if failures else None
    engine = ExecutionEngine(servers, tasks, energy_meter=meter, failure_model=fmodel)
    engine.run(max_steps=40_000)
    return servers, tasks


def _cost_decisions(num_tasks, num_servers, seed):
    """Cost-model scoring cost: score every task against every server once."""
    servers, tasks = random_workload(num_tasks=num_tasks, num_servers=num_servers, seed=seed)
    for task in tasks:
        for server in servers:
            WeightedCostModel.get_objectives(task, server, None, {})


def _routing_decisions(num_nodes, num_flows, seed):
    """Routing cost: build a mesh and route num_flows random pairs (latency-aware)."""
    rng = random.Random(seed)
    for cls in (Topology, NetworkSwitch, NetworkLink):
        cls._instances = []
        cls._object_count = 0
    topology = Topology()
    coords = [(rng.random(), rng.random()) for _ in range(num_nodes)]
    switches = [NetworkSwitch() for _ in range(num_nodes)]
    for sw in switches:
        topology.add_node(sw)
    import networkx as nx

    complete = nx.Graph()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            complete.add_edge(i, j, weight=math.dist(coords[i], coords[j]))
    edges = set(tuple(sorted(e)) for e in nx.minimum_spanning_tree(complete).edges())
    for i in range(num_nodes):
        for j in sorted(range(num_nodes), key=lambda k: math.dist(coords[i], coords[k]))[1:5]:
            edges.add(tuple(sorted((i, j))))
    for i, j in edges:
        link = NetworkLink()
        link["topology"] = topology
        link["nodes"] = [switches[i], switches[j]]
        link["delay"] = max(1, int(math.dist(coords[i], coords[j]) * 1000))
        link["bandwidth"] = 100
        link["bandwidth_demand"] = rng.uniform(0, 0.9) * 100
        topology.add_edge(switches[i], switches[j], **link)
    for _ in range(num_flows):
        a, b = rng.sample(switches, 2)
        LatencyAwareRouting.get_path(topology, a, b, parameters={"alpha": 0.5, "beta": 0.5})


def layer_overhead():
    """Incremental per-layer wall-clock over a fixed 1000-task / 20-server workload."""
    n_tasks, n_servers = 1000, 20
    layers = {
        "execution (bare)": lambda s: _run_engine(n_tasks, n_servers, s),
        "+ energy meter": lambda s: _run_engine(n_tasks, n_servers, s, energy=True),
        "+ failures": lambda s: _run_engine(n_tasks, n_servers, s, energy=True, failures=True),
        "cost decisions (1000x20)": lambda s: _cost_decisions(n_tasks, n_servers, s),
        "routing (2000 flows, 20 nodes)": lambda s: _routing_decisions(20, 2000, s),
    }
    rows = []
    print("A. Per-layer runtime (mean ms over seeds):")
    for name, fn in layers.items():
        times = [_time_ms(lambda: fn(seed))[1] for seed in SEEDS]
        mean_ms = sum(times) / len(times)
        print(f"  {name:<32} {mean_ms:8.1f} ms")
        rows.append({"layer": name, "mean_ms": mean_ms})
    return rows


def scaling_curves():
    """Wall-clock and peak memory vs. #tasks and #servers."""
    print("\nB. Scaling (execution + energy + failures):")
    task_sizes = [500, 1000, 2000, 4000]
    server_sizes = [10, 25, 50, 100]
    task_rows, server_rows = [], []

    for n in task_sizes:
        tms, mems = [], []
        for seed in SEEDS:
            tracemalloc.start()
            _, elapsed = _time_ms(lambda: _run_engine(n, 25, seed, energy=True, failures=True))
            peak = tracemalloc.get_traced_memory()[1] / 1e6
            tracemalloc.stop()
            tms.append(elapsed)
            mems.append(peak)
        task_rows.append({"num_tasks": n, "mean_ms": sum(tms) / len(tms), "peak_mb": sum(mems) / len(mems)})
        print(f"  tasks={n:>5}: {task_rows[-1]['mean_ms']:8.1f} ms   {task_rows[-1]['peak_mb']:6.1f} MB")

    for n in server_sizes:
        tms = []
        for seed in SEEDS:
            _, elapsed = _time_ms(lambda: _run_engine(2000, n, seed, energy=True, failures=True))
            tms.append(elapsed)
        server_rows.append({"num_servers": n, "mean_ms": sum(tms) / len(tms)})
        print(f"  servers={n:>4}: {server_rows[-1]['mean_ms']:8.1f} ms")

    return task_rows, server_rows


def main():
    print(f"Study 10 - Scalability & runtime overhead ({len(list(SEEDS))} seeds)\n")

    layers = layer_overhead()
    task_rows, server_rows = scaling_curves()

    results_to_csv(layers, output_path("study10_layer_overhead.csv"))
    results_to_csv(task_rows, output_path("study10_scaling_tasks.csv"))
    results_to_csv(server_rows, output_path("study10_scaling_servers.csv"))

    grouped_bar_chart(
        categories=[r["layer"].replace(" ", "\n", 1) for r in layers],
        series={"runtime (ms)": [r["mean_ms"] for r in layers]},
        title="Per-layer runtime over a fixed workload (1000 tasks / 20 servers)",
        ylabel="milliseconds",
        path=output_path("study10_layer_overhead.png"),
    )
    line_chart(
        x=[r["num_tasks"] for r in task_rows],
        series={"wall-clock (ms)": [r["mean_ms"] for r in task_rows], "peak memory (MB)": [r["peak_mb"] for r in task_rows]},
        title="Scaling with number of tasks (25 servers)",
        xlabel="Number of tasks",
        ylabel="ms  /  MB",
        path=output_path("study10_scaling_tasks.png"),
    )
    print(f"\nWrote CSVs + figures to {output_path('')}")


if __name__ == "__main__":
    main()
