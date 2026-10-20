# Pluggable Routing Policies & Programmable Edge Switches

This contribution extends EdgeSimPy's networking toward the direction the framework's
authors flagged as future work — *"advanced routing policies that leverage the
programmability of network switches at the edge."*

EdgeSimPy historically selected paths with a hardcoded delay-weighted
`networkx.shortest_path` call scattered across the components. This module turns
routing into a **pluggable layer** (mirroring the `flow_scheduling` and `power_models`
designs) and adds an **SDN/P4-style programmable-switch abstraction** with match-action
flow tables and a controller that installs forwarding rules.

## Components

| File | What it provides |
| --- | --- |
| `edge_sim_py/components/routing/base_routing.py` | Helpers: link utilization, delay normalization, single- and all-shortest-path computation. |
| `edge_sim_py/components/routing/shortest_path_routing.py` | `ShortestPathRouting` — delay-weighted shortest path (backward-compatible default). |
| `edge_sim_py/components/routing/least_congested_routing.py` | `LeastCongestedRouting` — weights links by bandwidth utilization to avoid congestion. |
| `edge_sim_py/components/routing/latency_aware_routing.py` | `LatencyAwareRouting` — tunable `α·delay + β·utilization` blend of latency and load. |
| `edge_sim_py/components/routing/ecmp_routing.py` | `ECMPRouting` — hashes a per-flow key across equal-cost paths (traffic splitting). |
| `edge_sim_py/components/routing/programmable_switch.py` | `FlowRule`, `RoutingController` and flow-table helpers (the SDN abstraction). |
| `examples/routing_policies_example.py` | Worked example comparing the policies and programming switches. |
| `tests/test_routing.py` | Unit tests (`pytest`). |

A routing policy is a class exposing:

```python
get_path(topology, source, target, flow=None, parameters=None) -> list  # list of switches
```

## Using a routing policy directly

```python
from edge_sim_py.components.routing import LeastCongestedRouting

path = LeastCongestedRouting.get_path(topology, source_switch, target_switch)
```

| Policy | Edge weight | When to use |
| --- | --- | --- |
| `ShortestPathRouting` | `delay` | Latency-minimizing baseline (current behavior). |
| `LeastCongestedRouting` | `bandwidth_demand / bandwidth` | Avoid congested links under load. |
| `LatencyAwareRouting` | `α·(delay/max_delay) + β·utilization` | Trade latency vs. congestion (`α`, `β` tunable). |
| `ECMPRouting` | equal-cost set + per-flow hash | Spread flows across equal-cost paths. |

## Opt-in wiring (backward compatible)

`Topology.get_path()` is the integration hook. With no policy set it reproduces the
historical delay-weighted shortest path, so existing simulations are unchanged:

```python
from edge_sim_py.components.routing import LatencyAwareRouting

topology.routing_policy = LatencyAwareRouting        # opt in
topology.routing_policy_parameters = {"alpha": 0.7, "beta": 0.3}
path = topology.get_path(source_switch, target_switch, flow=flow)
```

## Programmable switches (SDN)

Each `NetworkSwitch` now owns a `flow_table` of `FlowRule`s (`match` → `action`,
ordered by `priority`). The `RoutingController` separates control plane (deciding a
path) from data plane (the installed rules that forward traffic):

```python
from edge_sim_py.components.routing import RoutingController, LeastCongestedRouting

# Control plane: compute a path and program every switch on it.
RoutingController.install_path(
    topology, source_switch, target_switch,
    policy=LeastCongestedRouting, packet={"flow_id": flow.id},
)

# Data plane: follow the installed rules hop by hop.
visited = RoutingController.forward(source_switch, {"flow_id": flow.id})
```

Switches also expose `install_flow_rule()`, `lookup_flow_rule()` and
`clear_flow_table()` directly, and report their flow-table size via `collect()`.

## Running the tests and example

```bash
pip install pytest
pytest tests/test_routing.py
python examples/routing_policies_example.py
```

## Extending

Add a class with `get_path(topology, source, target, flow=None, parameters=None)` in
the `routing` package, export it from `routing/__init__.py`, and assign it to
`topology.routing_policy`. Compose new link weights from the helpers in
`base_routing.py` (delay, utilization) or any per-link/flow state you need.
