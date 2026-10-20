# Computation-Offloading Cost Models

This contribution adds a pluggable **computation-offloading cost model** layer to
EdgeSimPy, plus a reference cost-driven offloading policy, a worked example, and unit
tests. It lets you quantify *how expensive* it is to offload a service onto a given
edge server and to drive placement decisions by minimizing that cost.

The design mirrors EdgeSimPy's existing `power_models` package: each cost model is a
small class exposing class methods, and the package is auto-exported through
`edge_sim_py.components`.

## Components

| File | What it provides |
| --- | --- |
| `edge_sim_py/components/cost_models/base_cost_model.py` | Shared helpers: transmission delay, network propagation delay (uses the real topology), processing delay, payload-size resolution. |
| `edge_sim_py/components/cost_models/latency_cost_model.py` | `LatencyCostModel` — transmission + propagation + processing latency. |
| `edge_sim_py/components/cost_models/energy_cost_model.py` | `EnergyCostModel` — device transmit/idle energy + server compute energy (reuses the server's power model). |
| `edge_sim_py/components/cost_models/monetary_cost_model.py` | `MonetaryCostModel` — compute + memory + data-transfer charges, with optional per-server pricing. |
| `edge_sim_py/components/cost_models/weighted_cost_model.py` | `WeightedCostModel` — weighted multi-objective cost; `rank_servers()` min-max normalizes objectives across candidates. |
| `edge_sim_py/offloading/min_cost_offloading.py` | `min_cost_offloading` — a `resource_management_algorithm` that offloads each pending service to the cheapest reachable server. |
| `edge_sim_py/offloading/baseline_policies.py` | `latency_greedy_offloading` and `random_offloading` — baselines for benchmarking. |
| `edge_sim_py/offloading/metrics.py` | `collect_cost_metrics`, `collect_sla_metrics`, `collect_offloading_metrics` — allocation-wide aggregation. |
| `examples/offloading_cost_model_example.py` | End-to-end example (dataset-driven or stand-alone demo). |
| `tests/test_cost_models.py`, `tests/test_offloading_metrics.py` | Unit tests (`pytest`). |

## Cost formulation

Each model evaluates a `(service, server)` pair, optionally from a `user`'s
perspective (needed for the network path):

- **Latency** `= data/bandwidth + path_delay + cpu_demand/free_cpu`
- **Energy** `= P_tx·t_tx + P_idle·(t_prop + t_proc) + P_server·t_proc`
- **Monetary** `= price_cpu·cpu_demand·t_proc + price_mem·mem_demand·t_proc + price_gb·data_gb`
- **Weighted** `= Σ wᵢ · normalized(objectiveᵢ)` with weights renormalized to sum to 1.

Every model also exposes a `get_cost_breakdown()` (or `get_objectives()`) method so you
can inspect the individual terms for analysis or plotting.

## Usage

```python
from edge_sim_py.components.cost_models.weighted_cost_model import WeightedCostModel

params = {"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}}
ranking = WeightedCostModel.rank_servers(service, candidate_servers, user=user, parameters=params)
best_server = ranking[0]["server"]
```

Driving a simulation with the reference policy:

```python
from edge_sim_py import Simulator
from edge_sim_py.offloading import min_cost_offloading

simulator = Simulator(
    resource_management_algorithm=min_cost_offloading,
    resource_management_algorithm_parameters={"cost_model_parameters": params},
    # ... other Simulator arguments ...
)
simulator.initialize(input_file="dataset.json")
simulator.run_model()
```

## Measuring an allocation

After (or during) a run, aggregate cost and SLA metrics from the live component state:

```python
from edge_sim_py.offloading import collect_offloading_metrics

metrics = collect_offloading_metrics(parameters=params)
print(metrics["cost"]["placement_rate"], metrics["cost"]["total_cost"])
print(metrics["sla"]["satisfaction_rate"], metrics["sla"]["violations"])
```

`collect_cost_metrics` reports placement rate, total/average cost and per-objective
totals; `collect_sla_metrics` reports delay-SLA satisfaction across all users.

## Benchmarking against baselines

Swap the `resource_management_algorithm` to compare the cost-driven policy against
simpler strategies that share the same interface:

```python
from edge_sim_py.offloading import latency_greedy_offloading, random_offloading
# Simulator(resource_management_algorithm=latency_greedy_offloading, ...)
# Simulator(resource_management_algorithm=random_offloading,
#           resource_management_algorithm_parameters={"seed": 1}, ...)
```

## Running the tests and example

```bash
pip install pytest
pytest tests/test_cost_models.py
python examples/offloading_cost_model_example.py            # stand-alone demo
python examples/offloading_cost_model_example.py --dataset dataset.json
```

## Extending

To add a new objective (e.g. carbon footprint), create a class with
`get_cost(service, server, user=None, parameters=None)` in the `cost_models` package,
export it from `cost_models/__init__.py`, and optionally fold it into
`WeightedCostModel`. The same per-object `parameters` convention used by the power
models applies.
