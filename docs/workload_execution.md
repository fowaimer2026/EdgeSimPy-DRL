# Workload Execution, Energy & Reliability

EdgeSimPy models how a service *reaches* a server (container layers, migration, network
flows) in detail, but historically modeled neither the **execution** of work nor the
**energy consumed over time**, and its edge servers could never **fail**. This
contribution adds those three "consequence" layers, so a placement decision yields
ground-truth response times, queueing delays, deadline-violation rates, energy and
reliability impact — exactly the metrics needed to evaluate offloading policies.

All three are opt-in packages:

```python
from edge_sim_py.workload import ExecutionEngine, Task, random_workload, place_least_loaded, collect_workload_metrics
from edge_sim_py.energy import EnergyMeter
from edge_sim_py.reliability import FailureModel
```

## 1. Task execution (`edge_sim_py.workload`)

A `Task` is a discrete computation: it *arrives* at a step, *queues* for CPU, *executes*
for a workload-dependent duration, and *finishes* — optionally under a deadline.

**CPU model.** A server runs as many queued tasks as fit within its CPU capacity (the sum
of running tasks' `cpu_demand` may not exceed `server.cpu`); the rest wait FIFO by
arrival. A running task advances `cpu_demand * processing_rate` cycles per tick, so its
execution time is `cpu_cycles / (cpu_demand * processing_rate)` and **contention shows up
as queueing delay**. The server's `cpu_demand` tracks the running tasks, so the power
models (and the energy meter) reflect the real load.

> **Modeling assumptions** (state these in any write-up): space-sharing (not
> processor-sharing/time-slicing), no preemption, integer-step granularity, and tasks
> restart from scratch if their server fails (no checkpointing).

```python
servers, tasks = random_workload(num_tasks=60, num_servers=3, seed=7)
place_least_loaded(tasks, servers)
engine = ExecutionEngine(servers, tasks)
engine.run()
print(collect_workload_metrics(tasks, engine))
# completion_rate, mean_response_time, mean_waiting_time, mean_execution_time, deadline_violation_rate
```

Each `Task` exposes `waiting_time`, `execution_time`, `response_time` (inclusive-interval)
and `met_deadline`. Placement helpers: `place_least_loaded`, `place_round_robin`,
`place_random`.

## 2. Energy accounting (`edge_sim_py.energy`)

`EnergyMeter` integrates instantaneous power into cumulative energy. Sampled once per tick
(`energy += power * tick_duration`) over any device exposing `get_power_consumption()`
(edge servers, switches):

```python
meter = EnergyMeter(tick_duration=1.0)
engine = ExecutionEngine(servers, tasks, energy_meter=meter)
engine.run()
print(meter.metrics())   # total_energy, average_power, per-device breakdown
```

It also works as a monitor hook on a live `Simulator` (sample its servers/switches each
step).

## 3. Reliability (`edge_sim_py.reliability`)

`FailureModel` injects stochastic failures and repairs from MTBF/MTTR. Each tick a healthy
server fails with probability `tick_duration / MTBF` and a failed server recovers with
probability `tick_duration / MTTR`. A failed server is marked unavailable **and** inactive
(so power drops to zero and the engine stops scheduling on it); its running tasks are
re-queued (or marked failed).

```python
engine = ExecutionEngine(servers, tasks, failure_model=FailureModel(mtbf=40, mttr=10, seed=3))
engine.run()
FailureModel.steady_state_availability(40, 10)   # 0.8
```

## Why it matters

These layers turn the cost models' *analytical* estimates into *simulated ground truth*:
the latency a `LatencyCostModel` predicts can be checked against the response time the
engine measures, and the energy an `EnergyCostModel` predicts against metered energy. They
also give a DRL agent genuine temporal coupling — a placement that congests a server's
queue hurts later tasks.

## Running the tests and example

```bash
pytest tests/test_workload.py
python examples/workload_execution_example.py
```
