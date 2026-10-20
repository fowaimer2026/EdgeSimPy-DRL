"""Unit tests for the workload execution engine, energy meter and failure model.

These build small workloads directly (no dataset). Run with: ``pytest tests/test_workload.py``.
"""

# Python libraries
import pytest

# EdgeSimPy components
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.power_models.servers.linear_server_power_model import LinearServerPowerModel
from edge_sim_py.workload import (
    Task,
    ExecutionEngine,
    place_least_loaded,
    place_round_robin,
    collect_workload_metrics,
    random_workload,
    FINISHED,
    FAILED,
)
from edge_sim_py.workload.scenario import reset_workload_registries
from edge_sim_py.energy import EnergyMeter
from edge_sim_py.reliability import FailureModel


@pytest.fixture(autouse=True)
def _clean_registries():
    reset_workload_registries()
    yield


def _server(cpu=8, memory=4096, power=False):
    server = EdgeServer(cpu=cpu, memory=memory, disk=10_000)
    if power:
        server.power_model = LinearServerPowerModel
        server.power_model_parameters = {"max_power_consumption": 200, "static_power_percentage": 0.4}
    return server


# ----------------------------------------------------------------- task & engine


def test_task_execution_time_matches_workload():
    server = _server(cpu=8)
    # 10 cycles at cpu_demand=2 -> 5 ticks of execution, no queueing (arrives at 0).
    task = Task(cpu_cycles=10, cpu_demand=2, arrival_step=0, server=server)
    engine = ExecutionEngine([server], [task])
    engine.run()

    assert task.status == FINISHED
    assert task.execution_time == 5
    assert task.waiting_time == 0
    assert task.response_time == 5


def test_arrival_time_is_respected():
    server = _server()
    task = Task(cpu_cycles=4, cpu_demand=1, arrival_step=3, server=server)
    engine = ExecutionEngine([server], [task])
    engine.run()
    # Cannot start before arrival; starts at step 3 (the first tick >= arrival).
    assert task.start_step >= 3
    assert task.waiting_time == task.start_step - 3


def test_queueing_under_contention():
    # A 2-CPU server with two 2-CPU tasks can only run one at a time -> the second queues.
    server = _server(cpu=2)
    first = Task(cpu_cycles=4, cpu_demand=2, arrival_step=0, server=server)
    second = Task(cpu_cycles=4, cpu_demand=2, arrival_step=0, server=server)
    engine = ExecutionEngine([server], [first, second])
    engine.run()

    assert first.finished and second.finished
    # One of them had to wait for the other to finish.
    assert max(first.waiting_time, second.waiting_time) > 0


def test_capacity_allows_parallel_execution():
    # A 4-CPU server runs two 2-CPU tasks in parallel -> neither waits.
    server = _server(cpu=4)
    a = Task(cpu_cycles=4, cpu_demand=2, arrival_step=0, server=server)
    b = Task(cpu_cycles=4, cpu_demand=2, arrival_step=0, server=server)
    ExecutionEngine([server], [a, b]).run()
    assert a.waiting_time == 0 and b.waiting_time == 0


def test_deadline_violation_detected():
    server = _server(cpu=1)
    # Needs 10 ticks but deadline is step 5 -> violated.
    task = Task(cpu_cycles=10, cpu_demand=1, arrival_step=0, deadline=5, server=server)
    ExecutionEngine([server], [task]).run()
    assert task.finished
    assert task.met_deadline is False


# ----------------------------------------------------------------- energy


def test_energy_meter_accumulates():
    server = _server(cpu=8, power=True)
    task = Task(cpu_cycles=8, cpu_demand=4, arrival_step=0, server=server)
    meter = EnergyMeter(tick_duration=1.0)
    engine = ExecutionEngine([server], [task], energy_meter=meter)
    engine.run()

    assert meter.total_energy > 0
    assert meter.samples == engine.current_step
    assert meter.energy_of(server) == pytest.approx(meter.total_energy)


def test_energy_zero_without_power_model():
    server = _server(power=False)
    task = Task(cpu_cycles=4, cpu_demand=1, server=server)
    meter = EnergyMeter()
    ExecutionEngine([server], [task], energy_meter=meter).run()
    assert meter.total_energy == 0.0


# ----------------------------------------------------------------- reliability


def test_failure_model_toggles_availability():
    server = _server()
    # MTBF=1 with tick=1 -> failure probability clamps to 1.0 (fails immediately).
    model = FailureModel(mtbf=1, mttr=100, seed=0)
    events = model.step([server], current_step=1)
    assert server.available is False
    assert server.active is False
    assert events and events[0]["event"] == "failure"


def test_failure_requeues_running_task():
    server = _server(cpu=4)
    task = Task(cpu_cycles=100, cpu_demand=2, arrival_step=0, server=server)
    model = FailureModel(mtbf=1, mttr=1, seed=0)
    engine = ExecutionEngine([server], [task], failure_model=model)
    # Run a few ticks; the server will fail and recover, re-queuing the task.
    for _ in range(5):
        engine.step()
    # The task must never be lost: it is always in some valid state.
    assert task.status in ("waiting", "running", "finished")


def test_steady_state_availability_formula():
    assert FailureModel.steady_state_availability(90, 10) == pytest.approx(0.9)


# ----------------------------------------------------------------- placement & metrics


def test_random_workload_shapes():
    servers, tasks = random_workload(num_tasks=20, num_servers=3, seed=1)
    assert len(servers) == 3 and len(tasks) == 20
    assert all(task.server is None for task in tasks)  # unplaced until a placer runs


def test_placement_assigns_all_tasks():
    servers, tasks = random_workload(num_tasks=20, num_servers=3, seed=1)
    place_round_robin(tasks, servers)
    assert all(task.server is not None for task in tasks)


def test_metrics_completion_and_energy():
    servers, tasks = random_workload(num_tasks=30, num_servers=4, seed=2)
    place_least_loaded(tasks, servers)
    meter = EnergyMeter()
    engine = ExecutionEngine(servers, tasks, energy_meter=meter)
    engine.run()
    metrics = collect_workload_metrics(tasks, engine)

    assert metrics["tasks_total"] == 30
    assert metrics["completion_rate"] == 1.0
    assert metrics["mean_response_time"] > 0
    assert "total_energy" in metrics
