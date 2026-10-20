"""Unit tests for the real-dataset loaders (run against bundled format samples)."""

# EdgeSimPy
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.service import Service
from edge_sim_py.datasets import load_eua, load_shanghai, load_alibaba_tasks, sample_path
from edge_sim_py.workload import Task, ExecutionEngine, place_least_loaded, collect_workload_metrics
from edge_sim_py.workload.scenario import reset_workload_registries

import pytest


@pytest.fixture(autouse=True)
def _clean():
    for cls in (EdgeServer, Service, Task):
        cls._instances = []
        cls._object_count = 0
    yield


def test_eua_loader():
    servers, services = load_eua(sample_path("eua_base_stations_sample.csv"), sample_path("eua_users_sample.csv"), seed=1)
    assert len(servers) == 6 and len(services) == 8
    assert all(isinstance(s, EdgeServer) for s in servers)
    assert all(s.coordinates is not None for s in servers)
    # Coordinates are real Melbourne CBD latitudes/longitudes.
    assert -38 < servers[0].coordinates[0] < -37 and 144 < servers[0].coordinates[1] < 145
    assert servers[0].power_model is not None


def test_eua_service_demands_seeded():
    _, services_a = load_eua(sample_path("eua_base_stations_sample.csv"), sample_path("eua_users_sample.csv"), seed=42)
    reset_workload_registries()
    for cls in (EdgeServer, Service):
        cls._instances = []
        cls._object_count = 0
    _, services_b = load_eua(sample_path("eua_base_stations_sample.csv"), sample_path("eua_users_sample.csv"), seed=42)
    assert [s.cpu_demand for s in services_a] == [s.cpu_demand for s in services_b]  # reproducible


def test_shanghai_loader_deduplicates_base_stations():
    servers, services = load_shanghai(sample_path("shanghai_telecom_sample.csv"), seed=1)
    # 7 sessions but only 5 distinct coordinates -> 5 servers.
    assert len(services) == 7
    assert len(servers) == 5


def test_alibaba_loader():
    tasks = load_alibaba_tasks(sample_path("alibaba_batch_task_sample.csv"))
    assert len(tasks) == 8
    assert all(isinstance(t, Task) for t in tasks)
    # Earliest task (start_time 0) arrives at step 0 after normalization.
    assert min(t.arrival_step for t in tasks) == 0
    # A task with plan_cpu=100 has cpu_demand 1; execution reproduces its traced duration.
    first = next(t for t in tasks if t.arrival_step == 10)
    assert first.cpu_demand == 1
    assert first.cpu_cycles == 15  # end 25 - start 10


def test_alibaba_limit():
    tasks = load_alibaba_tasks(sample_path("alibaba_batch_task_sample.csv"), limit=3)
    assert len(tasks) == 3


def test_real_data_execution_run():
    servers, _ = load_eua(sample_path("eua_base_stations_sample.csv"), sample_path("eua_users_sample.csv"), seed=1)
    tasks = load_alibaba_tasks(sample_path("alibaba_batch_task_sample.csv"))
    place_least_loaded(tasks, servers)
    engine = ExecutionEngine(servers, tasks)
    engine.run()
    metrics = collect_workload_metrics(tasks, engine)
    assert metrics["completion_rate"] == 1.0
    assert metrics["mean_response_time"] > 0
