"""Unit tests for the offloading metrics helpers.

These tests build lightweight Service/EdgeServer/User/Application objects directly
(no dataset required) and verify the cost/SLA aggregation. Run with:
``pytest tests/test_offloading_metrics.py``.
"""

# Python libraries
import pytest

# EdgeSimPy components
from edge_sim_py.components.service import Service
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.user import User
from edge_sim_py.components.application import Application
from edge_sim_py.offloading.metrics import collect_cost_metrics, collect_sla_metrics


@pytest.fixture(autouse=True)
def _clean_registries():
    """Resets EdgeSimPy class-level instance registries before each test."""
    for cls in (Service, EdgeServer, User, Application):
        cls._instances = []
        cls._object_count = 0
    yield


def _make_service(cpu_demand=2, memory_demand=512, state=10):
    return Service(label="svc", cpu_demand=cpu_demand, memory_demand=memory_demand, state=state)


def _make_server(cpu=8, memory=4096):
    return EdgeServer(model_name="srv", cpu=cpu, memory=memory, disk=10000)


def test_cost_metrics_placement_rate():
    placed = _make_service()
    placed.server = _make_server()
    _make_service()  # unplaced

    metrics = collect_cost_metrics()

    assert metrics["services_total"] == 2
    assert metrics["services_placed"] == 1
    assert metrics["placement_rate"] == pytest.approx(0.5)


def test_cost_metrics_total_and_average_are_consistent():
    server = _make_server()
    for _ in range(2):
        service = _make_service()
        service.server = server

    metrics = collect_cost_metrics()

    assert metrics["services_placed"] == 2
    assert metrics["total_cost"] > 0
    assert metrics["average_cost"] == pytest.approx(metrics["total_cost"] / 2)
    # The weighted objective totals should also be populated.
    assert metrics["objective_totals"]["latency"] > 0


def test_cost_metrics_empty_infrastructure():
    metrics = collect_cost_metrics()

    assert metrics["services_total"] == 0
    assert metrics["placement_rate"] == 0.0
    assert metrics["average_cost"] == 0.0


def test_sla_metrics_counts_violations():
    application = Application(label="app")
    user = User()
    user.applications = [application]
    user.delays = {str(application.id): 50.0}
    user.delay_slas = {str(application.id): 30.0}  # 50 > 30 -> violation

    metrics = collect_sla_metrics()

    assert metrics["evaluated"] == 1
    assert metrics["violations"] == 1
    assert metrics["satisfaction_rate"] == pytest.approx(0.0)


def test_sla_metrics_counts_satisfied():
    application = Application(label="app")
    user = User()
    user.applications = [application]
    user.delays = {str(application.id): 10.0}
    user.delay_slas = {str(application.id): 30.0}  # 10 <= 30 -> satisfied

    metrics = collect_sla_metrics()

    assert metrics["evaluated"] == 1
    assert metrics["violations"] == 0
    assert metrics["satisfaction_rate"] == pytest.approx(1.0)
