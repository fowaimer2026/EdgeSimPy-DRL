"""Unit tests for the computation-offloading cost models.

These tests exercise the cost models in isolation, building lightweight Service and
EdgeServer objects directly (no dataset/topology required) by passing ``user=None``,
which makes the propagation-delay term zero. Run with: ``pytest tests/test_cost_models.py``.
"""

# Python libraries
import pytest

# EdgeSimPy components
from edge_sim_py.components.service import Service
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.cost_models.latency_cost_model import LatencyCostModel
from edge_sim_py.components.cost_models.energy_cost_model import EnergyCostModel
from edge_sim_py.components.cost_models.monetary_cost_model import MonetaryCostModel
from edge_sim_py.components.cost_models.weighted_cost_model import WeightedCostModel


@pytest.fixture(autouse=True)
def _clean_registries():
    """Resets EdgeSimPy class-level instance registries before each test."""
    for cls in (Service, EdgeServer):
        cls._instances = []
        cls._object_count = 0
    yield


def _make_service(cpu_demand=2, memory_demand=512, state=10):
    return Service(label="svc", cpu_demand=cpu_demand, memory_demand=memory_demand, state=state)


def _make_server(cpu=8, memory=4096, cpu_demand=0):
    server = EdgeServer(model_name="srv", cpu=cpu, memory=memory, disk=10000)
    server.cpu_demand = cpu_demand
    return server


def test_latency_breakdown_sums_to_total():
    service = _make_service()
    server = _make_server()

    breakdown = LatencyCostModel.get_cost_breakdown(service, server, user=None)

    assert breakdown["total"] == pytest.approx(breakdown["transmission"] + breakdown["propagation"] + breakdown["processing"])
    # No user -> no propagation delay.
    assert breakdown["propagation"] == 0.0


def test_latency_transmission_uses_access_bandwidth():
    service = _make_service(state=200)
    server = _make_server()

    # 200 Mbit over a 100 Mbps link -> 2.0 time units of transmission delay.
    breakdown = LatencyCostModel.get_cost_breakdown(service, server, user=None, parameters={"access_bandwidth": 100.0})
    assert breakdown["transmission"] == pytest.approx(2.0)


def test_processing_delay_grows_with_server_load():
    service = _make_service()
    idle_server = _make_server(cpu=8, cpu_demand=0)
    busy_server = _make_server(cpu=8, cpu_demand=6)

    idle = LatencyCostModel.get_cost_breakdown(service, idle_server, user=None)["processing"]
    busy = LatencyCostModel.get_cost_breakdown(service, busy_server, user=None)["processing"]

    assert busy > idle


def test_energy_includes_server_compute_term():
    service = _make_service()
    server = _make_server()

    breakdown = EnergyCostModel.get_cost_breakdown(service, server, user=None)

    assert breakdown["server_compute"] > 0
    assert breakdown["total"] == pytest.approx(breakdown["device_transmit"] + breakdown["device_idle"] + breakdown["server_compute"])


def test_monetary_respects_per_server_pricing():
    service = _make_service()
    cheap = _make_server()
    premium = _make_server()
    premium.cost_model_parameters = {"price_per_cpu_unit_time": 0.5}  # 10x the default

    cheap_cost = MonetaryCostModel.get_cost(service, cheap)
    premium_cost = MonetaryCostModel.get_cost(service, premium)

    assert premium_cost > cheap_cost


def test_weighted_weights_are_renormalized():
    params = {"weights": {"latency": 2.0, "energy": 2.0, "monetary": 2.0}}
    weights = WeightedCostModel._normalized_weights(params)

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["latency"] == pytest.approx(1.0 / 3.0)


def test_rank_servers_prefers_less_loaded_server():
    service = _make_service()
    idle_server = _make_server(cpu=8, cpu_demand=0)
    busy_server = _make_server(cpu=8, cpu_demand=6)

    ranking = WeightedCostModel.rank_servers(service, [busy_server, idle_server], user=None)

    # The idle server should be the cheapest (first) option.
    assert ranking[0]["server"] is idle_server
    assert ranking[0]["score"] <= ranking[1]["score"]


def test_rank_servers_scores_are_normalized_between_zero_and_one():
    service = _make_service()
    servers = [_make_server(cpu_demand=load) for load in (0, 3, 6)]

    ranking = WeightedCostModel.rank_servers(service, servers, user=None)
    scores = [entry["score"] for entry in ranking if entry["score"] != float("inf")]

    assert min(scores) == pytest.approx(0.0)
    assert max(scores) <= 1.0 + 1e-9
