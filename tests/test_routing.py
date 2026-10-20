""" Unit tests for the pluggable routing policies and programmable-switch abstraction.

The tests build a small diamond topology directly (no dataset required):

        s2
       /  \\
     s1    s4
       \\  /
        s3

The s1-s2-s4 branch has low delay but is congested; the s1-s3-s4 branch has higher
delay but is free. Run with: ``pytest tests/test_routing.py``.
"""

# Python libraries
import pytest

# EdgeSimPy components
from edge_sim_py.components.topology import Topology
from edge_sim_py.components.network_switch import NetworkSwitch
from edge_sim_py.components.network_link import NetworkLink
from edge_sim_py.components.routing import (
    ShortestPathRouting,
    LeastCongestedRouting,
    LatencyAwareRouting,
    ECMPRouting,
    FlowRule,
    RoutingController,
)


@pytest.fixture(autouse=True)
def _clean_registries():
    for cls in (Topology, NetworkSwitch, NetworkLink):
        cls._instances = []
        cls._object_count = 0
    yield


def _ids(path):
    return [node.id for node in path]


@pytest.fixture
def diamond():
    """Builds the diamond topology described in the module docstring."""
    topology = Topology()
    switches = {i: NetworkSwitch(obj_id=i) for i in range(1, 5)}
    for switch in switches.values():
        topology.add_node(switch)

    def add_link(a, b, delay, bandwidth, demand=0):
        link = NetworkLink()
        link["topology"] = topology
        link["nodes"] = [switches[a], switches[b]]
        link["delay"] = delay
        link["bandwidth"] = bandwidth
        link["bandwidth_demand"] = demand
        topology.add_edge(switches[a], switches[b], **link)

    add_link(1, 2, 1, 100, 95)  # low delay, congested
    add_link(2, 4, 1, 100, 95)
    add_link(1, 3, 3, 100, 0)  # high delay, free
    add_link(3, 4, 3, 100, 0)
    return topology, switches


def test_shortest_path_takes_low_delay_branch(diamond):
    topology, sw = diamond
    path = ShortestPathRouting.get_path(topology, sw[1], sw[4])
    assert _ids(path) == [1, 2, 4]


def test_least_congested_avoids_congested_branch(diamond):
    topology, sw = diamond
    path = LeastCongestedRouting.get_path(topology, sw[1], sw[4])
    assert _ids(path) == [1, 3, 4]


def test_latency_aware_degenerates_to_shortest_path(diamond):
    topology, sw = diamond
    # alpha=1, beta=0 -> pure delay weighting.
    path = LatencyAwareRouting.get_path(topology, sw[1], sw[4], parameters={"alpha": 1.0, "beta": 0.0})
    assert _ids(path) == [1, 2, 4]


def test_latency_aware_degenerates_to_least_congested(diamond):
    topology, sw = diamond
    # alpha=0, beta=1 -> pure congestion weighting.
    path = LatencyAwareRouting.get_path(topology, sw[1], sw[4], parameters={"alpha": 0.0, "beta": 1.0})
    assert _ids(path) == [1, 3, 4]


def test_ecmp_returns_a_valid_equal_cost_path(diamond):
    topology, sw = diamond
    # Make both branches equal-cost on hop count so ECMP has two choices.
    path = ECMPRouting.get_path(topology, sw[1], sw[4], parameters={"weight": None})
    assert _ids(path) in ([1, 2, 4], [1, 3, 4])


def test_no_path_returns_empty_list(diamond):
    topology, sw = diamond
    isolated = NetworkSwitch(obj_id=99)
    topology.add_node(isolated)
    assert ShortestPathRouting.get_path(topology, sw[1], isolated) == []


def test_flow_rule_matching():
    wildcard = FlowRule(match={}, action={"deliver": True})
    specific = FlowRule(match={"flow_id": 7}, action={"next_hop": "x"})

    assert wildcard.matches({"flow_id": 1}) is True  # empty match = wildcard
    assert specific.matches({"flow_id": 7}) is True
    assert specific.matches({"flow_id": 8}) is False


def test_controller_installs_and_forwards(diamond):
    topology, sw = diamond
    path = RoutingController.install_path(topology, sw[1], sw[4], policy=LeastCongestedRouting, packet={"flow_id": 7})
    assert _ids(path) == [1, 3, 4]

    # Every switch on the path got exactly one rule installed.
    assert all(len(switch.flow_table) == 1 for switch in path)

    # Following the installed rules reconstructs the same path in the data plane.
    forwarded = RoutingController.forward(sw[1], {"flow_id": 7})
    assert _ids(forwarded) == [1, 3, 4]


def test_forward_table_miss_returns_empty(diamond):
    topology, sw = diamond
    # No rules installed -> immediate table miss at the source.
    assert RoutingController.forward(sw[1], {"flow_id": 7}) == []


def test_priority_ordering(diamond):
    topology, sw = diamond
    sw[1].install_flow_rule(FlowRule(match={"flow_id": 7}, action={"deliver": "low"}, priority=1))
    sw[1].install_flow_rule(FlowRule(match={"flow_id": 7}, action={"deliver": "high"}, priority=5))

    # The higher-priority rule must win the lookup.
    assert sw[1].lookup_flow_rule({"flow_id": 7}).action["deliver"] == "high"


def test_topology_get_path_hook_defaults_and_overrides(diamond):
    topology, sw = diamond

    # Default (no policy set) -> delay-weighted shortest path.
    assert _ids(topology.get_path(sw[1], sw[4])) == [1, 2, 4]

    # Opt-in to a different policy.
    topology.routing_policy = LeastCongestedRouting
    assert _ids(topology.get_path(sw[1], sw[4])) == [1, 3, 4]
