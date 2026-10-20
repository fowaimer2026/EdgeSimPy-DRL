"""Worked example: pluggable routing policies and programmable edge switches.

This script demonstrates the networking contribution that extends EdgeSimPy toward
the authors' stated future work -- "advanced routing policies that leverage the
programmability of network switches at the edge". It:

    1. Builds a small diamond topology where one branch is fast but congested and the
       other is slower but free.
    2. Compares the four routing policies (shortest-path, least-congested,
       latency-aware, ECMP) on the same source/target pair.
    3. Uses the SDN-style RoutingController to install match-action rules along the
       chosen path and then forwards a packet through the programmed switches.

It needs no dataset -- just run:

    python examples/routing_policies_example.py
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
    RoutingController,
)


def build_diamond_topology():
    """Builds a 4-switch diamond: a fast-but-congested branch and a slow-but-free one."""
    topology = Topology()
    switches = {i: NetworkSwitch(obj_id=i) for i in range(1, 5)}
    for switch in switches.values():
        topology.add_node(switch)

    def add_link(a, b, delay, bandwidth, demand):
        link = NetworkLink()
        link["topology"] = topology
        link["nodes"] = [switches[a], switches[b]]
        link["delay"] = delay
        link["bandwidth"] = bandwidth
        link["bandwidth_demand"] = demand
        topology.add_edge(switches[a], switches[b], **link)

    # Upper branch (s1-s2-s4): low delay, heavily loaded.
    add_link(1, 2, 1, 100, 95)
    add_link(2, 4, 1, 100, 95)
    # Lower branch (s1-s3-s4): higher delay, idle.
    add_link(1, 3, 3, 100, 0)
    add_link(3, 4, 3, 100, 0)

    return topology, switches


def path_ids(path):
    return [switch.id for switch in path]


def main():
    topology, sw = build_diamond_topology()
    source, target = sw[1], sw[4]

    print("Diamond topology: upper branch 1-2-4 (fast, congested), lower 1-3-4 (slow, free)\n")

    policies = [
        ("Shortest-path (delay)", ShortestPathRouting, None),
        ("Least-congested", LeastCongestedRouting, None),
        ("Latency-aware (a=0.5, b=0.5)", LatencyAwareRouting, {"alpha": 0.5, "beta": 0.5}),
        ("ECMP", ECMPRouting, None),
    ]

    print("Routing policy comparison (source s1 -> target s4):")
    for name, policy, params in policies:
        path = policy.get_path(topology, source, target, parameters=params)
        print(f"  {name:<30} -> {path_ids(path)}")

    print("\nProgrammable switches (SDN): installing least-congested rules for flow 7...")
    installed = RoutingController.install_path(topology, source, target, policy=LeastCongestedRouting, packet={"flow_id": 7})
    print(f"  Control plane installed rules along: {path_ids(installed)}")
    for switch in installed:
        rule = switch.lookup_flow_rule({"flow_id": 7})
        action = "deliver" if rule.action.get("deliver") else f"next_hop=s{rule.action['next_hop'].id}"
        print(f"    switch s{switch.id}: match flow_id=7 -> {action}")

    forwarded = RoutingController.forward(source, {"flow_id": 7})
    print(f"  Data plane forwarded packet through: {path_ids(forwarded)}")


if __name__ == "__main__":
    main()
