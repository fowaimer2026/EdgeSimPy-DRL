"""Contains shared helpers used by the computation-offloading cost models.

These helpers translate the raw state already tracked by EdgeSimPy components
(services, edge servers, users, links and the network topology) into the
elementary terms that recurring offloading cost formulations rely on:

    * transmission delay  -- time to push a service's input data over the network;
    * propagation delay   -- accumulated link delay along the user <-> server path;
    * processing delay     -- time the target server needs to run the workload.

Keeping these in a single place lets every concrete cost model (latency, energy,
monetary, weighted) share the exact same network/compute assumptions.
"""

# EdgeSimPy components
from edge_sim_py.components.topology import Topology
from edge_sim_py.components.network_switch import NetworkSwitch

# Python libraries
import networkx as nx

# A tiny positive number used to keep divisions well-defined when a candidate
# server is fully booked (free capacity == 0) or a link advertises no bandwidth.
EPSILON = 1e-9


def resolve_input_data_size(service: object, parameters: dict) -> float:
    """Determines the amount of data (in Mbit) that must be offloaded for a service.

    The value is taken from ``parameters["input_data_size"]`` when provided,
    otherwise it falls back to the service's migratable ``state`` (a proxy for the
    payload that has to travel to the target server) and finally to a unit default.

    Args:
        service (object): Service whose payload size is being resolved.
        parameters (dict): Cost-model parameters.

    Returns:
        data_size (float): Input data size, in Mbit.
    """
    if parameters.get("input_data_size") is not None:
        return float(parameters["input_data_size"])
    if getattr(service, "state", 0):
        return float(service.state)
    return 1.0


def transmission_delay(data_size: float, bandwidth: float) -> float:
    """Computes the time needed to transfer ``data_size`` Mbit over a ``bandwidth`` Mbps channel.

    Args:
        data_size (float): Data to transfer, in Mbit.
        bandwidth (float): Available bandwidth, in Mbps.

    Returns:
        delay (float): Transmission delay (same time unit used across the simulation).
    """
    return data_size / max(bandwidth, EPSILON)


def network_path_delay(user: object, server: object) -> float:
    """Computes the propagation delay between a user and a candidate edge server.

    The delay is the wireless access delay of the user's base station plus the
    accumulated link delay along the shortest network path (weighted by ``delay``)
    connecting the user's base station to the server's base station. When no user
    is supplied or no path exists, the propagation delay is treated as zero so the
    cost model still returns a finite, comparable value.

    Args:
        user (object): User offloading the workload (may be ``None``).
        server (object): Candidate edge server hosting the workload.

    Returns:
        delay (float): Propagation delay between the user and the server.
    """
    if user is None or getattr(user, "base_station", None) is None:
        return 0.0
    if getattr(server, "base_station", None) is None:
        return 0.0

    delay = user.base_station.wireless_delay or 0.0

    # Co-located user and server: only the wireless access delay applies.
    if user.base_station.network_switch == server.base_station.network_switch:
        return delay

    topology = Topology.first()
    try:
        path = nx.shortest_path(
            G=topology,
            source=user.base_station.network_switch,
            target=server.base_station.network_switch,
            weight="delay",
            method="dijkstra",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")

    delay += topology.calculate_path_delay(path=[NetworkSwitch.find_by_id(node.id) for node in path])
    return delay


def processing_delay(service: object, server: object, parameters: dict) -> float:
    """Estimates the time a server needs to execute a service's workload.

    The workload is approximated by the service's CPU demand scaled by
    ``parameters["cpu_cycles_per_unit"]`` and divided by the server's *free* CPU
    capacity, capturing the slowdown of offloading onto an already loaded server.

    Args:
        service (object): Service being offloaded.
        server (object): Candidate edge server.
        parameters (dict): Cost-model parameters.

    Returns:
        delay (float): Processing delay on the target server.
    """
    cpu_cycles_per_unit = parameters.get("cpu_cycles_per_unit", 1.0)
    workload = service.cpu_demand * cpu_cycles_per_unit

    free_cpu = getattr(server, "cpu", 0) - getattr(server, "cpu_demand", 0)
    return workload / max(free_cpu, EPSILON)
