"""Contains shared helpers used by the pluggable routing policies.

EdgeSimPy historically selected network paths with a hardcoded delay-weighted
``networkx.shortest_path`` call. This package generalizes that into a pluggable
routing layer so that *advanced* policies -- the kind enabled by programmable edge
switches -- can be experimented with (least-congested, latency-aware load balancing,
ECMP, ...), as envisioned in the EdgeSimPy future-work agenda.

A routing policy is a class exposing::

    get_path(topology, source, target, flow=None, parameters=None) -> list

where ``source``/``target`` are :class:`NetworkSwitch` nodes of the topology graph
and the returned value is the ordered list of switches forming the path (empty when
no path exists). The helpers below expose the per-link quantities (delay, bandwidth
utilization) that policies combine into their edge weights.
"""

# Python libraries
import networkx as nx

# Small positive constant keeping edge weights strictly positive (Dijkstra requires
# non-negative weights and benefits from a tie-breaking floor).
EPSILON = 1e-9


def link_utilization(link: dict) -> float:
    """Returns the bandwidth utilization (in ``[0, 1+]``) of a network link.

    Args:
        link (dict): NetworkLink object (a dict carrying ``bandwidth``/``bandwidth_demand``).

    Returns:
        utilization (float): ``bandwidth_demand / bandwidth`` (0 when capacity is unknown).
    """
    bandwidth = link.get("bandwidth", 0) or 0
    if bandwidth <= 0:
        return 0.0
    return (link.get("bandwidth_demand", 0) or 0) / bandwidth


def max_link_delay(topology: object) -> float:
    """Returns the largest link delay in the topology (used to normalize delays).

    Args:
        topology (object): Network topology graph.

    Returns:
        max_delay (float): Maximum link delay, or 1.0 when the topology has no delays.
    """
    delays = [data.get("delay", 0) or 0 for _, _, data in topology.edges(data=True)]
    maximum = max(delays) if delays else 0
    return float(maximum) if maximum > 0 else 1.0


def compute_path(topology: object, source: object, target: object, weight) -> list:
    """Computes a single least-weight path between two switches.

    Args:
        topology (object): Network topology graph.
        source (object): Source network switch.
        target (object): Target network switch.
        weight (str | callable): Edge weight passed to NetworkX (attribute name or a
            ``f(u, v, edge_data) -> number`` callable).

    Returns:
        path (list): Ordered list of switches, or ``[]`` when no path exists.
    """
    if source == target:
        return [source]
    try:
        return nx.shortest_path(G=topology, source=source, target=target, weight=weight, method="dijkstra")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def compute_all_shortest_paths(topology: object, source: object, target: object, weight) -> list:
    """Computes every least-weight (equal-cost) path between two switches.

    Args:
        topology (object): Network topology graph.
        source (object): Source network switch.
        target (object): Target network switch.
        weight (str | callable): Edge weight passed to NetworkX.

    Returns:
        paths (list): List of equal-cost paths (each an ordered list of switches);
            ``[[source]]`` when source and target coincide and ``[]`` when unreachable.
    """
    if source == target:
        return [[source]]
    try:
        return list(nx.all_shortest_paths(G=topology, source=source, target=target, weight=weight, method="dijkstra"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
