"""Contains the Equal-Cost Multi-Path (ECMP) routing policy."""

# EdgeSimPy components
from edge_sim_py.components.routing.base_routing import compute_all_shortest_paths


class ECMPRouting:
    """Equal-Cost Multi-Path routing.

    When several paths share the minimum cost, ECMP spreads flows across them instead
    of always picking the same one, improving link utilization. The path is chosen by
    hashing a per-flow key (a flow-consistent decision, mirroring how programmable
    switches hash packet headers to select an output port), so all packets of a given
    flow follow the same path while different flows are balanced across equal-cost paths.
    """

    @classmethod
    def _flow_key(cls, source: object, target: object, flow: object) -> int:
        """Builds a deterministic hash key for selecting among equal-cost paths."""
        if flow is not None and getattr(flow, "id", None) is not None:
            return hash(flow.id)
        return hash((getattr(source, "id", id(source)), getattr(target, "id", id(target))))

    @classmethod
    def get_path(cls, topology: object, source: object, target: object, flow: object = None, parameters: dict = None) -> list:
        """Selects one of the equal-cost shortest paths between two switches.

        Args:
            topology (object): Network topology graph.
            source (object): Source network switch.
            target (object): Target network switch.
            flow (object, optional): Flow being routed; its id seeds the path choice.
            parameters (dict, optional): Policy parameters. Supports ``weight`` (edge
                attribute or callable used to define "equal cost"; default ``"delay"``).

        Returns:
            path (list): Ordered list of switches, or ``[]`` when no path exists.
        """
        params = parameters or {}
        weight = params.get("weight", "delay")

        paths = compute_all_shortest_paths(topology=topology, source=source, target=target, weight=weight)
        if not paths:
            return []

        # Hash a per-flow key onto the set of equal-cost paths for a consistent choice.
        chosen_index = cls._flow_key(source=source, target=target, flow=flow) % len(paths)
        return paths[chosen_index]
