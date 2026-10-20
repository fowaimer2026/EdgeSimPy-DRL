"""Contains the delay-weighted shortest-path routing policy."""

# EdgeSimPy components
from edge_sim_py.components.routing.base_routing import compute_path


class ShortestPathRouting:
    """Delay-weighted shortest-path routing.

    This policy reproduces EdgeSimPy's historical routing behavior (a Dijkstra
    shortest path weighted by each link's ``delay``), and therefore acts as the
    backward-compatible default of the pluggable routing layer.
    """

    @classmethod
    def get_path(cls, topology: object, source: object, target: object, flow: object = None, parameters: dict = None) -> list:
        """Computes the lowest-latency path between two switches.

        Args:
            topology (object): Network topology graph.
            source (object): Source network switch.
            target (object): Target network switch.
            flow (object, optional): Flow being routed (unused here). Defaults to None.
            parameters (dict, optional): Policy parameters (unused here). Defaults to None.

        Returns:
            path (list): Ordered list of switches, or ``[]`` when no path exists.
        """
        return compute_path(topology=topology, source=source, target=target, weight="delay")
