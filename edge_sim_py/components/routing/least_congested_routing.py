"""Contains the least-congested (load-aware) routing policy."""

# EdgeSimPy components
from edge_sim_py.components.routing.base_routing import compute_path, link_utilization, EPSILON


class LeastCongestedRouting:
    """Load-aware routing that steers flows away from congested links.

    Each link's weight is its current bandwidth utilization
    (``bandwidth_demand / bandwidth``) plus a small floor, so Dijkstra prefers the
    path whose busiest links are least loaded. This is the kind of advanced,
    state-dependent forwarding decision that programmable edge switches enable but
    that a static shortest-path scheme cannot express.
    """

    @classmethod
    def get_path(cls, topology: object, source: object, target: object, flow: object = None, parameters: dict = None) -> list:
        """Computes the least-congested path between two switches.

        Args:
            topology (object): Network topology graph.
            source (object): Source network switch.
            target (object): Target network switch.
            flow (object, optional): Flow being routed (unused here). Defaults to None.
            parameters (dict, optional): Policy parameters. Supports ``hop_penalty``
                (default ``EPSILON``) added per link to break ties toward shorter paths.

        Returns:
            path (list): Ordered list of switches, or ``[]`` when no path exists.
        """
        params = parameters or {}
        hop_penalty = params.get("hop_penalty", EPSILON)

        def weight(_source, _target, link):
            return link_utilization(link) + hop_penalty

        return compute_path(topology=topology, source=source, target=target, weight=weight)
