"""Contains the latency-aware load-balanced routing policy."""

# EdgeSimPy components
from edge_sim_py.components.routing.base_routing import compute_path, link_utilization, max_link_delay


class LatencyAwareRouting:
    """Latency-aware load-balanced routing.

    Each link weight is a tunable convex combination of its *normalized* delay and
    its bandwidth utilization::

        weight = alpha * (delay / max_delay) + beta * utilization

    With ``alpha = 1, beta = 0`` it degenerates to delay-weighted shortest path; with
    ``alpha = 0, beta = 1`` it becomes least-congested routing. Intermediate settings
    trade end-to-end latency against congestion avoidance -- a balance that
    programmable edge switches make practical to enforce per-flow.
    """

    @classmethod
    def get_path(cls, topology: object, source: object, target: object, flow: object = None, parameters: dict = None) -> list:
        """Computes a latency-aware, load-balanced path between two switches.

        Args:
            topology (object): Network topology graph.
            source (object): Source network switch.
            target (object): Target network switch.
            flow (object, optional): Flow being routed (unused here). Defaults to None.
            parameters (dict, optional): Policy parameters. Supports ``alpha`` (delay
                weight, default 0.5) and ``beta`` (utilization weight, default 0.5).

        Returns:
            path (list): Ordered list of switches, or ``[]`` when no path exists.
        """
        params = parameters or {}
        alpha = params.get("alpha", 0.5)
        beta = params.get("beta", 0.5)

        # Normalize delays so they are comparable to utilization (both in ~[0, 1]).
        delay_reference = max_link_delay(topology=topology)

        def weight(_source, _target, link):
            normalized_delay = (link.get("delay", 0) or 0) / delay_reference
            return alpha * normalized_delay + beta * link_utilization(link)

        return compute_path(topology=topology, source=source, target=target, weight=weight)
