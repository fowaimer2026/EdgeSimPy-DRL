"""Contains a reference cost-driven computation-offloading policy.

This module ships a ready-to-use ``resource_management_algorithm`` that decides
*where* each service should be offloaded by minimizing the weighted multi-objective
cost (latency + energy + money) provided by :class:`WeightedCostModel`. It is meant
both as a usable baseline and as a worked example of how to consume the cost models
inside an EdgeSimPy simulation loop.
"""

# EdgeSimPy components
from edge_sim_py.components.service import Service
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.cost_models.weighted_cost_model import WeightedCostModel


def _reference_user(service: object) -> object:
    """Picks a representative user for latency/energy estimation of a service.

    Args:
        service (object): Service being offloaded.

    Returns:
        user (object): A user accessing the service's application, or ``None``.
    """
    application = getattr(service, "application", None)
    users = getattr(application, "users", []) if application else []
    return users[0] if users else None


def min_cost_offloading(parameters: dict):
    """Cost-driven offloading policy executed at each simulation step.

    For every service that still needs to be placed (no host yet and not already
    being provisioned), the policy ranks the edge servers that have spare capacity
    using :meth:`WeightedCostModel.rank_servers` and offloads the service to the
    cheapest reachable one.

    Supported ``parameters`` keys:
        * ``cost_model_parameters`` (dict): forwarded to the weighted cost model
          (objective ``weights`` and the per-objective settings of the underlying
          latency/energy/monetary models).

    Args:
        parameters (dict): Resource-management algorithm parameters supplied by the
            :class:`~edge_sim_py.simulator.Simulator`.
    """
    cost_model_parameters = parameters.get("cost_model_parameters", {})

    for service in Service.all():
        # Skip services that are already hosted or mid-provisioning.
        if service.server is not None or service.being_provisioned:
            continue

        # Candidate servers must have enough free CPU/memory/disk to host the service.
        candidates = [server for server in EdgeServer.all() if server.has_capacity_to_host(service=service)]
        if not candidates:
            continue

        user = _reference_user(service=service)
        ranking = WeightedCostModel.rank_servers(
            service=service,
            servers=candidates,
            user=user,
            parameters=cost_model_parameters,
        )

        # The best candidate is the one with the lowest weighted cost and a finite score.
        best = ranking[0]
        if best["score"] == float("inf"):
            continue

        service.provision(target_server=best["server"])
