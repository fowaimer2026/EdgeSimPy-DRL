"""Contains baseline computation-offloading policies for benchmarking.

These simple policies share the exact interface of :func:`min_cost_offloading`
(a ``resource_management_algorithm`` invoked each simulation step), so they can be
swapped in to produce comparison baselines for the cost-driven policy:

    * ``latency_greedy_offloading`` -- always picks the lowest-latency reachable server;
    * ``random_offloading``         -- picks a random server with enough capacity.
"""

# EdgeSimPy components
from edge_sim_py.components.service import Service
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.cost_models.latency_cost_model import LatencyCostModel

# Python libraries
import random


def _reference_user(service: object) -> object:
    """Returns a representative user of the service's application (or ``None``)."""
    application = getattr(service, "application", None)
    users = getattr(application, "users", []) if application else []
    return users[0] if users else None


def _pending_services() -> list:
    """Services that still need to be placed (no host and not being provisioned)."""
    return [service for service in Service.all() if service.server is None and not service.being_provisioned]


def _candidate_servers(service: object) -> list:
    """Edge servers with enough free capacity to host the service."""
    return [server for server in EdgeServer.all() if server.has_capacity_to_host(service=service)]


def latency_greedy_offloading(parameters: dict):
    """Offloads each pending service to its lowest-latency reachable edge server.

    Args:
        parameters (dict): Resource-management parameters. Supports an optional
            ``cost_model_parameters`` dict forwarded to :class:`LatencyCostModel`.
    """
    cost_model_parameters = parameters.get("cost_model_parameters", {})

    for service in _pending_services():
        candidates = _candidate_servers(service=service)
        if not candidates:
            continue

        user = _reference_user(service=service)
        best = min(candidates, key=lambda server: LatencyCostModel.get_cost(service, server, user, cost_model_parameters))

        if LatencyCostModel.get_cost(service, best, user, cost_model_parameters) != float("inf"):
            service.provision(target_server=best)


def random_offloading(parameters: dict):
    """Offloads each pending service to a random edge server that can host it.

    Args:
        parameters (dict): Resource-management parameters. Supports an optional
            ``seed`` for reproducible random placement.
    """
    if parameters.get("seed") is not None:
        random.seed(parameters["seed"])

    for service in _pending_services():
        candidates = _candidate_servers(service=service)
        if not candidates:
            continue

        service.provision(target_server=random.choice(candidates))
