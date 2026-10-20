"""Task-placement helpers that decide which server runs each task.

Placement is the *decision* (where to offload); the :class:`ExecutionEngine` simulates the
*consequence* (queueing, compute time, energy). These helpers provide simple, comparable
placement strategies so experiments can isolate the effect of the decision policy on the
ground-truth execution metrics.
"""

# Python libraries
import random


def _free_cpu(server: object) -> float:
    return server.cpu - server.cpu_demand


def place_least_loaded(tasks: list, servers: list) -> None:
    """Assigns each task to the server with the most free CPU (greedy load balancing).

    The server demands are updated as tasks are assigned, so the load estimate reflects the
    accumulating assignment (a reservation view, not the live execution view).
    """
    for task in tasks:
        candidate = max(servers, key=_free_cpu)
        task.server = candidate
        candidate.cpu_demand += task.cpu_demand

    # Reset reservations so the engine starts from a clean slate.
    for server in servers:
        server.cpu_demand = 0


def place_random(tasks: list, servers: list, seed: int = None) -> None:
    """Assigns each task to a uniformly random server."""
    rng = random.Random(seed)
    for task in tasks:
        task.server = rng.choice(servers)


def place_round_robin(tasks: list, servers: list) -> None:
    """Assigns tasks to servers in round-robin order."""
    for index, task in enumerate(tasks):
        task.server = servers[index % len(servers)]
