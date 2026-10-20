"""Synthetic scenario generation for training the DRL offloading agents.

Training a placement agent benefits from many small, varied infrastructures. This
module builds randomized scenarios -- a set of pending :class:`Service` objects and a
set of :class:`EdgeServer` objects with heterogeneous capacities -- that the
:class:`~edge_sim_py.drl.offloading_env.OffloadingEnv` steps through.

The cost models used as the reward signal only read plain attributes
(``cpu``/``cpu_demand``/``memory``/``state`` ...), so the scenarios need no topology,
container images or network flows. Because EdgeSimPy components register themselves in
class-level registries, ``random_scenario`` resets the ``Service``/``EdgeServer``
registries on each call to keep memory bounded across thousands of training episodes.
"""

# EdgeSimPy components
from edge_sim_py.components.service import Service
from edge_sim_py.components.edge_server import EdgeServer

# Python libraries
import random


def reset_component_registries() -> None:
    """Clears the Service/EdgeServer class registries (used between training episodes)."""
    for cls in (Service, EdgeServer):
        cls._instances = []
        cls._object_count = 0


def random_scenario(
    num_services: int,
    num_servers: int,
    seed: int = None,
    cpu_capacity_range: tuple = (8, 32),
    memory_capacity_range: tuple = (4096, 16384),
    cpu_demand_range: tuple = (1, 8),
    memory_demand_range: tuple = (256, 2048),
    state_range: tuple = (1, 100),
) -> tuple:
    """Builds a randomized placement scenario.

    Args:
        num_services (int): Number of services to place.
        num_servers (int): Number of candidate edge servers (the action space size).
        seed (int, optional): Seed for reproducibility. Defaults to None.
        cpu_capacity_range (tuple): Min/max server CPU capacity.
        memory_capacity_range (tuple): Min/max server memory capacity.
        cpu_demand_range (tuple): Min/max service CPU demand.
        memory_demand_range (tuple): Min/max service memory demand.
        state_range (tuple): Min/max service state size (offloaded payload proxy).

    Returns:
        (services, servers) (tuple): Lists of fresh Service and EdgeServer objects.
    """
    rng = random.Random(seed)
    reset_component_registries()

    servers = []
    for _ in range(num_servers):
        server = EdgeServer(
            cpu=rng.randint(*cpu_capacity_range),
            memory=rng.randint(*memory_capacity_range),
            disk=10_000,
        )
        servers.append(server)

    services = []
    for _ in range(num_services):
        service = Service(
            cpu_demand=rng.randint(*cpu_demand_range),
            memory_demand=rng.randint(*memory_demand_range),
            state=rng.randint(*state_range),
        )
        services.append(service)

    return services, servers
