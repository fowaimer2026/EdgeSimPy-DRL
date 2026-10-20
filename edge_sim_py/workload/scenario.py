"""Synthetic workload generation for the execution engine.

Builds a set of edge servers and a stream of tasks with randomized arrival times, compute
sizes and deadlines, for experimenting with the :class:`ExecutionEngine` without needing a
full dataset. As with the DRL scenarios, the EdgeServer/Task registries are reset on each
call to keep memory bounded across repeated runs.
"""

# EdgeSimPy components
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.power_models.servers.linear_server_power_model import LinearServerPowerModel
from edge_sim_py.workload.task import Task

# Python libraries
import random


def reset_workload_registries() -> None:
    """Clears the EdgeServer/Task registries (used between runs)."""
    for cls in (EdgeServer, Task):
        cls._instances = []
        cls._object_count = 0


def random_workload(
    num_tasks: int,
    num_servers: int,
    seed: int = None,
    horizon: int = 50,
    cpu_capacity_range: tuple = (8, 16),
    cpu_cycles_range: tuple = (5, 30),
    cpu_demand_range: tuple = (1, 4),
    memory_demand_range: tuple = (128, 1024),
    deadline_slack_range: tuple = (5, 40),
    attach_power_model: bool = True,
) -> tuple:
    """Builds a randomized (servers, tasks) workload scenario.

    Args:
        num_tasks (int): Number of tasks to generate.
        num_servers (int): Number of edge servers.
        seed (int, optional): RNG seed. Defaults to None.
        horizon (int, optional): Tasks arrive uniformly within ``[0, horizon)``. Defaults to 50.
        cpu_capacity_range (tuple): Min/max server CPU capacity.
        cpu_cycles_range (tuple): Min/max task compute size.
        cpu_demand_range (tuple): Min/max CPU units a task occupies while running.
        memory_demand_range (tuple): Min/max task memory demand.
        deadline_slack_range (tuple): Extra steps added to the arrival time for the deadline.
        attach_power_model (bool, optional): Attach a linear power model to each server so the
            energy meter produces non-trivial numbers. Defaults to True.

    Returns:
        (servers, tasks) (tuple): Fresh EdgeServer and Task lists.
    """
    rng = random.Random(seed)
    reset_workload_registries()

    servers = []
    for _ in range(num_servers):
        server = EdgeServer(cpu=rng.randint(*cpu_capacity_range), memory=32_768, disk=10_000)
        if attach_power_model:
            server.power_model = LinearServerPowerModel
            server.power_model_parameters = {"max_power_consumption": 200, "static_power_percentage": 0.4}
        servers.append(server)

    tasks = []
    for _ in range(num_tasks):
        arrival = rng.randint(0, max(0, horizon - 1))
        cpu_demand = rng.randint(*cpu_demand_range)
        cpu_cycles = rng.randint(*cpu_cycles_range)
        # Deadline is the arrival plus the (single-server) execution time plus random slack.
        nominal_execution = cpu_cycles / cpu_demand
        deadline = int(arrival + nominal_execution + rng.randint(*deadline_slack_range))
        tasks.append(
            Task(
                cpu_cycles=cpu_cycles,
                cpu_demand=cpu_demand,
                memory_demand=rng.randint(*memory_demand_range),
                input_size=rng.randint(1, 100),
                arrival_step=arrival,
                deadline=deadline,
            )
        )

    return servers, tasks
