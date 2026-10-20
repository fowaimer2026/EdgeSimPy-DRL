"""Worked example: cost-driven computation offloading in EdgeSimPy.

This script demonstrates the computation-offloading contribution end to end:

    1. It defines a weighted multi-objective cost configuration (latency + energy +
       money) and feeds it to the reference ``min_cost_offloading`` policy.
    2. It runs an EdgeSimPy simulation that, at every step, offloads each pending
       service onto the cheapest reachable edge server according to that cost.
    3. It prints, per step, where each service was placed and the total cost of the
       resulting allocation.

Usage:
    python examples/offloading_cost_model_example.py --dataset path/to/dataset.json

The dataset is any standard EdgeSimPy input file (see the project README / docs for
how to generate one with the ``dataset_generator`` module). If you do not pass a
dataset, the script falls back to a small stand-alone demonstration of the cost
models that needs no input file at all.
"""

# Python libraries
import argparse

# EdgeSimPy
from edge_sim_py import Simulator, Service, EdgeServer
from edge_sim_py.components.cost_models.weighted_cost_model import WeightedCostModel
from edge_sim_py.offloading import min_cost_offloading

# Weighted multi-objective cost configuration shared by the policy and the reports.
COST_MODEL_PARAMETERS = {
    "weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2},
    "access_bandwidth": 100.0,
    "device_transmit_power": 1.5,
    "price_per_cpu_unit_time": 0.05,
}


def stopping_criterion(model: object) -> bool:
    """Stops once every service has been offloaded to some edge server."""
    return all(service.server is not None for service in Service.all())


def total_allocation_cost() -> float:
    """Sums the weighted cost of the current service-to-server allocation."""
    total = 0.0
    for service in Service.all():
        if service.server is not None:
            total += WeightedCostModel.get_cost(service, service.server, parameters=COST_MODEL_PARAMETERS)
    return total


def run_simulation(dataset: str) -> None:
    """Runs the cost-driven offloading simulation on a given dataset."""
    simulator = Simulator(
        dump_interval=float("inf"),
        tick_duration=1,
        tick_unit="seconds",
        stopping_criterion=stopping_criterion,
        resource_management_algorithm=min_cost_offloading,
        resource_management_algorithm_parameters={"cost_model_parameters": COST_MODEL_PARAMETERS},
    )
    simulator.initialize(input_file=dataset)
    simulator.run_model()

    print("\n=== Final cost-driven offloading allocation ===")
    for service in Service.all():
        host = service.server.id if service.server else "UNPLACED"
        print(f"  Service {service.id:>3} -> EdgeServer {host}")
    print(f"\nTotal weighted allocation cost: {total_allocation_cost():.4f}")


def run_standalone_demo() -> None:
    """Demonstrates the cost models without a dataset, ranking a few servers by cost."""
    print("No dataset supplied -- running a stand-alone cost-model demonstration.\n")

    service = Service(label="demo-service", cpu_demand=2, memory_demand=512, state=50)
    servers = [
        EdgeServer(model_name="edge-A", cpu=8, memory=4096, disk=10000),
        EdgeServer(model_name="edge-B", cpu=4, memory=2048, disk=10000),
        EdgeServer(model_name="edge-C", cpu=16, memory=8192, disk=10000),
    ]
    servers[1].cpu_demand = 3  # edge-B is partially loaded

    ranking = WeightedCostModel.rank_servers(service, servers, parameters=COST_MODEL_PARAMETERS)

    print(f"Offloading decision for service {service.id} (cpu_demand={service.cpu_demand}):")
    for rank, entry in enumerate(ranking, start=1):
        objectives = entry["objectives"]
        print(
            f"  #{rank} {entry['server'].model_name}: score={entry['score']:.4f} "
            f"(latency={objectives['latency']:.3f}, energy={objectives['energy']:.3f}, "
            f"money={objectives['monetary']:.3f})"
        )
    print(f"\nBest offloading target: {ranking[0]['server'].model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost-driven computation-offloading example.")
    parser.add_argument("--dataset", help="Path to an EdgeSimPy dataset JSON file.", default=None)
    args = parser.parse_args()

    if args.dataset:
        run_simulation(dataset=args.dataset)
    else:
        run_standalone_demo()


if __name__ == "__main__":
    main()
