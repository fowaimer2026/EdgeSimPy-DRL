"""Worked example: task execution, queueing, energy and failures.

Demonstrates the "consequences" layer that EdgeSimPy previously lacked: given a set of
tasks placed on edge servers, the execution engine simulates real queueing delays,
compute times, energy consumption and the impact of server failures.

Two studies are shown, neither needing a dataset:
    1. Placement strategies (least-loaded / round-robin / random) under contention,
       comparing response time, deadline violations and energy.
    2. The effect of server failures on the same workload.

Run with:  python examples/workload_execution_example.py
"""

# EdgeSimPy
from edge_sim_py.workload import (
    ExecutionEngine,
    random_workload,
    place_least_loaded,
    place_random,
    place_round_robin,
    collect_workload_metrics,
)
from edge_sim_py.energy import EnergyMeter
from edge_sim_py.reliability import FailureModel

# A contended workload: many tasks, few servers, so queueing actually happens.
NUM_TASKS = 60
NUM_SERVERS = 3
SEED = 7


def run(placer, failure_model=None):
    servers, tasks = random_workload(num_tasks=NUM_TASKS, num_servers=NUM_SERVERS, seed=SEED)
    placer(tasks, servers)
    meter = EnergyMeter(tick_duration=1.0)
    engine = ExecutionEngine(servers, tasks, energy_meter=meter, failure_model=failure_model)
    engine.run()
    metrics = collect_workload_metrics(tasks, engine)
    metrics["makespan"] = engine.current_step
    metrics["failures"] = len([event for event in engine.events if event["event"] == "failure"])
    return metrics


def main():
    print(f"Workload: {NUM_TASKS} tasks on {NUM_SERVERS} servers\n")

    print("Study 1 - placement strategies (no failures):")
    header = f"{'Strategy':<14}{'Resp.':>8}{'Wait':>8}{'DL viol.':>10}{'Energy':>10}{'Makespan':>10}"
    print(header)
    strategies = {
        "least-loaded": place_least_loaded,
        "round-robin": place_round_robin,
        "random": lambda tasks, servers: place_random(tasks, servers, seed=1),
    }
    for name, placer in strategies.items():
        m = run(placer)
        print(
            f"{name:<14}{m['mean_response_time']:>8.2f}{m['mean_waiting_time']:>8.2f}"
            f"{m['deadline_violation_rate']:>10.2f}{m['total_energy']:>10.0f}{m['makespan']:>10d}"
        )

    print("\nStudy 2 - impact of server failures (least-loaded placement):")
    print(f"{'Scenario':<14}{'Resp.':>8}{'DL viol.':>10}{'Done':>8}{'Failures':>10}{'Energy':>10}")
    baseline = run(place_least_loaded)
    print(
        f"{'no failures':<14}{baseline['mean_response_time']:>8.2f}{baseline['deadline_violation_rate']:>10.2f}"
        f"{baseline['completion_rate']:>8.2f}{baseline['failures']:>10d}{baseline['total_energy']:>10.0f}"
    )
    faulty = run(place_least_loaded, failure_model=FailureModel(mtbf=40, mttr=10, seed=3))
    print(
        f"{'with failures':<14}{faulty['mean_response_time']:>8.2f}{faulty['deadline_violation_rate']:>10.2f}"
        f"{faulty['completion_rate']:>8.2f}{faulty['failures']:>10d}{faulty['total_energy']:>10.0f}"
    )
    print(f"\nFailure model steady-state availability: {FailureModel.steady_state_availability(40, 10):.3f}")


if __name__ == "__main__":
    main()
