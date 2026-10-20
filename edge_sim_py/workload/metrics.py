"""Aggregates ground-truth metrics from an execution run."""

# Python libraries
import numpy as np


def collect_workload_metrics(tasks: list, engine: object = None) -> dict:
    """Summarizes the outcome of a set of executed tasks.

    Args:
        tasks (list): Tasks after an :class:`ExecutionEngine` run.
        engine (ExecutionEngine, optional): If given, energy metrics are pulled from its
            energy meter. Defaults to None.

    Returns:
        metrics (dict): Completion/deadline rates and mean response/waiting/execution times,
            plus energy metrics when an energy meter is available.
    """
    total = len(tasks)
    finished = [task for task in tasks if task.finished]
    failed = [task for task in tasks if task.failed]

    response_times = [task.response_time for task in finished]
    waiting_times = [task.waiting_time for task in finished]
    execution_times = [task.execution_time for task in finished]

    with_deadline = [task for task in finished if task.deadline is not None]
    deadline_met = [task for task in with_deadline if task.met_deadline]

    metrics = {
        "tasks_total": total,
        "tasks_finished": len(finished),
        "tasks_failed": len(failed),
        "completion_rate": len(finished) / total if total else 0.0,
        "mean_response_time": float(np.mean(response_times)) if response_times else 0.0,
        "mean_waiting_time": float(np.mean(waiting_times)) if waiting_times else 0.0,
        "mean_execution_time": float(np.mean(execution_times)) if execution_times else 0.0,
        "deadline_violation_rate": (1.0 - len(deadline_met) / len(with_deadline) if with_deadline else 0.0),
    }

    if engine is not None and getattr(engine, "energy_meter", None) is not None:
        energy = engine.energy_meter.metrics()
        metrics["total_energy"] = energy["total_energy"]
        metrics["average_power"] = energy["average_power"]

    return metrics
