"""Contains helpers to aggregate computation-offloading metrics from a simulation.

These functions read the live state of the EdgeSimPy components (services, edge
servers and users) and summarize an offloading allocation into paper-ready figures:
placement rate, total/average cost, per-objective cost breakdown and SLA violations.

They are intentionally decoupled from the :class:`~edge_sim_py.simulator.Simulator`
internals so they can be called at any point -- inside a custom ``monitor`` hook, at
the end of ``run_model``, or interactively while exploring results.
"""

# EdgeSimPy components
from edge_sim_py.components.service import Service
from edge_sim_py.components.user import User
from edge_sim_py.components.cost_models.weighted_cost_model import WeightedCostModel


def _reference_user(service: object) -> object:
    """Returns a representative user of the service's application (or ``None``)."""
    application = getattr(service, "application", None)
    users = getattr(application, "users", []) if application else []
    return users[0] if users else None


def collect_cost_metrics(cost_model=WeightedCostModel, parameters: dict = None) -> dict:
    """Aggregates the cost of the current service-to-server allocation.

    Args:
        cost_model: Cost-model class exposing ``get_objectives``/``get_cost``.
            Defaults to :class:`WeightedCostModel`.
        parameters (dict, optional): Cost-model parameters. Defaults to None.

    Returns:
        metrics (dict): Placement rate and total/average cost (overall and per objective)
            computed across the services that are currently hosted.
    """
    services = Service.all()
    placed = [service for service in services if service.server is not None]

    objective_totals = {"latency": 0.0, "energy": 0.0, "monetary": 0.0}
    total_cost = 0.0

    for service in placed:
        user = _reference_user(service=service)

        # Per-objective totals (only available when the cost model exposes them).
        if hasattr(cost_model, "get_objectives"):
            objectives = cost_model.get_objectives(service, service.server, user, parameters)
            for key in objective_totals:
                value = objectives.get(key, 0.0)
                objective_totals[key] += value if value != float("inf") else 0.0

        cost = cost_model.get_cost(service, service.server, user, parameters)
        total_cost += cost if cost != float("inf") else 0.0

    placed_count = len(placed)
    return {
        "services_total": len(services),
        "services_placed": placed_count,
        "placement_rate": placed_count / len(services) if services else 0.0,
        "total_cost": total_cost,
        "average_cost": total_cost / placed_count if placed_count else 0.0,
        "objective_totals": objective_totals,
    }


def collect_sla_metrics() -> dict:
    """Aggregates delay-SLA satisfaction across all users and their applications.

    A violation is counted whenever a user's measured delay for an application exceeds
    the application's delay SLA (services not yet available report an infinite delay and
    therefore count as violations).

    Returns:
        metrics (dict): Number of user-application pairs evaluated, satisfied and
            violated, plus the resulting satisfaction rate.
    """
    evaluated = 0
    violations = 0

    for user in User.all():
        for app in user.applications:
            app_id = str(app.id)
            delay = user.delays.get(app_id)
            sla = user.delay_slas.get(app_id)

            # Skip pairs without an SLA or without a measured delay yet.
            if sla is None or delay is None:
                continue

            evaluated += 1
            if delay > sla:
                violations += 1

    satisfied = evaluated - violations
    return {
        "evaluated": evaluated,
        "satisfied": satisfied,
        "violations": violations,
        "satisfaction_rate": satisfied / evaluated if evaluated else 0.0,
    }


def collect_offloading_metrics(cost_model=WeightedCostModel, parameters: dict = None) -> dict:
    """Convenience wrapper returning both cost and SLA metrics in a single dict.

    Args:
        cost_model: Cost-model class used for the cost aggregation. Defaults to
            :class:`WeightedCostModel`.
        parameters (dict, optional): Cost-model parameters. Defaults to None.

    Returns:
        metrics (dict): ``{"cost": ..., "sla": ...}`` summary of the allocation.
    """
    return {
        "cost": collect_cost_metrics(cost_model=cost_model, parameters=parameters),
        "sla": collect_sla_metrics(),
    }
