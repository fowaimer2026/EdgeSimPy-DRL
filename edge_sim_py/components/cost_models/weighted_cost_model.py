"""Contains a weighted multi-objective computation-offloading cost model definition."""

# EdgeSimPy components
from edge_sim_py.components.cost_models.latency_cost_model import LatencyCostModel
from edge_sim_py.components.cost_models.energy_cost_model import EnergyCostModel
from edge_sim_py.components.cost_models.monetary_cost_model import MonetaryCostModel
from edge_sim_py.components.cost_models.base_cost_model import EPSILON


class WeightedCostModel:
    """Weighted multi-objective offloading cost combining latency, energy and money.

    Each objective lives on a different scale (milliseconds, joules, currency), so a
    naive sum would be dominated by whichever has the largest magnitude. This model
    therefore exposes two complementary entry points:

        * ``get_cost`` -- a weighted sum using fixed normalization factors supplied in
          ``parameters`` (``latency_norm``, ``energy_norm``, ``monetary_norm``), useful
          when absolute, comparable-across-steps costs are needed.
        * ``rank_servers`` -- min-max normalizes every objective across a *set* of
          candidate servers before weighting, yielding scale-free scores in [0, 1] that
          are ideal for picking the best offloading target at a given decision point.

    The objective weights default to an equal split and are renormalized to sum to 1,
    so callers may pass unnormalized weights (e.g. ``{"latency": 2, "energy": 1, ...}``).
    """

    DEFAULT_PARAMETERS = {
        # Relative importance of each objective (renormalized to sum to 1 internally).
        "weights": {"latency": 1.0, "energy": 1.0, "monetary": 1.0},
        # Fixed normalization factors used by ``get_cost`` (ignored by ``rank_servers``).
        "latency_norm": 1.0,
        "energy_norm": 1.0,
        "monetary_norm": 1.0,
    }

    @classmethod
    def _normalized_weights(cls, parameters: dict) -> dict:
        """Returns the objective weights renormalized so they sum to 1."""
        weights = {**cls.DEFAULT_PARAMETERS["weights"], **(parameters.get("weights") or {})}
        total = sum(weights.values())
        if total <= 0:
            # Degenerate configuration: fall back to an equal split.
            return {key: 1.0 / len(weights) for key in weights}
        return {key: value / total for key, value in weights.items()}

    @classmethod
    def get_objectives(cls, service: object, server: object, user: object = None, parameters: dict = None) -> dict:
        """Computes the three raw objective values for a (service, server) pair.

        Args:
            service (object): Service to be offloaded.
            server (object): Candidate edge server.
            user (object, optional): User accessing the service. Defaults to None.
            parameters (dict, optional): Cost-model parameters. Defaults to None.

        Returns:
            objectives (dict): Raw latency, energy and monetary costs.
        """
        params = parameters or {}
        return {
            "latency": LatencyCostModel.get_cost(service, server, user, params),
            "energy": EnergyCostModel.get_cost(service, server, user, params),
            "monetary": MonetaryCostModel.get_cost(service, server, user, params),
        }

    @classmethod
    def get_cost(cls, service: object, server: object, user: object = None, parameters: dict = None) -> float:
        """Computes the weighted offloading cost using fixed normalization factors.

        Args:
            service (object): Service to be offloaded.
            server (object): Candidate edge server.
            user (object, optional): User accessing the service. Defaults to None.
            parameters (dict, optional): Cost-model parameters. Defaults to None.

        Returns:
            cost (float): Weighted multi-objective cost.
        """
        params = {**cls.DEFAULT_PARAMETERS, **(parameters or {})}
        weights = cls._normalized_weights(params)
        objectives = cls.get_objectives(service, server, user, params)

        return (
            weights["latency"] * (objectives["latency"] / max(params["latency_norm"], EPSILON))
            + weights["energy"] * (objectives["energy"] / max(params["energy_norm"], EPSILON))
            + weights["monetary"] * (objectives["monetary"] / max(params["monetary_norm"], EPSILON))
        )

    @classmethod
    def rank_servers(cls, service: object, servers: list, user: object = None, parameters: dict = None) -> list:
        """Ranks candidate servers by min-max-normalized weighted cost (lower is better).

        Each objective is normalized across the supplied ``servers`` so the three
        scales become comparable, then combined with the configured weights. Servers
        whose objectives are infinite (e.g. unreachable) receive an infinite score and
        sink to the bottom of the ranking.

        Args:
            service (object): Service to be offloaded.
            servers (list): Candidate edge servers to rank.
            user (object, optional): User accessing the service. Defaults to None.
            parameters (dict, optional): Cost-model parameters. Defaults to None.

        Returns:
            ranking (list): List of ``{"server", "score", "objectives"}`` dicts sorted
                by ascending score.
        """
        params = {**cls.DEFAULT_PARAMETERS, **(parameters or {})}
        weights = cls._normalized_weights(params)

        # Step 1: gather the raw objectives for every candidate server.
        raw = [{"server": server, "objectives": cls.get_objectives(service, server, user, params)} for server in servers]

        # Step 2: per-objective min-max bounds, ignoring non-finite values.
        bounds = {}
        for objective in ("latency", "energy", "monetary"):
            finite_values = [entry["objectives"][objective] for entry in raw if entry["objectives"][objective] != float("inf")]
            bounds[objective] = (min(finite_values), max(finite_values)) if finite_values else (0.0, 0.0)

        # Step 3: normalize each objective to [0, 1] and combine using the weights.
        ranking = []
        for entry in raw:
            score = 0.0
            for objective in ("latency", "energy", "monetary"):
                value = entry["objectives"][objective]
                if value == float("inf"):
                    score = float("inf")
                    break
                low, high = bounds[objective]
                normalized = 0.0 if high - low <= EPSILON else (value - low) / (high - low)
                score += weights[objective] * normalized
            ranking.append({"server": entry["server"], "score": score, "objectives": entry["objectives"]})

        return sorted(ranking, key=lambda entry: entry["score"])
