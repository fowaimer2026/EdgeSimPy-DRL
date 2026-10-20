"""A self-contained RL environment for the service-offloading placement problem.

``OffloadingEnv`` frames computation offloading as a short-horizon Markov Decision
Process: at each step the agent observes the next pending service together with the
state of every candidate edge server and chooses one server to host it. Placing a
service consumes capacity, so earlier decisions change the options available to later
ones -- a genuine sequential decision problem rather than a one-shot bandit.

    * State  -- per-service demand features + per-server features (free capacity and the
      min-max-normalized latency/energy/monetary costs of placing the service there).
    * Action -- index of the chosen edge server (with an action mask for servers that
      lack capacity).
    * Reward -- negative normalized weighted cost of the chosen placement (in ``[-1, 0]``),
      reusing :class:`WeightedCostModel`; an invalid (no-capacity) action is penalized.

The environment exposes a tiny gym-like API (``reset``/``step``) and needs no external
RL dependency.
"""

# EdgeSimPy components
from edge_sim_py.components.cost_models.weighted_cost_model import WeightedCostModel
from edge_sim_py.components.cost_models.base_cost_model import EPSILON

# Python libraries
import numpy as np

# Number of features emitted per candidate server in the state vector.
SERVER_FEATURES = 6
# Number of features describing the service awaiting placement.
SERVICE_FEATURES = 3


class OffloadingEnv:
    """Service-placement environment driven by the weighted cost model."""

    def __init__(self, services: list, servers: list, cost_parameters: dict = None, invalid_penalty: float = 2.0):
        """Creates the environment.

        Args:
            services (list): Services to place (the episode steps through them in order).
            servers (list): Candidate edge servers (defines the fixed action space size).
            cost_parameters (dict, optional): Parameters forwarded to the cost models
                (objective ``weights`` etc.). Defaults to None.
            invalid_penalty (float, optional): Reward penalty for choosing a server without
                capacity. Defaults to 2.0.
        """
        self.services = services
        self.servers = servers
        self.cost_parameters = cost_parameters or {}
        self.invalid_penalty = invalid_penalty

        self.num_actions = len(servers)
        self.state_dim = SERVICE_FEATURES + self.num_actions * SERVER_FEATURES

        # Capacity scales used to normalize service/server features.
        self._cpu_scale = max((server.cpu for server in servers), default=1) or 1
        self._memory_scale = max((server.memory for server in servers), default=1) or 1
        self._state_scale = max((getattr(service, "state", 0) or 0 for service in services), default=1) or 1

        self.pointer = 0
        self.placed = 0
        self.action_mask = np.ones(self.num_actions, dtype=bool)
        self._scores = np.zeros(self.num_actions, dtype=np.float32)

    # ------------------------------------------------------------------ helpers

    def _has_capacity(self, server: object, service: object) -> bool:
        """Returns whether a server can still host a service under the simple model."""
        free_cpu = server.cpu - server.cpu_demand
        free_memory = server.memory - server.memory_demand
        return free_cpu >= service.cpu_demand and free_memory >= service.memory_demand

    def _normalized_weights(self) -> dict:
        """Returns the (renormalized) objective weights of the weighted cost model."""
        return WeightedCostModel._normalized_weights({**WeightedCostModel.DEFAULT_PARAMETERS, **self.cost_parameters})

    def _compute(self, service: object) -> tuple:
        """Computes per-server normalized objectives and weighted scores for a service.

        Returns:
            (normalized, scores) (tuple): ``normalized`` is a list of per-server dicts with
                latency/energy/monetary in ``[0, 1]``; ``scores`` is the weighted combination.
        """
        weights = self._normalized_weights()
        objectives = [WeightedCostModel.get_objectives(service, server, None, self.cost_parameters) for server in self.servers]

        normalized = [{} for _ in self.servers]
        for key in ("latency", "energy", "monetary"):
            finite = [obj[key] for obj in objectives if obj[key] != float("inf")]
            low, high = (min(finite), max(finite)) if finite else (0.0, 0.0)
            span = high - low
            for index, obj in enumerate(objectives):
                value = obj[key]
                normalized[index][key] = 1.0 if value == float("inf") else (0.0 if span <= EPSILON else (value - low) / span)

        scores = np.array(
            [sum(weights[key] * normalized[index][key] for key in weights) for index in range(len(self.servers))],
            dtype=np.float32,
        )
        return normalized, scores

    def _encode(self, service: object) -> np.ndarray:
        """Builds the state vector for the service currently awaiting placement."""
        normalized, scores = self._compute(service)
        self._scores = scores

        features = [
            service.cpu_demand / self._cpu_scale,
            service.memory_demand / self._memory_scale,
            (getattr(service, "state", 0) or 0) / self._state_scale,
        ]

        mask = np.zeros(self.num_actions, dtype=bool)
        for index, server in enumerate(self.servers):
            free_cpu_ratio = (server.cpu - server.cpu_demand) / max(server.cpu, EPSILON)
            free_memory_ratio = (server.memory - server.memory_demand) / max(server.memory, EPSILON)
            has_capacity = self._has_capacity(server, service)
            mask[index] = has_capacity
            features.extend(
                [
                    free_cpu_ratio,
                    free_memory_ratio,
                    1.0 if has_capacity else 0.0,
                    normalized[index]["latency"],
                    normalized[index]["energy"],
                    normalized[index]["monetary"],
                ]
            )

        # If no server has capacity, unmask everything so the agent is not stuck (the
        # step will still apply the invalid-action penalty).
        if not mask.any():
            mask[:] = True
        self.action_mask = mask

        return np.asarray(features, dtype=np.float32)

    def observe(self, service: object) -> tuple:
        """Encodes an arbitrary service against the current server state (read-only).

        Useful for running a trained agent inside a live simulation without stepping the
        environment: returns the state vector and the action mask for ``service`` given
        the servers' current load.

        Args:
            service (object): Service to encode.

        Returns:
            (state, mask) (tuple): State vector and boolean action mask.
        """
        state = self._encode(service)
        return state, self.action_mask.copy()

    # ------------------------------------------------------------------ gym-like API

    def reset(self) -> np.ndarray:
        """Resets server load and the service pointer, returning the first state."""
        for server in self.servers:
            server.cpu_demand = 0
            server.memory_demand = 0
        for service in self.services:
            service.server = None

        self.pointer = 0
        self.placed = 0
        return self._encode(self.services[self.pointer])

    def step(self, action: int) -> tuple:
        """Places the current service on the chosen server and advances the episode.

        Args:
            action (int): Index of the selected edge server.

        Returns:
            (next_state, reward, done, info) (tuple): Standard RL transition. ``next_state``
                is ``None`` on the terminal step.
        """
        service = self.services[self.pointer]
        server = self.servers[action]
        valid = self._has_capacity(server, service)

        if valid:
            server.cpu_demand += service.cpu_demand
            server.memory_demand += service.memory_demand
            service.server = server
            score = float(self._scores[action])
            reward = -score
            self.placed += 1
        else:
            reward = -self.invalid_penalty

        self.pointer += 1
        done = self.pointer >= len(self.services)
        next_state = None if done else self._encode(self.services[self.pointer])
        info = {"valid": valid, "placed": self.placed, "action_mask": self.action_mask.copy()}
        return next_state, reward, done, info

    # ------------------------------------------------------------------ baselines

    def greedy_action(self) -> int:
        """Returns the lowest-cost valid action for the current service (min-cost baseline)."""
        masked_scores = np.where(self.action_mask, self._scores, np.inf)
        return int(np.argmin(masked_scores))

    def random_action(self, rng=None) -> int:
        """Returns a random valid action for the current service (random baseline)."""
        valid_actions = np.flatnonzero(self.action_mask)
        if rng is not None:
            return int(rng.choice(valid_actions))
        return int(np.random.choice(valid_actions))
