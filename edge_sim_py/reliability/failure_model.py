"""A pluggable MTBF/MTTR failure-and-recovery model for edge servers.

Out of the box an :class:`EdgeServer`'s ``available`` flag is set once and never changes,
so EdgeSimPy cannot study fault tolerance. ``FailureModel`` adds stochastic failures and
repairs driven by the classic reliability parameters:

    * MTBF -- Mean Time Between Failures (how long a healthy server runs on average);
    * MTTR -- Mean Time To Repair (how long a failed server takes to recover).

Each tick, a healthy server fails with probability ``tick_duration / MTBF`` and a failed
server recovers with probability ``tick_duration / MTTR`` (the discrete-time analogue of
exponential lifetimes). A failed server is marked both unavailable and inactive, so the
power models report zero draw and the execution engine stops scheduling tasks on it.
"""

# Python libraries
import random


class FailureModel:
    """Stochastic failure/recovery process toggling ``available``/``active`` on servers."""

    def __init__(self, mtbf: float, mttr: float, tick_duration: float = 1.0, seed: int = None):
        """Creates a failure model.

        Args:
            mtbf (float): Mean time between failures (in ticks*tick_duration units). Must be > 0.
            mttr (float): Mean time to repair. Must be > 0.
            tick_duration (float, optional): Duration of one tick. Defaults to 1.0.
            seed (int, optional): RNG seed for reproducibility. Defaults to None.
        """
        if mtbf <= 0 or mttr <= 0:
            raise ValueError("MTBF and MTTR must be positive.")
        self.mtbf = mtbf
        self.mttr = mttr
        self.tick_duration = tick_duration
        self.rng = random.Random(seed)

        # Per-tick transition probabilities (clamped to [0, 1]).
        self.failure_probability = min(1.0, tick_duration / mtbf)
        self.recovery_probability = min(1.0, tick_duration / mttr)

    def fail(self, server: object) -> None:
        """Marks a server as failed (unavailable and inactive)."""
        server.available = False
        server.active = False

    def recover(self, server: object) -> None:
        """Marks a server as recovered (available and active)."""
        server.available = True
        server.active = True

    def step(self, servers: list, current_step: int) -> list:
        """Advances the failure process one tick over a set of servers.

        Args:
            servers (list): Edge servers subject to failures.
            current_step (int): Current simulation step (recorded in the returned events).

        Returns:
            events (list): List of ``{"server", "event", "step"}`` dicts describing the
                failures and recoveries that happened this tick.
        """
        events = []
        for server in servers:
            if server.available:
                if self.rng.random() < self.failure_probability:
                    self.fail(server)
                    events.append({"server": server, "event": "failure", "step": current_step})
            else:
                if self.rng.random() < self.recovery_probability:
                    self.recover(server)
                    events.append({"server": server, "event": "recovery", "step": current_step})
        return events

    @staticmethod
    def steady_state_availability(mtbf: float, mttr: float) -> float:
        """Returns the theoretical steady-state availability ``MTBF / (MTBF + MTTR)``."""
        return mtbf / (mtbf + mttr)
