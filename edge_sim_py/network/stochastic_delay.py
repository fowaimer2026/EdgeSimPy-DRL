"""A stochastic delay model adding jitter and packet loss to deterministic delays.

EdgeSimPy treats link and wireless delays as fixed scalars, so end-to-end latency is
deterministic. Real wireless/edge networks exhibit *jitter* (random variation around the
mean) and *packet loss* (which forces retransmissions that inflate effective delay). This
model perturbs any base delay accordingly, and -- crucially for validation -- exposes both
a sampler (for simulation) and the closed-form expected delay (to check the sampler
against).

Effective delay model: a transmission is retried until it succeeds (each attempt fails
independently with probability ``packet_loss``, capped at ``max_retransmissions``); every
attempt costs one ``base_delay``, and zero-mean jitter is added on top.
"""

# Python libraries
import random


class StochasticDelayModel:
    """Perturbs deterministic delays with jitter and packet-loss retransmissions."""

    def __init__(
        self,
        jitter_std: float = 0.0,
        packet_loss: float = 0.0,
        distribution: str = "gaussian",
        max_retransmissions: int = 10,
        seed: int = None,
    ):
        """Creates a stochastic delay model.

        Args:
            jitter_std (float, optional): Jitter magnitude. For ``gaussian`` it is the standard
                deviation (mean 0); for ``exponential`` it is the scale (mean). Defaults to 0.0.
            packet_loss (float, optional): Per-attempt loss probability in ``[0, 1)``. Defaults to 0.0.
            distribution (str, optional): Jitter distribution: ``gaussian`` or ``exponential``. Defaults to "gaussian".
            max_retransmissions (int, optional): Cap on retransmission attempts. Defaults to 10.
            seed (int, optional): RNG seed. Defaults to None.
        """
        if not 0.0 <= packet_loss < 1.0:
            raise ValueError("packet_loss must be in [0, 1).")
        if distribution not in ("gaussian", "exponential"):
            raise ValueError("distribution must be 'gaussian' or 'exponential'.")

        self.jitter_std = jitter_std
        self.packet_loss = packet_loss
        self.distribution = distribution
        self.max_retransmissions = max_retransmissions
        self.rng = random.Random(seed)

    def _sample_jitter(self) -> float:
        """Samples a jitter value from the configured distribution."""
        if self.jitter_std <= 0:
            return 0.0
        if self.distribution == "gaussian":
            return self.rng.gauss(0.0, self.jitter_std)
        return self.rng.expovariate(1.0 / self.jitter_std) - self.jitter_std  # zero-mean exponential

    def _sample_attempts(self) -> int:
        """Samples the number of transmission attempts (>= 1) under packet loss."""
        attempts = 1
        while self.rng.random() < self.packet_loss and attempts < self.max_retransmissions:
            attempts += 1
        return attempts

    def sample(self, base_delay: float) -> float:
        """Returns a stochastic realization of ``base_delay`` (never negative).

        Args:
            base_delay (float): Deterministic base delay to perturb.

        Returns:
            delay (float): Sampled effective delay.
        """
        delay = base_delay * self._sample_attempts() + self._sample_jitter()
        return max(0.0, delay)

    def expected_attempts(self) -> float:
        """Returns the analytical expected number of transmission attempts.

        For a truncated geometric process with loss ``p`` and cap ``m``:
        ``E[attempts] = sum_{k=1}^{m-1} k * p^{k-1} (1-p) + m * p^{m-1}``.
        """
        p = self.packet_loss
        m = self.max_retransmissions
        if p == 0.0:
            return 1.0
        expectation = sum(k * (p ** (k - 1)) * (1 - p) for k in range(1, m))
        expectation += m * (p ** (m - 1))
        return expectation

    def expected_delay(self, base_delay: float) -> float:
        """Returns the analytical expected effective delay (jitter has zero mean)."""
        return base_delay * self.expected_attempts()
