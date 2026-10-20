"""Unit tests for the stochastic network delay model."""

# Python libraries
import numpy as np
import pytest

# EdgeSimPy
from edge_sim_py.network import StochasticDelayModel


def test_deterministic_when_no_jitter_or_loss():
    model = StochasticDelayModel(jitter_std=0.0, packet_loss=0.0, seed=0)
    assert model.sample(10.0) == 10.0
    assert model.expected_delay(10.0) == 10.0


def test_expected_attempts_without_loss_is_one():
    model = StochasticDelayModel(packet_loss=0.0)
    assert model.expected_attempts() == 1.0


def test_packet_loss_increases_expected_delay():
    lossless = StochasticDelayModel(packet_loss=0.0)
    lossy = StochasticDelayModel(packet_loss=0.3)
    assert lossy.expected_delay(10.0) > lossless.expected_delay(10.0)


def test_sample_never_negative():
    model = StochasticDelayModel(jitter_std=50.0, packet_loss=0.0, seed=1)
    assert all(model.sample(1.0) >= 0.0 for _ in range(1000))


def test_empirical_mean_matches_analytical_expectation():
    # Validation: the sampler's mean should track the closed-form expected delay.
    model = StochasticDelayModel(jitter_std=2.0, packet_loss=0.2, seed=7)
    samples = [model.sample(10.0) for _ in range(50_000)]
    assert np.mean(samples) == pytest.approx(model.expected_delay(10.0), rel=0.03)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        StochasticDelayModel(packet_loss=1.0)
    with pytest.raises(ValueError):
        StochasticDelayModel(distribution="poisson")
