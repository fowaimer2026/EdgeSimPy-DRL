"""Unit tests for the DRL offloading subsystem (environment + agents).

These tests use tiny scenarios and very short training runs so they stay fast while
still exercising the env dynamics, action masking, reward sign and each agent's
act/train/evaluate path. Requires PyTorch. Run with: ``pytest tests/test_drl.py``.
"""

# Python libraries
import numpy as np
import pytest

# The whole subsystem depends on torch; skip cleanly if it is unavailable.
pytest.importorskip("torch")

# EdgeSimPy DRL
from edge_sim_py.drl import (
    OffloadingEnv,
    DQNAgent,
    PPOAgent,
    A2CAgent,
    random_scenario,
    evaluate_agent,
    evaluate_greedy,
    evaluate_random,
)
from edge_sim_py.drl.networks import QNetwork, ActorCritic
from edge_sim_py.drl.policy import drl_offloading_policy


@pytest.fixture
def env():
    services, servers = random_scenario(num_services=4, num_servers=3, seed=7)
    return OffloadingEnv(services, servers, cost_parameters={"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}})


def test_env_dimensions_and_reset(env):
    state = env.reset()
    assert env.num_actions == 3
    assert env.state_dim == 3 + 3 * 6
    assert state.shape == (env.state_dim,)
    assert env.action_mask.dtype == bool


def test_env_step_reward_is_negative_normalized_cost(env):
    env.reset()
    _, reward, _, info = env.step(env.greedy_action())
    # Valid placement reward is the negative normalized score, in [-1, 0].
    assert info["valid"] is True
    assert -1.0 - 1e-6 <= reward <= 0.0 + 1e-6


def test_env_episode_terminates_after_all_services(env):
    env.reset()
    steps = 0
    done = False
    while not done:
        _, _, done, _ = env.step(env.greedy_action())
        steps += 1
    assert steps == len(env.services)


def test_invalid_action_is_penalized():
    # One tiny server that cannot host a large service -> the action is invalid.
    services, servers = random_scenario(num_services=1, num_servers=1, seed=1)
    servers[0].cpu = 1
    services[0].cpu_demand = 1000  # impossible to host
    env = OffloadingEnv(services, servers)
    env.reset()
    _, reward, done, info = env.step(0)
    assert info["valid"] is False
    assert reward == pytest.approx(-env.invalid_penalty)
    assert done is True


def test_qnetwork_forward_shapes():
    net = QNetwork(state_dim=10, action_dim=4)
    import torch

    out = net(torch.zeros(5, 10))
    assert out.shape == (5, 4)


def test_actor_critic_forward_shapes():
    net = ActorCritic(state_dim=10, action_dim=4)
    import torch

    logits, value = net(torch.zeros(5, 10))
    assert logits.shape == (5, 4)
    assert value.shape == (5,)


@pytest.mark.parametrize("agent_class", [DQNAgent, PPOAgent, A2CAgent])
def test_agent_acts_within_mask(env, agent_class):
    state = env.reset()
    agent = agent_class(env.state_dim, env.num_actions, seed=0)
    action = agent.act(state, env.action_mask, greedy=True)
    assert 0 <= action < env.num_actions
    assert env.action_mask[action]  # the chosen action must be valid


@pytest.mark.parametrize("agent_class", [DQNAgent, PPOAgent, A2CAgent])
def test_agent_trains_and_returns_history(env, agent_class):
    agent = agent_class(env.state_dim, env.num_actions, seed=0)
    history = agent.train(env, num_episodes=3)
    assert len(history) == 3
    assert all(isinstance(reward, float) for reward in history)


def test_trained_agent_beats_random():
    services, servers = random_scenario(num_services=5, num_servers=4, seed=3)
    env = OffloadingEnv(services, servers)
    agent = DQNAgent(env.state_dim, env.num_actions, seed=0)
    agent.train(env, num_episodes=80)

    trained = evaluate_agent(agent, env, num_episodes=20)
    random_baseline = evaluate_random(env, num_episodes=20)
    greedy = evaluate_greedy(env, num_episodes=20)

    # A trained agent should beat random and approach the greedy optimum.
    assert trained["mean_total_cost"] < random_baseline["mean_total_cost"]
    assert trained["mean_total_cost"] <= greedy["mean_total_cost"] + 0.5


def test_policy_adapter_requires_agent():
    with pytest.raises(ValueError):
        drl_offloading_policy(parameters={})
