"""Verifies the DRL agent template satisfies the interface the env/evaluation rely on.

This is the extensibility guarantee (Part 3, item 5): any new agent implementing act/train can
be trained and evaluated without changing the simulator core or the environment.
"""

import numpy as np
import pytest

pytest.importorskip("torch")  # importing the agents package pulls in the torch-based agents

from edge_sim_py.drl import OffloadingEnv, random_scenario, evaluate_agent
from edge_sim_py.drl.agents.agent_template import TemplateAgent


def _env(seed=0):
    services, servers = random_scenario(num_services=5, num_servers=4, seed=seed)
    return OffloadingEnv(services, servers)


def test_template_act_returns_valid_action():
    env = _env()
    state = env.reset()
    agent = TemplateAgent(env.state_dim, env.num_actions, seed=0)
    action = agent.act(state, env.action_mask, greedy=True)
    assert 0 <= action < env.num_actions
    assert env.action_mask[action]  # must respect the mask


def test_template_trains_and_evaluates_via_shared_helpers():
    env = _env()
    agent = TemplateAgent(env.state_dim, env.num_actions, seed=0)
    history = agent.train(env, num_episodes=3)
    assert len(history) == 3 and all(isinstance(r, float) for r in history)
    # The standard evaluation helper works unchanged on the new agent.
    metrics = evaluate_agent(agent, env, num_episodes=5)
    assert "mean_total_cost" in metrics and metrics["mean_placement_rate"] == 1.0
