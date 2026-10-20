"""Evaluation utilities and baselines for the DRL offloading agents."""

# Python libraries
import random

import numpy as np


def evaluate_agent(agent, env, num_episodes: int = 50) -> dict:
    """Runs an agent greedily and reports mean reward, placement rate and total cost.

    Args:
        agent: A trained agent exposing ``act(state, mask, greedy=True)``.
        env: An :class:`OffloadingEnv`.
        num_episodes (int): Number of evaluation episodes.

    Returns:
        metrics (dict): Mean episode reward, mean placement rate and mean total cost
            (``-reward`` summed over the episode).
    """
    rewards, placement_rates = [], []
    for _ in range(num_episodes):
        state = env.reset()
        mask = env.action_mask.copy()
        episode_reward = 0.0
        done = False
        while not done:
            action = agent.act(state, mask, greedy=True)
            state, reward, done, info = env.step(action)
            mask = info["action_mask"]
            episode_reward += reward
        rewards.append(episode_reward)
        placement_rates.append(info["placed"] / len(env.services))
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_placement_rate": float(np.mean(placement_rates)),
        "mean_total_cost": float(-np.mean(rewards)),
    }


def evaluate_greedy(env, num_episodes: int = 50) -> dict:
    """Evaluates the min-cost greedy baseline (always pick the lowest-cost valid server)."""
    rewards, placement_rates = [], []
    for _ in range(num_episodes):
        env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            _, reward, done, info = env.step(env.greedy_action())
            episode_reward += reward
        rewards.append(episode_reward)
        placement_rates.append(info["placed"] / len(env.services))
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_placement_rate": float(np.mean(placement_rates)),
        "mean_total_cost": float(-np.mean(rewards)),
    }


def evaluate_random(env, num_episodes: int = 50, seed: int = 0) -> dict:
    """Evaluates the random baseline (pick a random valid server)."""
    rng = random.Random(seed)
    rewards, placement_rates = [], []
    for _ in range(num_episodes):
        env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            _, reward, done, info = env.step(env.random_action(rng=rng))
            episode_reward += reward
        rewards.append(episode_reward)
        placement_rates.append(info["placed"] / len(env.services))
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_placement_rate": float(np.mean(placement_rates)),
        "mean_total_cost": float(-np.mean(rewards)),
    }
