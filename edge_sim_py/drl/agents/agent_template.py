"""Template for adding a new DRL agent (Part 3, item 5 - extensibility).

A new agent needs to satisfy only the small duck-typed contract that ``OffloadingEnv``, the
evaluation helpers (``evaluate_agent``) and the live-sim adapter (``drl_offloading_policy``)
already rely on -- **no change to the simulator core or the environment is required**. Copy this
file, rename the class, and fill in the three methods with your algorithm (e.g. SAC, TD3,
Rainbow DQN); then export it from ``edge_sim_py/drl/agents/__init__.py``.

Contract (identical to DQNAgent / PPOAgent / A2CAgent):

    __init__(state_dim, action_dim, **hyperparams)
    act(state, mask, greedy=False) -> int      # MUST respect the boolean action mask
    train(env, num_episodes)      -> list[float]  # returns per-episode total reward

Integration notes for specific algorithm families:
    * Discrete value-based (Rainbow DQN): drops in directly; reuse the replay buffer.
    * Continuous actor-critic (SAC, TD3): the action space here is *discrete* (choose a server),
      so either (a) use a Gumbel-softmax / categorical actor head, or (b) have the actor output a
      score per server and take ``argmax`` over the masked scores in ``act``.
    * Always apply the mask (servers without capacity): set masked logits/Q-values to -inf before
      the arg-max/sample so the agent never selects an infeasible action.

This template is a runnable (random-but-masked) agent so the interface is testable as-is.
"""

# Python libraries
import random

import numpy as np


class TemplateAgent:
    """Minimal runnable agent skeleton — replace the marked sections with your algorithm."""

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3, seed: int = None, **hyperparams):
        """Store dimensions and build your networks/optimizer here.

        Args:
            state_dim (int): Length of the observation vector from ``OffloadingEnv``.
            action_dim (int): Number of servers (discrete actions).
            learning_rate (float): Example hyperparameter (add your own via **hyperparams).
            seed (int, optional): RNG/torch seed for reproducibility.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rng = random.Random(seed)
        # --- BUILD YOUR MODEL HERE ---
        # e.g. self.q_net = QNetwork(state_dim, action_dim); self.optimizer = Adam(...)
        # For SAC/TD3 build actor + twin critics; for Rainbow build the dueling/noisy net + buffer.

    def act(self, state: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        """Return a *valid* action index (a server with capacity).

        MUST honor ``mask``: only indices where ``mask[i]`` is True are feasible. The example
        below picks uniformly among valid actions; replace it with your policy's decision
        (arg-max of masked Q-values, or a sample from the masked categorical policy).
        """
        valid = np.flatnonzero(mask)
        # --- REPLACE WITH YOUR POLICY ---
        # scores = self.q_net(state); scores[~mask] = -inf; return int(scores.argmax())
        return int(self.rng.choice(valid))

    def train(self, env, num_episodes: int) -> list:
        """Run ``num_episodes`` of training and return the per-episode total reward.

        The standard loop is shown; insert your ``remember``/``update`` calls where marked. The
        environment API is ``reset() -> state`` and ``step(action) -> (next_state, reward, done,
        info)`` with ``info["action_mask"]`` giving the next mask.
        """
        history = []
        for _ in range(num_episodes):
            state = env.reset()
            mask = env.action_mask.copy()
            episode_reward, done = 0.0, False
            while not done:
                action = self.act(state, mask)
                next_state, reward, done, info = env.step(action)
                next_mask = info["action_mask"]
                # --- STORE THE TRANSITION AND UPDATE YOUR MODEL HERE ---
                # self.remember(state, action, reward, next_state, done, next_mask); self.update()
                state, mask = next_state, next_mask
                episode_reward += reward
            history.append(episode_reward)
        return history
