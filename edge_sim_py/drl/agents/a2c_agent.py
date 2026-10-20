"""Advantage Actor-Critic (A2C) agent for offloading placement.

A2C is the synchronous form of A3C: the same advantage actor-critic update, but
collected from sequential rollouts rather than asynchronous worker threads. The policy
and value share a trunk (see :class:`ActorCritic`); each rollout yields one on-policy
gradient step using GAE advantages.
"""

# EdgeSimPy DRL
from edge_sim_py.drl.networks import ActorCritic

# Python libraries
import random

import numpy as np
import torch
from torch.distributions import Categorical


class A2CAgent:
    """Synchronous advantage actor-critic agent with action masking."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple = (128, 128),
        learning_rate: float = 7e-4,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        update_every_episodes: int = 1,
        seed: int = None,
        device: str = "cpu",
    ):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_every_episodes = update_every_episodes
        self.device = torch.device(device)

        self.net = ActorCritic(state_dim, action_dim, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def _distribution(self, logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
        """Builds a categorical policy with invalid actions masked to probability zero."""
        neg_inf = torch.finfo(logits.dtype).min
        return Categorical(logits=logits.masked_fill(~masks, neg_inf))

    def act(self, state: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        """Samples a valid action (greedy = arg-max of the masked policy)."""
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
            dist = self._distribution(logits, torch.as_tensor(mask, device=self.device).unsqueeze(0))
            if greedy:
                return int(dist.probs.argmax(dim=-1).item())
            return int(dist.sample().item())

    def _compute_gae(self, rewards, values, dones) -> tuple:
        """Computes GAE returns and advantages from a rollout."""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0.0
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if (t + 1 >= len(rewards) or dones[t]) else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_advantage
            advantages[t] = last_advantage
        return advantages + values, advantages

    def _update(self, rollout: dict):
        """Runs one A2C gradient step over the collected rollout."""
        states = torch.as_tensor(np.asarray(rollout["states"], dtype=np.float32), device=self.device)
        actions = torch.as_tensor(np.asarray(rollout["actions"], dtype=np.int64), device=self.device)
        masks = torch.as_tensor(np.asarray(rollout["masks"], dtype=bool), device=self.device)

        returns, advantages = self._compute_gae(
            np.asarray(rollout["rewards"], dtype=np.float32),
            np.asarray(rollout["values"], dtype=np.float32),
            np.asarray(rollout["dones"], dtype=np.float32),
        )
        returns = torch.as_tensor(returns, device=self.device)
        advantages = torch.as_tensor(advantages, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logits, values = self.net(states)
        dist = self._distribution(logits, masks)
        log_probs = dist.log_prob(actions)

        policy_loss = -(log_probs * advantages).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns)
        entropy = dist.entropy().mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

    def train(self, env, num_episodes: int = 200) -> list:
        """Trains the agent, updating every ``update_every_episodes`` episodes."""
        history = []
        rollout = {key: [] for key in ("states", "actions", "masks", "rewards", "values", "dones")}

        for episode in range(num_episodes):
            state = env.reset()
            mask = env.action_mask.copy()
            episode_reward = 0.0
            done = False
            while not done:
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits, value = self.net(state_tensor)
                    dist = self._distribution(logits, torch.as_tensor(mask, device=self.device).unsqueeze(0))
                    action = dist.sample()

                next_state, reward, done, info = env.step(int(action.item()))

                rollout["states"].append(state)
                rollout["actions"].append(int(action.item()))
                rollout["masks"].append(mask)
                rollout["rewards"].append(reward)
                rollout["values"].append(float(value.item()))
                rollout["dones"].append(float(done))

                state, mask = next_state, info["action_mask"]
                episode_reward += reward

            history.append(episode_reward)
            if (episode + 1) % self.update_every_episodes == 0:
                self._update(rollout)
                rollout = {key: [] for key in rollout}

        if rollout["states"]:
            self._update(rollout)
        return history
