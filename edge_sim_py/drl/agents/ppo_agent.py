"""Proximal Policy Optimization (PPO) agent for offloading placement."""

# EdgeSimPy DRL
from edge_sim_py.drl.networks import ActorCritic

# Python libraries
import random

import numpy as np
import torch
from torch.distributions import Categorical


class PPOAgent:
    """On-policy actor-critic agent trained with the clipped PPO objective and GAE.

    Invalid actions are masked out of the policy distribution, so the agent only ever
    samples (and is only ever updated toward) servers that can host the current service.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple = (128, 128),
        learning_rate: float = 3e-4,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        update_epochs: int = 4,
        minibatch_size: int = 64,
        update_every_episodes: int = 4,
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
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.update_every_episodes = update_every_episodes
        self.device = torch.device(device)

        self.net = ActorCritic(state_dim, action_dim, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def _distribution(self, logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
        """Builds a categorical policy with invalid actions masked to probability zero."""
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~masks, neg_inf)
        return Categorical(logits=masked_logits)

    def act(self, state: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        """Samples a valid action (greedy = take the arg-max of the masked policy)."""
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
            dist = self._distribution(logits, torch.as_tensor(mask, device=self.device).unsqueeze(0))
            if greedy:
                return int(dist.probs.argmax(dim=-1).item())
            return int(dist.sample().item())

    def _compute_gae(self, rewards, values, dones) -> tuple:
        """Computes Generalized Advantage Estimation returns and advantages."""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0.0
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if (t + 1 >= len(rewards) or dones[t]) else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_advantage
            advantages[t] = last_advantage
        returns = advantages + values
        return returns, advantages

    def _update(self, rollout: dict):
        """Runs the PPO update over the collected rollout."""
        states = torch.as_tensor(np.asarray(rollout["states"], dtype=np.float32), device=self.device)
        actions = torch.as_tensor(np.asarray(rollout["actions"], dtype=np.int64), device=self.device)
        masks = torch.as_tensor(np.asarray(rollout["masks"], dtype=bool), device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(rollout["log_probs"], dtype=np.float32), device=self.device)

        returns, advantages = self._compute_gae(
            np.asarray(rollout["rewards"], dtype=np.float32),
            np.asarray(rollout["values"], dtype=np.float32),
            np.asarray(rollout["dones"], dtype=np.float32),
        )
        returns = torch.as_tensor(returns, device=self.device)
        advantages = torch.as_tensor(advantages, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = len(actions)
        indices = np.arange(num_samples)
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.minibatch_size):
                batch = indices[start : start + self.minibatch_size]
                logits, values = self.net(states[batch])
                dist = self._distribution(logits, masks[batch])

                new_log_probs = dist.log_prob(actions[batch])
                ratio = torch.exp(new_log_probs - old_log_probs[batch])
                surrogate1 = ratio * advantages[batch]
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages[batch]
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = torch.nn.functional.mse_loss(values, returns[batch])
                entropy = dist.entropy().mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

    def train(self, env, num_episodes: int = 200) -> list:
        """Trains the agent, updating every ``update_every_episodes`` episodes."""
        history = []
        rollout = {key: [] for key in ("states", "actions", "masks", "log_probs", "rewards", "values", "dones")}

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
                    log_prob = dist.log_prob(action)

                next_state, reward, done, info = env.step(int(action.item()))

                rollout["states"].append(state)
                rollout["actions"].append(int(action.item()))
                rollout["masks"].append(mask)
                rollout["log_probs"].append(float(log_prob.item()))
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
