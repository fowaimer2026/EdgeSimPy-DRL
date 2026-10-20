"""Deep Q-Network agent (with Double and Dueling variants) for offloading placement."""

# EdgeSimPy DRL
from edge_sim_py.drl.networks import QNetwork
from edge_sim_py.drl.replay_buffer import ReplayBuffer

# Python libraries
import random

import numpy as np
import torch
import torch.nn as nn


class DQNAgent:
    """Value-based agent that learns Q(state, server) and acts epsilon-greedily.

    Supports Double DQN (decoupling action selection from evaluation to reduce
    overestimation) and Dueling networks. Invalid actions (servers without capacity) are
    masked out both when acting and when computing bootstrap targets.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple = (128, 128),
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update_interval: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        double_dqn: bool = True,
        dueling: bool = True,
        seed: int = None,
        device: str = "cpu",
    ):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.double_dqn = double_dqn
        self.device = torch.device(device)

        self.online = QNetwork(state_dim, action_dim, hidden_sizes, dueling).to(self.device)
        self.target = QNetwork(state_dim, action_dim, hidden_sizes, dueling).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_capacity)
        self._learn_steps = 0

    def act(self, state: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        """Selects a valid action (epsilon-greedy unless ``greedy``)."""
        valid_actions = np.flatnonzero(mask)
        if not greedy and random.random() < self.epsilon:
            return int(random.choice(valid_actions))

        with torch.no_grad():
            q_values = self.online(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
            q_values = q_values.squeeze(0).cpu().numpy()

        masked_q = np.full(self.action_dim, -np.inf, dtype=np.float32)
        masked_q[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked_q))

    def remember(self, state, action, reward, next_state, done, next_mask):
        """Stores a transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done, next_mask)

    def update(self):
        """Performs one gradient step on a sampled batch (no-op until the buffer fills)."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, next_masks = self.buffer.sample(self.batch_size)
        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)
        next_masks = torch.as_tensor(next_masks, device=self.device)

        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.online(next_states)
            next_q_target = self.target(next_states)
            # Mask invalid next-state actions with -inf before taking the max.
            neg_inf = torch.finfo(next_q_online.dtype).min
            next_q_online = next_q_online.masked_fill(~next_masks, neg_inf)

            if self.double_dqn:
                best_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_value = next_q_target.gather(1, best_actions).squeeze(1)
            else:
                next_q_target = next_q_target.masked_fill(~next_masks, neg_inf)
                next_value = next_q_target.max(dim=1).values

            target = rewards + self.gamma * next_value * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def train(self, env, num_episodes: int = 200) -> list:
        """Trains the agent on an environment, returning the per-episode reward history."""
        history = []
        for _ in range(num_episodes):
            state = env.reset()
            mask = env.action_mask.copy()
            episode_reward = 0.0
            done = False
            while not done:
                action = self.act(state, mask)
                next_state, reward, done, info = env.step(action)
                next_mask = info["action_mask"]
                self.remember(state, action, reward, next_state, done, next_mask)
                self.update()
                state, mask = next_state, next_mask
                episode_reward += reward

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            history.append(episode_reward)
        return history
