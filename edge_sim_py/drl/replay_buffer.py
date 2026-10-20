"""Experience replay buffer for the DQN agent."""

# Python libraries
import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """Fixed-capacity circular buffer of transitions sampled uniformly at random.

    Each transition stores the action mask of the *next* state so the agent can compute
    masked max-Q targets (servers without capacity must be excluded from the bootstrap).
    """

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        """Adds a transition. ``next_state`` may be ``None`` on terminal steps."""
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int) -> tuple:
        """Samples a batch, returning stacked NumPy arrays.

        Terminal transitions (``next_state is None``) are encoded as zero next-states with
        ``done = 1`` so they contribute no bootstrap value.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        state_dim = len(states[0])
        num_actions = len(next_masks[0])
        next_states = [ns if ns is not None else np.zeros(state_dim, dtype=np.float32) for ns in next_states]

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(next_masks, dtype=bool).reshape(-1, num_actions),
        )

    def __len__(self) -> int:
        return len(self.buffer)
