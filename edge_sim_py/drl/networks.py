"""Neural network architectures shared by the DRL offloading agents."""

# Python libraries
import torch
import torch.nn as nn


def _mlp(input_dim: int, hidden_sizes: tuple, output_dim: int) -> nn.Sequential:
    """Builds a simple fully-connected ReLU network."""
    layers = []
    last = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last, size))
        layers.append(nn.ReLU())
        last = size
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Q-network for DQN, with an optional dueling architecture.

    The dueling variant splits the final representation into a state-value stream and an
    advantage stream, recombining them as ``Q = V + (A - mean(A))`` which often learns
    state values more efficiently.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: tuple = (128, 128), dueling: bool = True):
        super().__init__()
        self.dueling = dueling
        if dueling:
            self.feature = _mlp(state_dim, hidden_sizes[:-1], hidden_sizes[-1])
            self.value_head = nn.Linear(hidden_sizes[-1], 1)
            self.advantage_head = nn.Linear(hidden_sizes[-1], action_dim)
        else:
            self.net = _mlp(state_dim, hidden_sizes, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if not self.dueling:
            return self.net(state)
        features = torch.relu(self.feature(state))
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic network for the PPO and A2C agents."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: tuple = (128, 128)):
        super().__init__()
        self.trunk = _mlp(state_dim, hidden_sizes[:-1], hidden_sizes[-1])
        self.policy_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, state: torch.Tensor) -> tuple:
        """Returns ``(policy_logits, value)``."""
        features = torch.relu(self.trunk(state))
        return self.policy_head(features), self.value_head(features).squeeze(-1)
