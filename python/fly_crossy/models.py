from __future__ import annotations

import torch
from torch import Tensor, nn


class DensePolicy(nn.Module):
    """Compact shared actor-critic used by the conventional PPO baseline."""

    def __init__(self, observation_size: int, hidden_size: int, actions: int) -> None:
        super().__init__()
        if observation_size <= 0 or hidden_size <= 0 or actions <= 0:
            raise ValueError("Dense policy dimensions must be positive.")
        self.hidden_1 = nn.Linear(observation_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        hidden = torch.tanh(self.hidden_1(observations))
        hidden = torch.tanh(self.hidden_2(hidden))
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

