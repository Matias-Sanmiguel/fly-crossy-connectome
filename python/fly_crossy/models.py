from __future__ import annotations

import torch
from torch import Tensor, nn

from .connectome import ReducedGraphArtifact


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


class FixedGraphPolicy(nn.Module):
    """Actor-critic with fixed MaleCNS topology and trainable artificial interfaces."""

    def __init__(
        self, graph: ReducedGraphArtifact, observation_size: int, actions: int = 5
    ) -> None:
        super().__init__()
        if observation_size <= 0 or actions <= 0:
            raise ValueError("Fixed graph policy dimensions must be positive.")
        self.graph = graph
        self.register_buffer("adjacency", graph.sparse_adjacency(), persistent=False)
        self.sensory = nn.Linear(observation_size, graph.node_count, bias=False)
        self.actor = nn.Linear(graph.node_count, actions)
        self.critic = nn.Linear(graph.node_count, 1)
        self.recurrent_gain = nn.Parameter(torch.tensor(1.0))
        self.time_constant = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, observation: Tensor, hidden: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if observation.ndim != 2 or observation.shape[1] != self.sensory.in_features:
            raise ValueError("Fixed graph observation has an incompatible shape.")
        if hidden.shape != (observation.shape[0], self.graph.node_count):
            raise ValueError("Fixed graph hidden state has an incompatible shape.")
        recurrent = torch.sparse.mm(self.adjacency, hidden.T).T
        candidate = torch.tanh(
            self.sensory(observation)
            + torch.clamp(self.recurrent_gain, min=0.0) * recurrent
        )
        time_constant = torch.clamp(self.time_constant, min=1e-4, max=1.0)
        activity = hidden + time_constant * (candidate - hidden)
        return self.actor(activity), self.critic(activity).squeeze(-1), activity

    @staticmethod
    def normalized_activity(activity: Tensor) -> Tensor:
        """Map bounded tanh activity into the browser's display-only [0, 1] range."""
        return torch.clamp((activity + 1.0) / 2.0, min=0.0, max=1.0)
