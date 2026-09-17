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
        return (
            self.actor_from_activity(activity),
            self.critic(activity).squeeze(-1),
            activity,
        )

    def actor_from_activity(self, activity: Tensor) -> Tensor:
        return self.actor(activity)

    @staticmethod
    def normalized_activity(activity: Tensor) -> Tensor:
        """Map bounded tanh activity into the browser's display-only [0, 1] range."""
        return torch.clamp((activity + 1.0) / 2.0, min=0.0, max=1.0)


class PopulationFixedGraphPolicy(nn.Module):
    """Fixed MaleCNS topology with interfaces restricted to declared cell populations."""

    def __init__(
        self, graph: ReducedGraphArtifact, observation_size: int, actions: int = 5
    ) -> None:
        super().__init__()
        if observation_size <= 0 or actions <= 0:
            raise ValueError("Population fixed graph policy dimensions must be positive.")
        self.graph = graph
        index_by_body_id = {
            int(body_id): index for index, body_id in enumerate(graph.body_ids)
        }
        try:
            sensory_indices = [
                index_by_body_id[int(body_id)] for body_id in graph.sensory_body_ids
            ]
            readout_indices = [
                index_by_body_id[int(body_id)] for body_id in graph.readout_body_ids
            ]
        except KeyError as error:
            raise ValueError(
                "Population fixed graph policy references cells outside the graph."
            ) from error
        if not sensory_indices or not readout_indices:
            raise ValueError(
                "Population fixed graph policy requires sensory and readout populations."
            )

        self.register_buffer("adjacency", graph.sparse_adjacency(), persistent=False)
        self.register_buffer(
            "sensory_indices",
            torch.tensor(sensory_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "readout_indices",
            torch.tensor(readout_indices, dtype=torch.long),
            persistent=False,
        )
        self.sensory = nn.Linear(
            observation_size, len(sensory_indices), bias=False
        )
        self.actor = nn.Linear(len(readout_indices), actions)
        self.critic = nn.Linear(len(readout_indices), 1)
        self.recurrent_gain = nn.Parameter(torch.tensor(1.0))
        self.time_constant = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, observation: Tensor, hidden: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if observation.ndim != 2 or observation.shape[1] != self.sensory.in_features:
            raise ValueError(
                "Population fixed graph observation has an incompatible shape."
            )
        if hidden.shape != (observation.shape[0], self.graph.node_count):
            raise ValueError(
                "Population fixed graph hidden state has an incompatible shape."
            )

        sensory_drive = self.sensory(observation)
        injected = torch.zeros_like(hidden)
        injected = torch.index_copy(
            injected, 1, self.sensory_indices, sensory_drive
        )
        recurrent = torch.sparse.mm(self.adjacency, hidden.T).T
        candidate = torch.tanh(
            injected + torch.clamp(self.recurrent_gain, min=0.0) * recurrent
        )
        time_constant = torch.clamp(self.time_constant, min=1e-4, max=1.0)
        activity = hidden + time_constant * (candidate - hidden)
        readout_activity = activity.index_select(1, self.readout_indices)
        return (
            self.actor(readout_activity),
            self.critic(readout_activity).squeeze(-1),
            activity,
        )

    def actor_from_activity(self, activity: Tensor) -> Tensor:
        return self.actor(activity.index_select(1, self.readout_indices))

    @staticmethod
    def normalized_activity(activity: Tensor) -> Tensor:
        return FixedGraphPolicy.normalized_activity(activity)
