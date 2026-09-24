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
            self.actor_from_activity(activity),
            self.critic(readout_activity).squeeze(-1),
            activity,
        )

    def actor_from_activity(self, activity: Tensor) -> Tensor:
        return self.actor(activity.index_select(1, self.readout_indices))

    @staticmethod
    def normalized_activity(activity: Tensor) -> Tensor:
        return FixedGraphPolicy.normalized_activity(activity)

class TrafficAwarePopulationPolicy(PopulationFixedGraphPolicy):
    """Controller V2: recurrent connectome plus learned per-action safety heads."""

    SAFETY_GAIN = 4.0
    ROUTE_GAIN = 0.75

    def __init__(
        self,
        graph: ReducedGraphArtifact,
        observation_size: int,
        actions: int = 5,
    ) -> None:
        super().__init__(graph, observation_size, actions)
        readout_count = len(self.readout_indices)
        self.risk_head = nn.Linear(readout_count, actions)
        self.route_head = nn.Linear(readout_count, actions)

    def auxiliary_from_activity(self, activity: Tensor) -> tuple[Tensor, Tensor]:
        readout_activity = activity.index_select(1, self.readout_indices)
        return (
            self.risk_head(readout_activity),
            self.route_head(readout_activity),
        )

    def actor_from_activity(self, activity: Tensor) -> Tensor:
        readout_activity = activity.index_select(1, self.readout_indices)
        base_logits = self.actor(readout_activity)
        risk_logits = self.risk_head(readout_activity)
        route_logits = self.route_head(readout_activity)
        return (
            base_logits
            - self.SAFETY_GAIN * torch.sigmoid(risk_logits)
            + self.ROUTE_GAIN * torch.sigmoid(route_logits)
        )

class NestedPopulationFixedGraphPolicy(nn.Module):
    """Population interface with an exactly preserved 80-cell recurrent core."""

    def __init__(
        self,
        graph: ReducedGraphArtifact,
        core_graph: ReducedGraphArtifact,
        observation_size: int,
        actions: int = 5,
    ) -> None:
        super().__init__()
        if observation_size <= 0 or actions <= 0:
            raise ValueError("Nested fixed graph policy dimensions must be positive.")

        graph_ids = [int(body_id) for body_id in graph.body_ids]
        core_ids = [int(body_id) for body_id in core_graph.body_ids]
        graph_id_set = set(graph_ids)
        core_id_set = set(core_ids)

        if not core_id_set.issubset(graph_id_set):
            raise ValueError("Nested core body IDs must be contained in the full graph.")
        if set(int(x) for x in graph.sensory_body_ids) != set(
            int(x) for x in core_graph.sensory_body_ids
        ):
            raise ValueError("Nested graph sensory population must match the core.")
        if set(int(x) for x in graph.readout_body_ids) != set(
            int(x) for x in core_graph.readout_body_ids
        ):
            raise ValueError("Nested graph readout population must match the core.")

        self.graph = graph
        self.core_graph = core_graph
        index_by_body_id = {
            int(body_id): index for index, body_id in enumerate(graph.body_ids)
        }

        sensory_indices = [
            index_by_body_id[int(body_id)] for body_id in graph.sensory_body_ids
        ]
        readout_indices = [
            index_by_body_id[int(body_id)] for body_id in graph.readout_body_ids
        ]
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

        core_sources: list[int] = []
        core_targets: list[int] = []
        core_weights: list[float] = []
        for edge, weight in zip(
            core_graph.edge_index.T.tolist(),
            core_graph.edge_weight.tolist(),
            strict=True,
        ):
            source_body_id = int(core_graph.body_ids[int(edge[0])])
            target_body_id = int(core_graph.body_ids[int(edge[1])])
            core_sources.append(index_by_body_id[source_body_id])
            core_targets.append(index_by_body_id[target_body_id])
            core_weights.append(float(weight))

        self.register_buffer(
            "core_adjacency",
            self._make_sparse_adjacency(
                graph.node_count,
                core_sources,
                core_targets,
                core_weights,
            ),
            persistent=False,
        )

        # The full 1k artifact was normalized over every incoming edge. For the
        # nested model we remove core->core edges from the expansion and
        # re-normalize all remaining edges per target. Because all incoming
        # weights to a given target shared the same old denominator, this
        # recovers the relative signed contact weights within the expansion.
        expansion_edges: list[tuple[int, int, float]] = []
        expansion_denominators: dict[int, float] = {}
        for edge, weight in zip(
            graph.edge_index.T.tolist(),
            graph.edge_weight.tolist(),
            strict=True,
        ):
            source = int(edge[0])
            target = int(edge[1])
            source_body_id = int(graph.body_ids[source])
            target_body_id = int(graph.body_ids[target])
            if source_body_id in core_id_set and target_body_id in core_id_set:
                continue
            numeric_weight = float(weight)
            expansion_edges.append((source, target, numeric_weight))
            expansion_denominators[target] = (
                expansion_denominators.get(target, 0.0) + abs(numeric_weight)
            )

        expansion_sources: list[int] = []
        expansion_targets: list[int] = []
        expansion_weights: list[float] = []
        for source, target, weight in expansion_edges:
            denominator = expansion_denominators.get(target, 0.0)
            expansion_sources.append(source)
            expansion_targets.append(target)
            expansion_weights.append(
                0.0 if denominator == 0.0 else weight / denominator
            )

        self.register_buffer(
            "expansion_adjacency",
            self._make_sparse_adjacency(
                graph.node_count,
                expansion_sources,
                expansion_targets,
                expansion_weights,
            ),
            persistent=False,
        )

        self.sensory = nn.Linear(
            observation_size, len(sensory_indices), bias=False
        )
        self.actor = nn.Linear(len(readout_indices), actions)
        self.critic = nn.Linear(len(readout_indices), 1)
        self.recurrent_gain = nn.Parameter(torch.tensor(1.0))
        self.expansion_gain = nn.Parameter(torch.tensor(0.1))
        self.time_constant = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _make_sparse_adjacency(
        node_count: int,
        sources: list[int],
        targets: list[int],
        weights: list[float],
    ) -> Tensor:
        if len(sources) != len(targets) or len(sources) != len(weights):
            raise ValueError("Nested adjacency arrays must have matching lengths.")
        if weights:
            indices = torch.tensor([targets, sources], dtype=torch.long)
            values = torch.tensor(weights, dtype=torch.float32)
        else:
            indices = torch.empty((2, 0), dtype=torch.long)
            values = torch.empty((0,), dtype=torch.float32)
        with torch.sparse.check_sparse_tensor_invariants():
            return torch.sparse_coo_tensor(
                indices,
                values,
                (node_count, node_count),
                dtype=torch.float32,
                check_invariants=True,
            ).coalesce()

    @property
    def core_edge_count(self) -> int:
        return int(self.core_adjacency._nnz())

    @property
    def expansion_edge_count(self) -> int:
        return int(self.expansion_adjacency._nnz())

    def forward(
        self, observation: Tensor, hidden: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if observation.ndim != 2 or observation.shape[1] != self.sensory.in_features:
            raise ValueError("Nested fixed graph observation has an incompatible shape.")
        if hidden.shape != (observation.shape[0], self.graph.node_count):
            raise ValueError("Nested fixed graph hidden state has an incompatible shape.")

        sensory_drive = self.sensory(observation)
        injected = torch.zeros_like(hidden)
        injected = torch.index_copy(
            injected, 1, self.sensory_indices, sensory_drive
        )

        core_recurrent = torch.sparse.mm(self.core_adjacency, hidden.T).T
        expansion_recurrent = torch.sparse.mm(
            self.expansion_adjacency, hidden.T
        ).T

        core_gain = torch.clamp(self.recurrent_gain, min=0.0)
        expansion_gain = torch.clamp(self.expansion_gain, min=0.0, max=1.0)
        candidate = torch.tanh(
            injected
            + core_gain * core_recurrent
            + expansion_gain * expansion_recurrent
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

    def effective_recurrent_edges(self) -> tuple[Tensor, Tensor]:
        """Return source-target indices and already-gained recurrent weights."""
        core = self.core_adjacency.coalesce()
        expansion = self.expansion_adjacency.coalesce()

        index_parts: list[Tensor] = []
        value_parts: list[Tensor] = []

        if core._nnz():
            index_parts.append(core.indices()[[1, 0]])
            value_parts.append(
                core.values() * torch.clamp(self.recurrent_gain, min=0.0)
            )
        if expansion._nnz():
            index_parts.append(expansion.indices()[[1, 0]])
            value_parts.append(
                expansion.values()
                * torch.clamp(self.expansion_gain, min=0.0, max=1.0)
            )

        if not index_parts:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.float32),
            )
        return torch.cat(index_parts, dim=1), torch.cat(value_parts, dim=0)

    @staticmethod
    def normalized_activity(activity: Tensor) -> Tensor:
        return FixedGraphPolicy.normalized_activity(activity)
class GatedNestedPopulationFixedGraphPolicy(NestedPopulationFixedGraphPolicy):
    """Nested core-preserving policy with a differentiable bounded expansion gate."""

    def __init__(
        self,
        graph: ReducedGraphArtifact,
        core_graph: ReducedGraphArtifact,
        observation_size: int,
        actions: int = 5,
    ) -> None:
        super().__init__(graph, core_graph, observation_size, actions)

        # Preserve old "nested" checkpoint semantics by using a separate class
        # and a different state key. The new gate is parameterized in logit
        # space so its effective value always remains strictly in (0, 1).
        del self.expansion_gain
        initial_strength = torch.tensor(0.1, dtype=torch.float32)
        self.expansion_logit = nn.Parameter(torch.logit(initial_strength))

    def expansion_strength(self) -> Tensor:
        return torch.sigmoid(self.expansion_logit)

    def forward(
        self, observation: Tensor, hidden: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if observation.ndim != 2 or observation.shape[1] != self.sensory.in_features:
            raise ValueError(
                "Gated nested fixed graph observation has an incompatible shape."
            )
        if hidden.shape != (observation.shape[0], self.graph.node_count):
            raise ValueError(
                "Gated nested fixed graph hidden state has an incompatible shape."
            )

        sensory_drive = self.sensory(observation)
        injected = torch.zeros_like(hidden)
        injected = torch.index_copy(
            injected, 1, self.sensory_indices, sensory_drive
        )

        core_recurrent = torch.sparse.mm(self.core_adjacency, hidden.T).T
        expansion_recurrent = torch.sparse.mm(
            self.expansion_adjacency, hidden.T
        ).T

        core_gain = torch.clamp(self.recurrent_gain, min=0.0)
        candidate = torch.tanh(
            injected
            + core_gain * core_recurrent
            + self.expansion_strength() * expansion_recurrent
        )

        time_constant = torch.clamp(self.time_constant, min=1e-4, max=1.0)
        activity = hidden + time_constant * (candidate - hidden)
        readout_activity = activity.index_select(1, self.readout_indices)

        return (
            self.actor(readout_activity),
            self.critic(readout_activity).squeeze(-1),
            activity,
        )

    def effective_recurrent_edges(self) -> tuple[Tensor, Tensor]:
        """Return source-target indices and already-gained recurrent weights."""
        core = self.core_adjacency.coalesce()
        expansion = self.expansion_adjacency.coalesce()

        index_parts: list[Tensor] = []
        value_parts: list[Tensor] = []

        if core._nnz():
            index_parts.append(core.indices()[[1, 0]])
            value_parts.append(
                core.values() * torch.clamp(self.recurrent_gain, min=0.0)
            )
        if expansion._nnz():
            index_parts.append(expansion.indices()[[1, 0]])
            value_parts.append(
                expansion.values() * self.expansion_strength()
            )

        if not index_parts:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.float32),
            )
        return torch.cat(index_parts, dim=1), torch.cat(value_parts, dim=0)


class FeedbackNestedPopulationFixedGraphPolicy(NestedPopulationFixedGraphPolicy):
    """Core-preserving nested policy with always-active added-cell dynamics.

    Core->added and added->added edges stay active. Only added->core feedback
    is gated, so feedback=0 preserves the 80-cell core exactly without starving
    the 920-cell branch of activity.
    """

    def __init__(
        self,
        graph: ReducedGraphArtifact,
        core_graph: ReducedGraphArtifact,
        observation_size: int,
        actions: int = 5,
    ) -> None:
        super().__init__(graph, core_graph, observation_size, actions)
        del self.expansion_gain
        initial_strength = torch.tensor(0.1, dtype=torch.float32)
        self.feedback_logit = nn.Parameter(torch.logit(initial_strength))

        core_ids = set(int(body_id) for body_id in core_graph.body_ids)
        expansion = self.expansion_adjacency.coalesce()
        indices = expansion.indices()
        values = expansion.values()

        branch_sources: list[int] = []
        branch_targets: list[int] = []
        branch_weights: list[float] = []
        feedback_sources: list[int] = []
        feedback_targets: list[int] = []
        feedback_weights: list[float] = []

        for edge in range(values.numel()):
            target = int(indices[0, edge])
            source = int(indices[1, edge])
            weight = float(values[edge])
            if int(graph.body_ids[target]) in core_ids:
                feedback_sources.append(source)
                feedback_targets.append(target)
                feedback_weights.append(weight)
            else:
                branch_sources.append(source)
                branch_targets.append(target)
                branch_weights.append(weight)

        self.register_buffer(
            "branch_adjacency",
            self._make_sparse_adjacency(
                graph.node_count,
                branch_sources,
                branch_targets,
                branch_weights,
            ),
            persistent=False,
        )
        self.register_buffer(
            "feedback_adjacency",
            self._make_sparse_adjacency(
                graph.node_count,
                feedback_sources,
                feedback_targets,
                feedback_weights,
            ),
            persistent=False,
        )

    def feedback_strength(self) -> Tensor:
        return torch.sigmoid(self.feedback_logit)

    @property
    def branch_edge_count(self) -> int:
        return int(self.branch_adjacency._nnz())

    @property
    def feedback_edge_count(self) -> int:
        return int(self.feedback_adjacency._nnz())

    def forward(
        self, observation: Tensor, hidden: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if observation.ndim != 2 or observation.shape[1] != self.sensory.in_features:
            raise ValueError(
                "Feedback nested fixed graph observation has an incompatible shape."
            )
        if hidden.shape != (observation.shape[0], self.graph.node_count):
            raise ValueError(
                "Feedback nested fixed graph hidden state has an incompatible shape."
            )

        sensory_drive = self.sensory(observation)
        injected = torch.zeros_like(hidden)
        injected = torch.index_copy(
            injected, 1, self.sensory_indices, sensory_drive
        )

        core_recurrent = torch.sparse.mm(self.core_adjacency, hidden.T).T
        branch_recurrent = torch.sparse.mm(self.branch_adjacency, hidden.T).T
        feedback_recurrent = torch.sparse.mm(self.feedback_adjacency, hidden.T).T
        recurrent_gain = torch.clamp(self.recurrent_gain, min=0.0)

        candidate = torch.tanh(
            injected
            + recurrent_gain
            * (
                core_recurrent
                + branch_recurrent
                + self.feedback_strength() * feedback_recurrent
            )
        )

        time_constant = torch.clamp(self.time_constant, min=1e-4, max=1.0)
        activity = hidden + time_constant * (candidate - hidden)
        readout_activity = activity.index_select(1, self.readout_indices)
        return (
            self.actor(readout_activity),
            self.critic(readout_activity).squeeze(-1),
            activity,
        )

    def effective_recurrent_edges(self) -> tuple[Tensor, Tensor]:
        recurrent_gain = torch.clamp(self.recurrent_gain, min=0.0)
        groups = (
            (self.core_adjacency.coalesce(), recurrent_gain),
            (self.branch_adjacency.coalesce(), recurrent_gain),
            (
                self.feedback_adjacency.coalesce(),
                recurrent_gain * self.feedback_strength(),
            ),
        )

        index_parts: list[Tensor] = []
        value_parts: list[Tensor] = []
        for adjacency, gain in groups:
            if adjacency._nnz():
                index_parts.append(adjacency.indices()[[1, 0]])
                value_parts.append(adjacency.values() * gain)

        if not index_parts:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.float32),
            )
        return torch.cat(index_parts, dim=1), torch.cat(value_parts, dim=0)
