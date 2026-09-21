from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

from fly_crossy.checkpoint import validate_checkpoint
from fly_crossy.env import WORLD_VERSION
from fly_crossy.models import (
    FeedbackNestedPopulationFixedGraphPolicy,
    FixedGraphPolicy,
    GatedNestedPopulationFixedGraphPolicy,
    NestedPopulationFixedGraphPolicy,
    PopulationFixedGraphPolicy,
)
from fly_crossy.protocol import Action, Observation
from fly_crossy.schema import (
    ACTION_ORDER,
    OBSERVATION_INPUT_SIZE,
)


class ConnectomeActionSelector:
    """Stateful deterministic runtime for one trained connectome checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        expected_environment_version: int = WORLD_VERSION,
    ) -> None:
        self._device = torch.device(device)

        path = Path(checkpoint_path)

        try:
            saved = torch.load(
                path,
                map_location=self._device,
                weights_only=True,
            )
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Unable to load connectome checkpoint: {path}"
            ) from exc

        checkpoint = validate_checkpoint(
            saved,
            expected_controller="connectome",
            expected_environment_version=expected_environment_version,
            expected_observation_size=OBSERVATION_INPUT_SIZE,
        )

        graph = checkpoint.graph

        if graph is None:
            raise ValueError(
                "Connectome checkpoint has no validated graph."
            )

        if checkpoint.connectome_interface == "nested-feedback":
            core_graph = checkpoint.core_graph
            if core_graph is None:
                raise ValueError(
                    "Nested connectome checkpoint has no validated core graph."
                )
            model = FeedbackNestedPopulationFixedGraphPolicy(
                graph,
                core_graph,
                checkpoint.observation_size,
                checkpoint.actions,
            )
        elif checkpoint.connectome_interface == "nested-gated":
            core_graph = checkpoint.core_graph
            if core_graph is None:
                raise ValueError(
                    "Nested connectome checkpoint has no validated core graph."
                )
            model = GatedNestedPopulationFixedGraphPolicy(
                graph,
                core_graph,
                checkpoint.observation_size,
                checkpoint.actions,
            )
        elif checkpoint.connectome_interface == "nested":
            core_graph = checkpoint.core_graph
            if core_graph is None:
                raise ValueError(
                    "Nested connectome checkpoint has no validated core graph."
                )
            model = NestedPopulationFixedGraphPolicy(
                graph,
                core_graph,
                checkpoint.observation_size,
                checkpoint.actions,
            )
        elif checkpoint.connectome_interface == "population":
            model = PopulationFixedGraphPolicy(
                graph,
                checkpoint.observation_size,
                checkpoint.actions,
            )
        else:
            model = FixedGraphPolicy(
                graph,
                checkpoint.observation_size,
                checkpoint.actions,
            )

        model.load_state_dict(
            checkpoint.state_dict
        )

        model.to(self._device)
        model.eval()

        self._model = model

        self._hidden = torch.zeros(
            1,
            graph.node_count,
            dtype=torch.float32,
            device=self._device,
        )

        self._episode_id: str | None = None

        self._last_logits = torch.zeros(
            len(ACTION_ORDER),
            dtype=torch.float32,
        )

    @property
    def node_count(self) -> int:
        return self._model.graph.node_count

    @property
    def body_ids(self) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in self._model.graph.body_ids
        )

    @property
    def activity(self) -> tuple[float, ...]:
        """Display-only normalized activity in [0, 1]."""

        normalized = (
            self._model
            .normalized_activity(self._hidden)
            .squeeze(0)
            .detach()
            .cpu()
        )

        return tuple(
            float(value)
            for value in normalized.tolist()
        )

    @property
    def logits(self) -> tuple[float, ...]:
        return tuple(
            float(value)
            for value in self._last_logits.tolist()
        )

    def reset(self) -> None:
        self._hidden.zero_()
        self._episode_id = None
        self._last_logits.zero_()

    def __call__(
        self,
        observation: Observation,
    ) -> Action:
        if observation.episode_id != self._episode_id:
            self._hidden.zero_()
            self._episode_id = observation.episode_id

        values = torch.tensor(
            observation.observation,
            dtype=torch.float32,
            device=self._device,
        ).unsqueeze(0)

        with torch.inference_mode():
            logits, _, next_hidden = self._model(
                values,
                self._hidden,
            )

        self._hidden = next_hidden.detach()

        self._last_logits = (
            logits
            .squeeze(0)
            .detach()
            .cpu()
        )

        action_index = int(
            torch.argmax(
                logits,
                dim=1,
            ).item()
        )

        return cast(
            Action,
            ACTION_ORDER[action_index].value,
        )