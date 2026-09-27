from __future__ import annotations

from pathlib import Path
from typing import Sequence, cast

import numpy as np
import torch

from fly_crossy.checkpoint import validate_checkpoint
from fly_crossy.env import WORLD_VERSION
from fly_crossy.models import (
    FeedbackNestedPopulationFixedGraphPolicy,
    FixedGraphPolicy,
    GatedNestedPopulationFixedGraphPolicy,
    NestedPopulationFixedGraphPolicy,
    PopulationFixedGraphPolicy,
    SettledPopulationFixedGraphPolicy,
    TrafficAwarePopulationPolicy,
)
from fly_crossy.protocol import Action, Observation
from fly_crossy.schema import (
    ACTION_ORDER,
    OBSERVATION_INPUT_SIZE,
)


LEGACY_OBSERVATION_INPUT_SIZE = 370


def adapt_current_observation_to_v1(values: Sequence[float]) -> tuple[float, ...]:
    """Project encoded ObservationV4 onto the released ObservationV1 contract."""
    current = np.asarray(values, dtype=np.float32)
    if current.shape != (OBSERVATION_INPUT_SIZE,) or not np.isfinite(current).all():
        raise ValueError(
            f"Current observation must contain {OBSERVATION_INPUT_SIZE} finite values."
        )

    raw_cells = np.rint(current[:121] * 8).astype(np.int16)
    legacy_cells = np.where(raw_cells == 8, 0.0, raw_cells / 7.0)
    current_motion = current[121:363].reshape(121, 2)
    legacy_motion = current_motion.copy()
    speed_scale = np.where(raw_cells == 6, 12.0, 5.0)
    legacy_motion[:, 1] = np.minimum(
        1.0,
        current_motion[:, 1] * speed_scale / 3.0,
    )
    legacy = np.concatenate(
        (
            legacy_cells,
            legacy_motion.reshape(-1),
            current[484:491],
        )
    ).astype(np.float32, copy=False)
    if legacy.shape != (LEGACY_OBSERVATION_INPUT_SIZE,):
        raise RuntimeError("Legacy observation adapter produced an invalid shape.")
    return tuple(float(value) for value in legacy)


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

        expected_observation_size = (
            OBSERVATION_INPUT_SIZE
            if expected_environment_version == WORLD_VERSION
            else None
        )
        checkpoint = validate_checkpoint(
            saved,
            expected_controller="connectome",
            expected_environment_version=expected_environment_version,
            expected_observation_size=expected_observation_size,
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
        elif checkpoint.connectome_interface == "population-settled":
            model = SettledPopulationFixedGraphPolicy(
                graph,
                checkpoint.observation_size,
                checkpoint.actions,
            )
        elif checkpoint.connectome_interface == "controller-v2":
            model = TrafficAwarePopulationPolicy(
                graph,
                checkpoint.observation_size,
                checkpoint.actions,
            )
        elif checkpoint.connectome_interface in (
            "population",
            "population-wide-predictive",
        ):
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
        self._observation_size = checkpoint.observation_size

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

        encoded = observation.observation
        if (
            self._observation_size == LEGACY_OBSERVATION_INPUT_SIZE
            and len(encoded) == OBSERVATION_INPUT_SIZE
        ):
            encoded = adapt_current_observation_to_v1(encoded)

        values = torch.tensor(
            encoded,
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
