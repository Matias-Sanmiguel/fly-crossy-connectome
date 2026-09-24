from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


class Action(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    WAIT = "wait"


ACTION_ORDER = (
    Action.FORWARD,
    Action.BACKWARD,
    Action.LEFT,
    Action.RIGHT,
    Action.WAIT,
)
OBSERVATION_RADIUS = 5
OBSERVATION_VERSION = 3
OBSERVATION_INPUT_SIZE = 492


@dataclass(frozen=True, slots=True)
class ObservationV1:
    cells: Sequence[Sequence[int]]
    motion: Sequence[Sequence[Sequence[float]]]
    hazard_offset: Sequence[Sequence[float]]
    support: int
    previous_action: Action
    edge_distance: float
    signed_column: float
    version: int = OBSERVATION_VERSION
    radius: int = OBSERVATION_RADIUS


def flatten_observation(observation: ObservationV1) -> NDArray[np.float32]:
    """Encode ObservationV3 in the browser policy's stable numeric order."""
    cells = [value / 8 for row in observation.cells for value in row]
    motion = [value for row in observation.motion for cell in row for value in cell]
    hazard_offset = [value for row in observation.hazard_offset for value in row]
    previous_action = [float(observation.previous_action == action) for action in ACTION_ORDER]
    encoded = np.asarray(
        [
            *cells,
            *motion,
            *hazard_offset,
            observation.support,
            *previous_action,
            observation.edge_distance,
            observation.signed_column,
        ],
        dtype=np.float32,
    )
    if encoded.shape != (OBSERVATION_INPUT_SIZE,) or not np.isfinite(encoded).all():
        raise ValueError(
            f"Observation must encode to {OBSERVATION_INPUT_SIZE} finite values."
        )
    return encoded
