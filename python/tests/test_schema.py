from __future__ import annotations

import numpy as np
import pytest

from fly_crossy.schema import (
    ACTION_ORDER,
    OBSERVATION_INPUT_SIZE,
    OBSERVATION_VERSION,
    Action,
    ObservationV1,
    flatten_observation,
)


def test_action_order_matches_browser_policy_contract() -> None:
    assert ACTION_ORDER == (
        Action.FORWARD,
        Action.BACKWARD,
        Action.LEFT,
        Action.RIGHT,
        Action.WAIT,
    )


def test_flatten_observation_v3_matches_browser_encoding_order() -> None:
    cells = [[0 for _ in range(11)] for _ in range(11)]
    motion = [[[0.0, 0.0] for _ in range(11)] for _ in range(11)]
    hazard_offset = [[0.0 for _ in range(11)] for _ in range(11)]
    cells[0][0] = 8
    motion[0][0] = [-1.0, 1.0 / 3.0]
    hazard_offset[0][0] = 0.25
    observation = ObservationV1(
        cells=cells,
        motion=motion,
        hazard_offset=hazard_offset,
        support=1,
        previous_action=Action.LEFT,
        edge_distance=0.4,
        signed_column=-0.6,
    )

    encoded = flatten_observation(observation)

    assert OBSERVATION_VERSION == 3
    assert observation.version == 3
    assert OBSERVATION_INPUT_SIZE == 492
    assert encoded.shape == (492,)
    assert encoded.dtype == np.float32
    assert encoded[0] == pytest.approx(1.0)
    assert encoded[121:125].tolist() == pytest.approx(
        [-1.0, 1.0 / 3.0, 0.0, 0.0]
    )
    assert encoded[363] == pytest.approx(0.25)
    assert encoded[484] == pytest.approx(1.0)
    assert encoded[485:490].tolist() == pytest.approx(
        [0.0, 0.0, 1.0, 0.0, 0.0]
    )
    assert encoded[490] == pytest.approx(0.4)
    assert encoded[491] == pytest.approx(-0.6)


def test_flatten_observation_rejects_wrong_shape_or_nonfinite_values() -> None:
    valid_motion = [[[0.0, 0.0] for _ in range(11)] for _ in range(11)]
    valid_offset = [[0.0 for _ in range(11)] for _ in range(11)]
    malformed = ObservationV1(
        cells=[[0]],
        motion=valid_motion,
        hazard_offset=valid_offset,
        support=0,
        previous_action=Action.WAIT,
        edge_distance=1.0,
        signed_column=0.0,
    )
    nonfinite = ObservationV1(
        cells=[[0 for _ in range(11)] for _ in range(11)],
        motion=valid_motion,
        hazard_offset=valid_offset,
        support=0,
        previous_action=Action.WAIT,
        edge_distance=1.0,
        signed_column=float("nan"),
    )

    with pytest.raises(ValueError, match="492 finite values"):
        flatten_observation(malformed)
    with pytest.raises(ValueError, match="492 finite values"):
        flatten_observation(nonfinite)
