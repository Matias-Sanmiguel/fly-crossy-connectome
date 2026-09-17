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
    assert [action.value for action in ACTION_ORDER] == [
        "forward",
        "backward",
        "left",
        "right",
        "wait",
    ]


def test_flatten_observation_matches_browser_encoding_order() -> None:
    cells = [[0 for _ in range(11)] for _ in range(11)]
    motion = [[[0.0, 0.0] for _ in range(11)] for _ in range(11)]
    cells[0][0] = 8
    motion[0][0] = [-1.0, 1.0 / 3.0]
    observation = ObservationV1(
        cells=cells,
        motion=motion,
        support=1,
        previous_action=Action.LEFT,
        edge_distance=0.4,
    )

    encoded = flatten_observation(observation)

    assert encoded.shape == (OBSERVATION_INPUT_SIZE,)
    assert encoded.dtype == np.float32
    assert OBSERVATION_VERSION == 2
    assert observation.version == 2
    assert encoded[0] == pytest.approx(1.0)
    assert encoded[1:121].tolist() == [0.0] * 120
    assert encoded[121:125].tolist() == pytest.approx([-1.0, 1.0 / 3.0, 0.0, 0.0])
    assert encoded[-7:].tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.4])


def test_flatten_observation_rejects_wrong_shape_or_nonfinite_values() -> None:
    valid_motion = [[[0.0, 0.0] for _ in range(11)] for _ in range(11)]
    malformed = ObservationV1(
        cells=[[0]],
        motion=valid_motion,
        support=0,
        previous_action=Action.WAIT,
        edge_distance=1.0,
    )
    nonfinite = ObservationV1(
        cells=[[0 for _ in range(11)] for _ in range(11)],
        motion=valid_motion,
        support=0,
        previous_action=Action.WAIT,
        edge_distance=float("nan"),
    )

    with pytest.raises(ValueError, match="370 finite values"):
        flatten_observation(malformed)
    with pytest.raises(ValueError, match="370 finite values"):
        flatten_observation(nonfinite)
