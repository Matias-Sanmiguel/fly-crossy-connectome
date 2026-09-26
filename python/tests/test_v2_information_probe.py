from __future__ import annotations

import numpy as np

from fly_crossy.schema import ACTION_ORDER
from fly_crossy.v2.information_probe import (
    CORE_ACTION_INDICES,
    class_weights,
    metrics_from_predictions,
)


def test_probe_metrics_do_not_hide_forward_collapse() -> None:
    # 80% forward labels, but a forward-only predictor should still have poor
    # balanced/core macro accuracy.
    labels = np.asarray(
        [0] * 80 + [2] * 10 + [3] * 5 + [4] * 5,
        dtype=np.uint8,
    )
    predicted = np.zeros_like(labels)

    metrics = metrics_from_predictions(labels, predicted)

    assert metrics["accuracy"] == 0.8
    assert metrics["coreMacroAccuracy"] == 0.25
    assert metrics["perActionAccuracy"]["left"] == 0.0
    assert metrics["perActionAccuracy"]["right"] == 0.0
    assert metrics["perActionAccuracy"]["wait"] == 0.0


def test_probe_class_weights_are_finite_and_moderate() -> None:
    labels = np.asarray(
        [0] * 100 + [1] * 2 + [2] * 20 + [3] * 20 + [4] * 10,
        dtype=np.uint8,
    )
    weights = class_weights(labels)

    assert weights.shape == (len(ACTION_ORDER),)
    assert np.isfinite(weights).all()
    assert np.all(weights >= 0.35)
    assert np.all(weights <= 6.0)
    assert weights[1] > weights[0]


def test_core_macro_action_set_is_forward_left_right_wait() -> None:
    assert tuple(CORE_ACTION_INDICES) == (0, 2, 3, 4)
