from __future__ import annotations

import numpy as np
import torch

from fly_crossy.schema import ACTION_ORDER
from fly_crossy.v4.recovery_dagger import (
    RecoveryData,
    RecoveryDecoder,
    _choose_safe_perturbation,
    compute_training_weights,
    concat_data,
)


def _tiny(source: str, *, recovery: bool, behavior: int, safe_behavior: bool) -> RecoveryData:
    n_actions = len(ACTION_ORDER)
    safe = np.ones((1, n_actions), dtype=np.bool_)
    safe[0, behavior] = safe_behavior
    acceptable = np.zeros((1, n_actions), dtype=np.bool_)
    acceptable[0, 0] = True
    return RecoveryData(
        motor=np.zeros((1, 708), dtype=np.float32),
        acceptable=acceptable,
        safe=safe,
        pair_sign=np.zeros((1, 10), dtype=np.int8),
        pair_weight=np.ones((1, 10), dtype=np.float32),
        best_action=np.asarray([0], dtype=np.int64),
        semantic=np.zeros((1, 34), dtype=np.float32),
        behavior_action=np.asarray([behavior], dtype=np.int64),
        recovery=np.asarray([recovery], dtype=np.bool_),
        source=np.asarray([source]),
    )


def test_recovery_decoder_shape():
    decoder = RecoveryDecoder(
        708,
        mean=torch.zeros(708),
        std=torch.ones(708),
    )
    out = decoder(torch.zeros(3, 708))
    assert out.shape == (3, len(ACTION_ORDER))


def test_concat_data_preserves_rows():
    a = _tiny("teacher", recovery=False, behavior=0, safe_behavior=True)
    b = _tiny("onpolicy", recovery=True, behavior=1, safe_behavior=False)
    merged = concat_data([a, b])
    assert merged.length == 2
    assert merged.source.tolist() == ["teacher", "onpolicy"]
    assert merged.recovery.tolist() == [False, True]


def test_unsafe_recovery_state_gets_larger_weight_than_teacher_anchor():
    anchor = _tiny("teacher", recovery=False, behavior=0, safe_behavior=True)
    risky = _tiny("onpolicy", recovery=True, behavior=1, safe_behavior=False)
    merged = concat_data([anchor, risky])
    weights, _ = compute_training_weights(merged)
    assert weights[1] > weights[0]


def test_safe_perturbation_never_returns_unsafe_action():
    rng = np.random.default_rng(7)
    acceptable = np.asarray([True, False, False, False, False])
    safe = np.asarray([True, False, True, False, True])
    for _ in range(20):
        action = _choose_safe_perturbation(
            acceptable=acceptable,
            safe=safe,
            best_index=0,
            rng=rng,
        )
        assert action in {2, 4}
