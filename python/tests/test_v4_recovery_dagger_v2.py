from __future__ import annotations

import numpy as np
import torch

from fly_crossy.schema import ACTION_ORDER
from fly_crossy.v4.recovery_dagger_v2 import (
    RecoveryDataV2,
    RecoveryDecoderV2,
    compute_training_weights,
    concat_data,
    progress_preference_loss,
    selection_key,
)


def _tiny(
    source: str,
    *,
    recovery: bool,
    behavior: int,
    safe_behavior: bool,
    progress_index: int | None = 0,
) -> RecoveryDataV2:
    n_actions = len(ACTION_ORDER)
    safe = np.ones((1, n_actions), dtype=np.bool_)
    safe[0, behavior] = safe_behavior
    acceptable = np.zeros((1, n_actions), dtype=np.bool_)
    acceptable[0, 0] = True
    progress = np.zeros((1, n_actions), dtype=np.bool_)
    if progress_index is not None:
        progress[0, progress_index] = True
    return RecoveryDataV2(
        motor=np.zeros((1, 708), dtype=np.float32),
        acceptable=acceptable,
        safe=safe,
        progress=progress,
        pair_sign=np.zeros((1, 10), dtype=np.int8),
        pair_weight=np.ones((1, 10), dtype=np.float32),
        best_action=np.asarray([0], dtype=np.int64),
        semantic=np.zeros((1, 34), dtype=np.float32),
        behavior_action=np.asarray([behavior], dtype=np.int64),
        recovery=np.asarray([recovery], dtype=np.bool_),
        source=np.asarray([source]),
    )


def test_decoder_architectures_shape_and_parameter_order():
    linear = RecoveryDecoderV2(
        708, mean=torch.zeros(708), std=torch.ones(708), architecture="linear"
    )
    mlp = RecoveryDecoderV2(
        708, mean=torch.zeros(708), std=torch.ones(708), architecture="mlp64"
    )
    assert linear(torch.zeros(3, 708)).shape == (3, len(ACTION_ORDER))
    assert mlp(torch.zeros(3, 708)).shape == (3, len(ACTION_ORDER))
    assert sum(p.numel() for p in mlp.parameters()) > sum(p.numel() for p in linear.parameters())


def test_mlp_residual_starts_at_zero():
    decoder = RecoveryDecoderV2(
        708, mean=torch.zeros(708), std=torch.ones(708), architecture="mlp64"
    )
    x = torch.randn(2, 708)
    normalized = (x - decoder.mean) / decoder.std
    assert torch.allclose(decoder(x), decoder.linear(normalized))


def test_concat_data_preserves_progress_and_source():
    a = _tiny("teacher", recovery=False, behavior=0, safe_behavior=True)
    b = _tiny("onpolicy", recovery=True, behavior=1, safe_behavior=False, progress_index=None)
    merged = concat_data([a, b])
    assert merged.length == 2
    assert merged.source.tolist() == ["teacher", "onpolicy"]
    assert merged.recovery.tolist() == [False, True]
    assert merged.progress.shape == (2, len(ACTION_ORDER))


def test_training_weights_focus_on_unsafe_onpolicy_not_action_class():
    anchor = _tiny("teacher", recovery=False, behavior=0, safe_behavior=True)
    risky = _tiny("onpolicy", recovery=True, behavior=1, safe_behavior=False)
    merged = concat_data([anchor, risky])
    weights, report = compute_training_weights(merged)
    assert weights[1] > weights[0]
    assert report["classBalancing"] is False


def test_progress_loss_rewards_acceptable_progress_action():
    logits_good = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0]])
    logits_bad = torch.tensor([[0.0, 0.0, 0.0, 0.0, 3.0]])
    acceptable = torch.tensor([[True, False, False, False, True]])
    progress = torch.tensor([[True, False, False, False, False]])
    weights = torch.ones(1)
    good = progress_preference_loss(logits_good, acceptable, progress, weights)
    bad = progress_preference_loss(logits_bad, acceptable, progress, weights)
    assert good < bad


def test_progress_loss_ignores_states_without_acceptable_progress():
    logits = torch.randn(2, len(ACTION_ORDER), requires_grad=True)
    acceptable = torch.tensor(
        [[True, False, False, False, False], [False, False, True, False, False]]
    )
    progress = torch.zeros_like(acceptable)
    weights = torch.ones(2)
    loss = progress_preference_loss(logits, acceptable, progress, weights)
    assert float(loss.detach()) == 0.0


def _metrics(*, success: bool, score: float, robust_success: int, robust_mean: float, stalled: int):
    return {
        "exactExpo": {"success": success, "score": score},
        "robust": {
            "successes": robust_success,
            "meanScore": robust_mean,
            "medianScore": robust_mean,
            "stalled": stalled,
        },
    }


def test_selection_does_not_reward_200_step_stall_over_real_progress():
    # The V1 exploit (survive at score 26) must lose to a non-successful model
    # that genuinely reaches score 29.  In V2 both have success=False unless the
    # progress threshold is met, so score decides instead of reached-step-limit.
    stalled = _metrics(success=False, score=26.0, robust_success=3, robust_mean=8.7, stalled=3)
    progressed = _metrics(success=False, score=29.0, robust_success=0, robust_mean=10.6, stalled=0)
    assert selection_key(progressed) > selection_key(stalled)


def test_true_success_beats_higher_score_failure():
    success = _metrics(success=True, score=100.0, robust_success=1, robust_mean=20.0, stalled=0)
    failure = _metrics(success=False, score=120.0, robust_success=10, robust_mean=80.0, stalled=0)
    assert selection_key(success) > selection_key(failure)
