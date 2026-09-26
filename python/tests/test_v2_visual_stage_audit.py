from __future__ import annotations

import numpy as np
import torch

from fly_crossy.v2.visual_stage_audit import (
    CAMERA_CURRENT_SIZE,
    NEURAL_FEATURE_SIZE,
    camera_delta_features,
    camera_now_features,
    interpret,
    make_sketch,
)


def test_camera_feature_contract_is_equal_width() -> None:
    current = torch.arange(CAMERA_CURRENT_SIZE, dtype=torch.float32)
    previous = current - 1.0

    now = camera_now_features(current)
    delta = camera_delta_features(current, previous)

    assert now.shape == (NEURAL_FEATURE_SIZE,)
    assert delta.shape == (NEURAL_FEATURE_SIZE,)
    assert torch.all(now[CAMERA_CURRENT_SIZE:] == 0)
    assert torch.all(delta[CAMERA_CURRENT_SIZE:] == 1)


def test_countsketch_is_deterministic() -> None:
    device = torch.device("cpu")
    a = make_sketch(100, 32, seed=7, device=device)
    b = make_sketch(100, 32, seed=7, device=device)
    values = torch.arange(100, dtype=torch.float32)

    assert torch.equal(a.bucket, b.bucket)
    assert torch.equal(a.sign, b.sign)
    assert torch.allclose(a.project(values), b.project(values))


def test_interpretation_can_localize_retina_loss() -> None:
    def result(core: float, disagree: float, critical: float) -> dict[str, object]:
        return {
            "oof": {"coreMacroAccuracy": core},
            "disagreementRecovery": disagree,
            "criticalCorrection": critical,
        }

    conclusion = interpret(
        {
            "brain_readout": result(0.50, 0.35, 0.40),
            "retina_sequence": result(0.53, 0.37, 0.42),
            "camera_now": result(0.55, 0.40, 0.44),
            "camera_rgb_delta": result(0.66, 0.50, 0.54),
            "symbolic_oracle": result(0.80, 0.70, 0.72),
        }
    )

    assert conclusion["diagnosis"] == "retina_frontend_bottleneck"
    assert conclusion["cameraDeltaVsRetina"]["strong"] is True
