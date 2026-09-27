from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from fly_crossy.v7.contracts import ACTION_NAMES
from fly_crossy.v7.dataset import TransitionDataset
from fly_crossy.v7.teacher import (
    PrivilegedTeacher,
    fit_teacher_epoch,
    teacher_agreement,
    teacher_behavior,
    teacher_loss,
)


def _dataset(rows: int = 10) -> TransitionDataset:
    acceptable = np.zeros((rows, 5), dtype=np.bool_)
    acceptable[:, 0] = True
    safe = np.ones((rows, 5), dtype=np.bool_)
    safe[:, 4] = False
    pair_sign = np.zeros((rows, 10), dtype=np.int8)
    pair_sign[:, :4] = 1
    pair_weight = np.zeros((rows, 10), dtype=np.float32)
    pair_weight[:, :4] = 1.0
    values = np.zeros((rows, 5, 6), dtype=np.float32)
    values[:, 0, 0] = 1.0
    return TransitionDataset(
        frames=np.zeros((rows, 24, 48, 3), dtype=np.uint8),
        observations=np.linspace(-1, 1, rows * 517, dtype=np.float32).reshape(rows, 517),
        planner_values=values,
        acceptable=acceptable,
        immediate_safe=safe,
        pair_sign=pair_sign,
        pair_weight=pair_weight,
        best_action=np.zeros(rows, dtype=np.int64),
        behavior_action=np.zeros(rows, dtype=np.int64),
        episode_seed=np.asarray(["unit-v7-80-training-000"] * rows),
        step_index=np.arange(rows, dtype=np.int64),
        source=np.asarray(["planner"] * rows),
        action_order=ACTION_NAMES,
    )


def test_teacher_has_exact_stateless_517_128_128_5_architecture() -> None:
    model = PrivilegedTeacher()
    layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]

    assert [(layer.in_features, layer.out_features) for layer in layers] == [
        (517, 128),
        (128, 128),
        (128, 5),
    ]
    observations = torch.zeros(3, 517)
    assert model(observations).shape == (3, 5)
    assert torch.equal(model(observations), model(observations))


def test_teacher_rejects_rgb_input_or_reordered_actions() -> None:
    model = PrivilegedTeacher()
    with pytest.raises(ValueError, match="517"):
        model(torch.zeros(2, 24, 48, 3))
    with pytest.raises(ValueError, match="canonical ACTION_ORDER"):
        PrivilegedTeacher(action_order=tuple(reversed(ACTION_NAMES)))


def test_teacher_loss_is_finite_and_reports_all_components() -> None:
    model = PrivilegedTeacher()
    dataset = _dataset()
    rows = np.arange(len(dataset.frames))
    logits = model(torch.from_numpy(dataset.observations))

    loss = teacher_loss(logits, dataset, rows)

    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.preference)
    assert torch.isfinite(loss.acceptable)
    assert torch.isfinite(loss.safety)
    assert loss.total.requires_grad


def test_teacher_loss_rejects_nonfinite_logits() -> None:
    dataset = _dataset(2)
    logits = torch.zeros(2, 5)
    logits[0, 0] = torch.nan

    with pytest.raises(FloatingPointError, match="logits"):
        teacher_loss(logits, dataset, np.arange(2))


def test_fit_teacher_epoch_performs_a_real_deterministic_update() -> None:
    torch.manual_seed(91)
    model = PrivilegedTeacher()
    dataset = _dataset()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = fit_teacher_epoch(
        model,
        dataset,
        optimizer,
        batch_size=4,
        seed=123,
    )

    assert set(metrics) == {"total", "preference", "acceptable", "safety"}
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters(), strict=True)
    )


def test_teacher_agreement_uses_planner_acceptable_set() -> None:
    model = PrivilegedTeacher()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output.bias[0] = 1.0
    dataset = _dataset(10)

    assert teacher_agreement(model, dataset, torch.device("cpu")) == 1.0

    dataset.acceptable[0, 0] = False
    dataset.acceptable[0, 1] = True
    assert teacher_agreement(model, dataset, torch.device("cpu")) == pytest.approx(0.9)


def test_teacher_behavior_uses_only_structured_observation() -> None:
    model = PrivilegedTeacher()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output.bias[2] = 2.0
    behavior = teacher_behavior(model, torch.device("cpu"))
    behavior.reset()

    action = behavior.act(
        state=None,  # type: ignore[arg-type]
        frame=np.full((24, 48, 3), 255, dtype=np.uint8),
        observation=np.zeros(517, dtype=np.float32),
    )

    assert action == 2
