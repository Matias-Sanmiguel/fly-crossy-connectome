from __future__ import annotations

import numpy as np
import pytest
import torch

from fly_crossy.v7.contracts import ACTION_NAMES
from fly_crossy.v7.dataset import TransitionDataset
from fly_crossy.v7.student import StudentOutput, build_visual_student
from fly_crossy.v7.teacher import PrivilegedTeacher
from fly_crossy.v7.training import (
    StudentBatch,
    attach_teacher_logits,
    _planner_row_weights,
    fit_student_epoch,
    student_behavior,
    student_loss,
)


def _dataset(rows: int = 6) -> TransitionDataset:
    rng = np.random.default_rng(8)
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
        frames=rng.integers(0, 256, (rows, 24, 48, 3), dtype=np.uint8),
        observations=rng.normal(size=(rows, 517)).astype(np.float32),
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


def _batch(rows: int = 3) -> StudentBatch:
    return StudentBatch(
        frames=torch.rand(rows, 24, 48, 3),
        acceptable=torch.tensor([[True, False, False, False, False]] * rows),
        immediate_safe=torch.tensor([[True, True, True, True, False]] * rows),
        pair_sign=torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]] * rows),
        pair_weight=torch.tensor([[1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0]] * rows),
        best_action=torch.zeros(rows, dtype=torch.long),
        best_action_weight=torch.ones(rows),
        teacher_logits=torch.tensor([[2.0, 0.0, 0.0, 0.0, -1.0]] * rows),
        mask=torch.ones(rows, dtype=torch.bool),
    )


def test_attach_teacher_logits_is_finite_deterministic_and_rgb_independent() -> None:
    dataset = _dataset()
    teacher = PrivilegedTeacher()
    first = attach_teacher_logits(dataset, teacher, torch.device("cpu"))
    dataset.frames[:] = 0
    second = attach_teacher_logits(dataset, teacher, torch.device("cpu"))

    assert first.shape == (6, 5)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)


def test_student_loss_has_exact_weighted_components_and_temperature_two_kl() -> None:
    logits = torch.tensor([[0.5, -0.2, 0.1, 0.0, -0.4]] * 3, requires_grad=True)
    output = StudentOutput(logits, torch.zeros(3, 4), torch.zeros(3, 4))
    trust = torch.tensor(3.0, requires_grad=True)

    loss = student_loss(output, _batch(), trust_penalty=trust)

    assert torch.isfinite(loss.total)
    assert loss.distillation > 0
    assert loss.preference > 0
    assert loss.safety > 0
    assert loss.planner > 0
    assert loss.trust is trust
    assert torch.allclose(
        loss.total,
        loss.distillation
        + loss.preference
        + 2.0 * loss.safety
        + 0.75 * loss.planner
        + 0.01 * trust,
    )


@pytest.mark.parametrize(
    "field", ["teacher_logits", "pair_weight", "best_action_weight"]
)
def test_student_loss_rejects_nonfinite_inputs(field: str) -> None:
    batch = _batch()
    getattr(batch, field).view(-1)[0] = torch.nan
    output = StudentOutput(torch.zeros(3, 5), torch.zeros(3, 2), torch.zeros(3, 2))
    with pytest.raises(FloatingPointError):
        student_loss(output, batch, trust_penalty=torch.tensor(0.0))


def test_planner_row_weights_upweight_rare_best_actions() -> None:
    dataset = _dataset(10)
    dataset.best_action[:] = 0
    dataset.best_action[-1] = 3

    weights = _planner_row_weights(dataset)

    assert weights.shape == (10,)
    assert weights.dtype == np.float32
    assert weights[-1] > weights[0]
    assert weights.mean() == pytest.approx(1.0)


def test_joint_update_changes_visual_dynamics_and_motor_but_not_topology() -> None:
    torch.manual_seed(12)
    model = build_visual_student("80", device=torch.device("cpu"))
    dataset = _dataset()
    teacher_logits = np.tile(
        np.asarray([[3.0, 0.0, -1.0, -2.0, -3.0]], dtype=np.float32),
        (len(dataset.frames), 1),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=5e-3,
    )
    tracked = (
        "retina.retina_kernel",
        "retina.sensory_gain",
        "retina.sensory_bias",
        "dynamics.recurrent_gain",
        "dynamics.time_constant",
        "dynamics.expansion_logit",
        "motor_head.0.weight",
        "motor_head.2.weight",
    )
    before = {name: dict(model.named_parameters())[name].detach().clone() for name in tracked}
    core_indices = model.dynamics.core_adjacency.indices().clone()
    core_values = model.dynamics.core_adjacency.values().clone()
    expansion_indices = model.dynamics.expansion_adjacency.indices().clone()
    expansion_values = model.dynamics.expansion_adjacency.values().clone()

    metrics = fit_student_epoch(
        model,
        dataset,
        teacher_logits,
        optimizer,
        batch_size=4,
        window=3,
        seed=41,
    )

    assert set(metrics) == {
        "total",
        "distillation",
        "preference",
        "safety",
        "planner",
        "trust",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    after = dict(model.named_parameters())
    assert all(not torch.equal(before[name], after[name].detach()) for name in tracked)
    assert torch.equal(core_indices, model.dynamics.core_adjacency.indices())
    assert torch.equal(core_values, model.dynamics.core_adjacency.values())
    assert torch.equal(expansion_indices, model.dynamics.expansion_adjacency.indices())
    assert torch.equal(expansion_values, model.dynamics.expansion_adjacency.values())


def test_fit_aborts_before_step_on_nonfinite_teacher_logits() -> None:
    model = build_visual_student("80", device=torch.device("cpu"))
    dataset = _dataset(2)
    teacher_logits = np.zeros((2, 5), dtype=np.float32)
    teacher_logits[0, 0] = np.inf
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    with pytest.raises(FloatingPointError):
        fit_student_epoch(
            model, dataset, teacher_logits, optimizer, batch_size=2, window=2, seed=1
        )
    assert all(
        torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters(), strict=True)
    )


def test_student_behavior_uses_frame_and_preserves_recurrent_state() -> None:
    model = build_visual_student("80", device=torch.device("cpu"))
    behavior = student_behavior(model, torch.device("cpu"))
    behavior.reset()
    frame = np.zeros((24, 48, 3), dtype=np.uint8)
    observation = np.full(517, np.nan, dtype=np.float32)
    first = behavior.act(None, frame, observation)  # type: ignore[arg-type]
    second = behavior.act(None, frame, observation)  # type: ignore[arg-type]
    assert 0 <= first < 5
    assert 0 <= second < 5
    behavior.reset()
