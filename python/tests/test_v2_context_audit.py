from __future__ import annotations

import numpy as np

from fly_crossy.v2.context_audit import (
    BRAIN_PROJECTED_SIZE,
    CONTEXT_SIZE,
    ContextDataset,
    build_condition_features,
    interpret,
    motor_context,
)


def _dataset() -> ContextDataset:
    brain = np.zeros((5, 21_022), dtype=np.float16)
    brain[:, 0] = np.arange(5, dtype=np.float16)
    return ContextDataset(
        brain=brain,
        expert_action=np.asarray([0, 0, 2, 3, 4], dtype=np.uint8),
        student_action=np.asarray([0, 2, 3, 4, 0], dtype=np.uint8),
        disagreement=np.asarray([False, True, True, True, True]),
        critical_error=np.asarray([False, False, True, False, True]),
        episode=np.asarray([0, 0, 0, 1, 1], dtype=np.int16),
        step=np.asarray([0, 1, 2, 0, 1], dtype=np.int16),
        fly_column=np.asarray([0, 1, 2, -1, -2], dtype=np.float32),
    )


def test_motor_context_never_crosses_episode_boundary() -> None:
    dataset = _dataset()

    first_second_episode = motor_context(
        dataset,
        3,
        include_column=True,
    )
    assert first_second_episode[:20].sum() == 0.0
    assert first_second_episode[20:24].sum() == 0.0
    assert first_second_episode[24] == -0.2


def test_every_condition_has_same_input_width() -> None:
    dataset = _dataset()
    conditions = (
        "brain_now",
        "brain_history4",
        "brain_now_efference",
        "brain_now_efference_column",
        "brain_history4_efference_column",
        "context_only_efference_column",
    )

    for index, condition in enumerate(conditions):
        x = build_condition_features(
            dataset,
            condition=condition,
            seed=100 + index,
        )
        assert x.shape == (
            len(dataset.expert_action),
            BRAIN_PROJECTED_SIZE + CONTEXT_SIZE,
        )
        assert np.isfinite(x).all()


def test_interpretation_identifies_combined_context_gain() -> None:
    def result(core: float, disagree: float, critical: float) -> dict[str, object]:
        return {
            "oof": {"coreMacroAccuracy": core},
            "disagreementRecovery": disagree,
            "criticalCorrection": critical,
        }

    conclusion = interpret(
        {
            "brain_now": result(0.50, 0.35, 0.40),
            "brain_history4": result(0.54, 0.38, 0.42),
            "brain_now_efference": result(0.55, 0.40, 0.43),
            "brain_now_efference_column": result(0.57, 0.42, 0.46),
            "brain_history4_efference_column": result(0.61, 0.46, 0.52),
            "context_only_efference_column": result(0.30, 0.20, 0.20),
        }
    )

    assert conclusion["strongContextEvidence"] is True
    assert conclusion["diagnosis"] == "combined_temporal_and_self_state_context_missing"
