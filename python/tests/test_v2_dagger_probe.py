from __future__ import annotations

import numpy as np

from fly_crossy.v2.dagger_probe import (
    DaggerProbeDataset,
    evaluate_oof,
    make_episode_folds,
    sample_weights,
)


def _dataset() -> DaggerProbeDataset:
    labels = np.asarray([0, 2, 3, 4, 0, 2, 3, 4], dtype=np.uint8)
    student = np.asarray([0, 0, 3, 3, 0, 2, 0, 4], dtype=np.uint8)
    disagreement = labels != student
    critical = np.asarray([False, True, False, False, False, False, True, False])

    return DaggerProbeDataset(
        features=np.zeros((8, 21_022), dtype=np.float16),
        expert_action=labels,
        student_action=student,
        disagreement=disagreement,
        critical_error=critical,
        student_terminal=np.asarray([0, 1, 0, 0, 0, 0, 3, 0], dtype=np.uint8),
        episode_terminal=np.asarray([1, 1, 0, 0, 3, 3, 3, 3], dtype=np.uint8),
        steps_to_terminal=np.asarray([1, 0, -1, -1, 3, 2, 0, 1], dtype=np.int16),
        episode=np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int16),
    )


def test_episode_folds_have_no_leakage_and_cover_every_row_once() -> None:
    dataset = _dataset()
    folds = make_episode_folds(dataset.episode, folds=2, seed=7)

    validation_seen = np.zeros(len(dataset.episode), dtype=np.int16)
    for train_rows, validation_rows in folds:
        train_episodes = set(dataset.episode[train_rows].tolist())
        val_episodes = set(dataset.episode[validation_rows].tolist())
        assert not (train_episodes & val_episodes)
        validation_seen[validation_rows] += 1

    assert np.all(validation_seen == 1)


def test_sample_weights_emphasize_disagreement_and_critical_errors() -> None:
    dataset = _dataset()
    weights = sample_weights(
        dataset.expert_action,
        dataset.disagreement,
        dataset.critical_error,
    )

    assert np.isfinite(weights).all()
    assert weights[1] > weights[5]  # both expert LEFT; row 1 is critical disagreement


def test_oof_evaluation_measures_recovery_over_r0() -> None:
    dataset = _dataset()

    # Perfect diagnostic probe: predicts every expert correction.
    predicted = dataset.expert_action.copy()
    result = evaluate_oof(predicted, dataset)

    assert result["allStates"]["accuracy"] == 1.0
    assert result["disagreementStates"]["expertActionAccuracy"] == 1.0
    assert result["criticalErrors"]["correctionRate"] == 1.0
    assert result["coreMacroImprovementVsR0"] > 0
