from __future__ import annotations

import numpy as np

from fly_crossy.v2.dagger_collect import (
    DaggerRolloutDataset,
    summarize_dataset,
)


def _tiny_dataset() -> DaggerRolloutDataset:
    # Expert rows:
    # forward -> student forward
    # left    -> student forward (disagree, immediate vehicle death, expert survives)
    # right   -> student right
    n = 3
    return DaggerRolloutDataset(
        features=np.zeros((n, 21_022), dtype=np.float16),
        logits=np.asarray(
            [
                [5, 0, 0, 0, 0],
                [5, 0, 3, 0, 0],
                [0, 0, 0, 5, 0],
            ],
            dtype=np.float16,
        ),
        student_action=np.asarray([0, 0, 3], dtype=np.uint8),
        expert_action=np.asarray([0, 2, 3], dtype=np.uint8),
        disagreement=np.asarray([False, True, False]),
        critical_error=np.asarray([False, True, False]),
        student_terminal=np.asarray([0, 1, 0], dtype=np.uint8),
        expert_terminal=np.asarray([0, 0, 0], dtype=np.uint8),
        episode_terminal=np.asarray([1, 1, 0], dtype=np.uint8),
        steps_to_terminal=np.asarray([1, 0, -1], dtype=np.int16),
        episode=np.asarray([0, 0, 1], dtype=np.int16),
        step=np.asarray([0, 1, 0], dtype=np.int16),
        score=np.asarray([0, 1, 0], dtype=np.float32),
        fly_row=np.asarray([0, 1, 0], dtype=np.int16),
        fly_column=np.asarray([0, 0, 0], dtype=np.float32),
        current_lane=np.asarray([0, 0, 0], dtype=np.uint8),
        forward_lane=np.asarray([0, 1, 0], dtype=np.uint8),
        student_destination_lane=np.asarray([0, 1, 0], dtype=np.uint8),
        expert_destination_lane=np.asarray([0, 0, 0], dtype=np.uint8),
        max_probability=np.asarray([0.9, 0.8, 0.9], dtype=np.float32),
        probability_margin=np.asarray([0.8, 0.4, 0.8], dtype=np.float32),
        entropy=np.asarray([0.2, 0.5, 0.2], dtype=np.float32),
        seeds=("a", "b"),
    )


def test_dagger_summary_marks_immediately_fatal_student_error() -> None:
    dataset = _tiny_dataset()
    summary = summarize_dataset(
        dataset,
        episode_scores=[1.0, 0.0],
        episode_lengths=[2, 1],
        terminals={"vehicle": 1},
        baseline_validation=None,
    )

    assert summary["samples"] == 3
    assert summary["criticalErrors"] == 1
    assert summary["criticalErrorsByStudentTerminal"]["vehicle"] == 1
    assert summary["studentImmediateTerminalActions"] == 1
    assert summary["expertImmediateTerminalActions"] == 0
    assert summary["disagreementRate"] == 1 / 3


def test_dagger_predeath_windows_use_backfilled_distance() -> None:
    dataset = _tiny_dataset()
    summary = summarize_dataset(
        dataset,
        episode_scores=[1.0, 0.0],
        episode_lengths=[2, 1],
        terminals={"vehicle": 1},
        baseline_validation=None,
    )

    vehicle = summary["preDeathWindows"]["vehicle"]
    assert vehicle["1"]["samples"] == 1
    assert vehicle["1"]["criticalErrors"] == 1
    assert vehicle["3"]["samples"] == 2


def test_dagger_core_macro_is_balanced_over_supported_core_actions() -> None:
    dataset = _tiny_dataset()
    summary = summarize_dataset(
        dataset,
        episode_scores=[1.0, 0.0],
        episode_lengths=[2, 1],
        terminals={"vehicle": 1},
        baseline_validation=None,
    )

    # forward=1.0, left=0.0, right=1.0, wait unsupported -> mean 2/3
    assert summary["agreementByExpertAction"]["coreMacroAgreement"] == np.mean(
        [1.0, 0.0, 1.0]
    )
