from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fly_crossy.v7.contracts import ACTION_NAMES
from fly_crossy.v7.dataset import (
    DatasetShard,
    TransitionDataset,
    collect_labeled_episode,
    load_dataset_shard,
    write_dataset_shard,
)
from fly_crossy.v7.physical_gate import GatedAction


def _dataset(rows: int = 2, *, seed: str = "unit-v7-smoke-training-000") -> TransitionDataset:
    values = np.zeros((rows, 5, 6), dtype=np.float32)
    values[:, 0, 0] = 1.0
    acceptable = np.zeros((rows, 5), dtype=np.bool_)
    acceptable[:, 0] = True
    safe = np.ones((rows, 5), dtype=np.bool_)
    return TransitionDataset(
        frames=np.zeros((rows, 24, 48, 3), dtype=np.uint8),
        observations=np.zeros((rows, 517), dtype=np.float32),
        planner_values=values,
        acceptable=acceptable,
        immediate_safe=safe,
        pair_sign=np.zeros((rows, 10), dtype=np.int8),
        pair_weight=np.zeros((rows, 10), dtype=np.float32),
        best_action=np.zeros(rows, dtype=np.int64),
        behavior_action=np.zeros(rows, dtype=np.int64),
        episode_seed=np.asarray([seed] * rows),
        step_index=np.arange(rows, dtype=np.int64),
        source=np.asarray(["planner"] * rows),
        action_order=ACTION_NAMES,
    )


def test_dataset_accepts_exact_v7_shapes_and_dtypes() -> None:
    dataset = _dataset()

    dataset.validate()

    assert dataset.frames.dtype == np.uint8
    assert dataset.frames.shape == (2, 24, 48, 3)
    assert dataset.observations.dtype == np.float32
    assert dataset.planner_values.shape == (2, 5, 6)
    assert dataset.acceptable.dtype == np.bool_
    assert dataset.immediate_safe.dtype == np.bool_
    assert dataset.pair_sign.dtype == np.int8
    assert dataset.pair_weight.dtype == np.float32
    assert dataset.best_action.dtype == np.int64
    assert dataset.behavior_action.dtype == np.int64


@pytest.mark.parametrize(
    "mutation",
    (
        "empty",
        "nan-observation",
        "inf-values",
        "bad-action",
        "bad-behavior",
        "short-source",
        "wrong-frame-dtype",
        "wrong-action-order",
        "no-acceptable-action",
    ),
)
def test_dataset_rejects_malformed_or_nonfinite_rows(mutation: str) -> None:
    dataset = _dataset()
    if mutation == "empty":
        dataset = _dataset(0)
    elif mutation == "nan-observation":
        dataset.observations[0, 0] = np.nan
    elif mutation == "inf-values":
        dataset.planner_values[0, 0, 0] = np.inf
    elif mutation == "bad-action":
        dataset.best_action[0] = 5
    elif mutation == "bad-behavior":
        dataset.behavior_action[0] = -1
    elif mutation == "short-source":
        dataset.source = dataset.source[:1]
    elif mutation == "wrong-frame-dtype":
        dataset.frames = dataset.frames.astype(np.float32)
    elif mutation == "wrong-action-order":
        dataset.action_order = tuple(reversed(ACTION_NAMES))
    elif mutation == "no-acceptable-action":
        dataset.acceptable[0] = False

    with pytest.raises(ValueError):
        dataset.validate()


def test_collection_is_deterministic_and_records_episode_identity() -> None:
    kwargs = {
        "max_steps": 3,
        "planner_depth": 1,
        "source": "planner",
        "behavior": None,
    }
    first = collect_labeled_episode("unit-v7-smoke-training-007", **kwargs)
    second = collect_labeled_episode("unit-v7-smoke-training-007", **kwargs)

    for field in (
        "frames",
        "observations",
        "planner_values",
        "acceptable",
        "immediate_safe",
        "pair_sign",
        "pair_weight",
        "best_action",
        "behavior_action",
        "episode_seed",
        "step_index",
        "source",
    ):
        assert np.array_equal(getattr(first, field), getattr(second, field))
    assert first.frames.dtype == np.uint8
    assert first.observations.dtype == np.float32
    assert first.episode_seed.tolist() == ["unit-v7-smoke-training-007"] * len(first.frames)
    assert first.step_index.tolist() == list(range(len(first.frames)))
    assert first.source.tolist() == ["planner"] * len(first.frames)


def test_student_closed_loop_collection_uses_physical_effective_actions() -> None:
    class ForwardPolicy:
        def reset(self) -> None:
            pass

        def act(self, state, frame, observation) -> int:
            return 0

    class FailedGate:
        def __init__(self) -> None:
            self.requested: list[int] = []

        def reset_episode(self) -> None:
            pass

        def execute(self, requested_index: int) -> GatedAction:
            self.requested.append(requested_index)
            return GatedAction(requested_index, 4, "failed", "contact-disabled")

    gate = FailedGate()
    dataset = collect_labeled_episode(
        "unit-v7-smoke-training-physical-gate",
        max_steps=3,
        planner_depth=1,
        source="student",
        behavior=ForwardPolicy(),
        action_gate=gate,
    )

    assert dataset.behavior_action.tolist() == [0, 0, 0]
    assert gate.requested == [0, 0, 0]
    assert dataset.observations[1, 485:490].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_concatenate_preserves_order_and_revalidates() -> None:
    first = _dataset(seed="unit-v7-smoke-training-000")
    second = _dataset(seed="unit-v7-smoke-training-001")

    combined = first.concatenate(second)

    assert combined.episode_seed.tolist() == [
        "unit-v7-smoke-training-000",
        "unit-v7-smoke-training-000",
        "unit-v7-smoke-training-001",
        "unit-v7-smoke-training-001",
    ]


def test_content_addressed_shard_round_trips_and_changes_with_content(tmp_path: Path) -> None:
    dataset = _dataset()
    first = write_dataset_shard(dataset, tmp_path, partition="training")
    loaded = load_dataset_shard(first)

    assert first.path.name == f"{first.sha256}.npz"
    assert first.rows == 2
    assert first.partition == "training"
    assert np.array_equal(loaded.frames, dataset.frames)
    assert loaded.action_order == ACTION_NAMES

    changed = _dataset()
    changed.frames[0, 0, 0, 0] = 1
    second = write_dataset_shard(changed, tmp_path, partition="training")
    assert second.sha256 != first.sha256


def test_writer_rejects_seed_from_another_partition(tmp_path: Path) -> None:
    dataset = _dataset(seed="unit-v7-smoke-validation-000")

    with pytest.raises(ValueError, match="partition"):
        write_dataset_shard(dataset, tmp_path, partition="training")


def test_loader_rejects_hash_mismatch_or_corrupt_shard(tmp_path: Path) -> None:
    shard = write_dataset_shard(_dataset(), tmp_path, partition="training")
    wrong_hash = replace(shard, sha256="0" * 64)
    with pytest.raises(ValueError, match="hash"):
        load_dataset_shard(wrong_hash)

    content = bytearray(shard.path.read_bytes())
    content[-1] ^= 0xFF
    shard.path.write_bytes(content)
    with pytest.raises(ValueError, match="hash"):
        load_dataset_shard(shard)


def test_loader_rejects_missing_or_corrupt_metadata(tmp_path: Path) -> None:
    shard = write_dataset_shard(_dataset(), tmp_path, partition="training")
    metadata = shard.path.with_suffix(".json")
    metadata.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="metadata"):
        load_dataset_shard(shard)
