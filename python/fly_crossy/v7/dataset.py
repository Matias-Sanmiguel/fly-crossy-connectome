from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Literal, Protocol

import numpy as np

from fly_crossy.env import GameState, create_game, observe, step_game
from fly_crossy.schema import ACTION_ORDER, flatten_observation
from fly_crossy.v2.crossy_camera import render_crossy_neural_frame
from fly_crossy.v2.preference_distill import (
    pairwise_targets,
    planner_action_preferences,
    primary_acceptable_mask,
)
from fly_crossy.v4.train_expo_specialist import resize_rgb

from .contracts import ACTION_NAMES, FRAME_SHAPE, OBSERVATION_SIZE
from .physical_gate import ActionGate, BiomechanicalActionGate


Partition = Literal["training", "validation", "final-test"]
Source = Literal["planner", "teacher", "student"]
_SOURCES = frozenset(("planner", "teacher", "student"))
_PARTITIONS = frozenset(("training", "validation", "final-test"))


class BehaviorPolicy(Protocol):
    def reset(self) -> None: ...

    def act(
        self,
        state: GameState,
        frame: np.ndarray,
        observation: np.ndarray,
    ) -> int: ...


@dataclass(slots=True)
class TransitionDataset:
    frames: np.ndarray
    observations: np.ndarray
    planner_values: np.ndarray
    acceptable: np.ndarray
    immediate_safe: np.ndarray
    pair_sign: np.ndarray
    pair_weight: np.ndarray
    best_action: np.ndarray
    behavior_action: np.ndarray
    episode_seed: np.ndarray
    step_index: np.ndarray
    source: np.ndarray
    action_order: tuple[str, ...]

    def validate(self) -> None:
        rows = len(self.frames)
        if rows == 0:
            raise ValueError("dataset must contain at least one row")
        expected = {
            "frames": ((rows, *FRAME_SHAPE), np.dtype(np.uint8)),
            "observations": ((rows, OBSERVATION_SIZE), np.dtype(np.float32)),
            "planner_values": ((rows, 5, 6), np.dtype(np.float32)),
            "acceptable": ((rows, 5), np.dtype(np.bool_)),
            "immediate_safe": ((rows, 5), np.dtype(np.bool_)),
            "pair_sign": ((rows, 10), np.dtype(np.int8)),
            "pair_weight": ((rows, 10), np.dtype(np.float32)),
            "best_action": ((rows,), np.dtype(np.int64)),
            "behavior_action": ((rows,), np.dtype(np.int64)),
            "step_index": ((rows,), np.dtype(np.int64)),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(
                    f"dataset {name} must have shape {shape} and dtype {dtype}"
                )
        for name in ("episode_seed", "source"):
            value = getattr(self, name)
            if value.shape != (rows,) or value.dtype.kind not in {"U", "S"}:
                raise ValueError(f"dataset {name} must contain one string per row")
        if self.action_order != ACTION_NAMES:
            raise ValueError("dataset action order must match canonical ACTION_ORDER")
        if not np.isfinite(self.observations).all():
            raise ValueError("dataset observations contain non-finite values")
        if not np.isfinite(self.planner_values).all():
            raise ValueError("dataset planner values contain non-finite values")
        if not np.isfinite(self.pair_weight).all():
            raise ValueError("dataset pair weights contain non-finite values")
        if not self.acceptable.any(axis=1).all():
            raise ValueError("every dataset row must have an acceptable action")
        for name in ("best_action", "behavior_action"):
            value = getattr(self, name)
            if ((value < 0) | (value >= len(ACTION_ORDER))).any():
                raise ValueError(f"dataset {name} contains an invalid action")
        if (self.step_index < 0).any():
            raise ValueError("dataset step indices must be non-negative")
        if not set(self.source.tolist()).issubset(_SOURCES):
            raise ValueError("dataset source contains an invalid value")

    def concatenate(self, other: "TransitionDataset") -> "TransitionDataset":
        self.validate()
        other.validate()
        if self.action_order != other.action_order:
            raise ValueError("cannot concatenate datasets with different action order")
        values: dict[str, object] = {}
        for field in fields(self):
            if field.name == "action_order":
                values[field.name] = self.action_order
            else:
                values[field.name] = np.concatenate(
                    (getattr(self, field.name), getattr(other, field.name)), axis=0
                )
        combined = TransitionDataset(**values)  # type: ignore[arg-type]
        combined.validate()
        return combined


@dataclass(frozen=True, slots=True)
class DatasetShard:
    path: Path
    sha256: str
    rows: int
    partition: Partition


def collect_labeled_episode(
    seed: str,
    *,
    max_steps: int,
    planner_depth: int,
    source: Source,
    behavior: BehaviorPolicy | None,
    action_gate: ActionGate | None = None,
) -> TransitionDataset:
    if source not in _SOURCES:
        raise ValueError(f"invalid collection source: {source}")
    if source != "planner" and behavior is None:
        raise ValueError("teacher/student collection requires a behavior policy")
    if behavior is not None:
        behavior.reset()
    gate = action_gate
    if source == "student":
        gate = gate or BiomechanicalActionGate()
        gate.reset_episode()

    state = create_game(seed)
    rows: dict[str, list[object]] = {
        name: []
        for name in (
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
        )
    }
    for step_index in range(max_steps):
        if state.terminal is not None:
            break
        normalized_frame = resize_rgb(render_crossy_neural_frame(state))
        frame = np.rint(normalized_frame * 255.0).clip(0, 255).astype(np.uint8)
        observation = flatten_observation(observe(state))
        values, best_action, immediate_safe = planner_action_preferences(
            state, depth=planner_depth
        )
        acceptable = primary_acceptable_mask(values)
        pair_sign, pair_weight = pairwise_targets(values)
        behavior_action = (
            int(best_action)
            if behavior is None
            else int(behavior.act(state, frame, observation))
        )

        rows["frames"].append(frame)
        rows["observations"].append(observation)
        rows["planner_values"].append(values)
        rows["acceptable"].append(acceptable)
        rows["immediate_safe"].append(immediate_safe)
        rows["pair_sign"].append(pair_sign)
        rows["pair_weight"].append(pair_weight)
        rows["best_action"].append(best_action)
        rows["behavior_action"].append(behavior_action)
        rows["episode_seed"].append(seed)
        rows["step_index"].append(step_index)
        rows["source"].append(source)
        effective_action = (
            gate.execute(behavior_action).effective_index
            if gate is not None
            else behavior_action
        )
        state = step_game(state, ACTION_ORDER[effective_action]).state

    dataset = TransitionDataset(
        frames=np.asarray(rows["frames"], dtype=np.uint8),
        observations=np.asarray(rows["observations"], dtype=np.float32),
        planner_values=np.asarray(rows["planner_values"], dtype=np.float32),
        acceptable=np.asarray(rows["acceptable"], dtype=np.bool_),
        immediate_safe=np.asarray(rows["immediate_safe"], dtype=np.bool_),
        pair_sign=np.asarray(rows["pair_sign"], dtype=np.int8),
        pair_weight=np.asarray(rows["pair_weight"], dtype=np.float32),
        best_action=np.asarray(rows["best_action"], dtype=np.int64),
        behavior_action=np.asarray(rows["behavior_action"], dtype=np.int64),
        episode_seed=np.asarray(rows["episode_seed"]),
        step_index=np.asarray(rows["step_index"], dtype=np.int64),
        source=np.asarray(rows["source"]),
        action_order=ACTION_NAMES,
    )
    dataset.validate()
    return dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".v7-meta-")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_dataset_shard(
    dataset: TransitionDataset,
    root: Path,
    *,
    partition: str,
) -> DatasetShard:
    dataset.validate()
    if partition not in _PARTITIONS:
        raise ValueError(f"invalid dataset partition: {partition}")
    partition_token = f"-{partition}-"
    if any(partition_token not in seed for seed in dataset.episode_seed.tolist()):
        raise ValueError("dataset episode seed does not match shard partition")
    root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=root, prefix=".v7-shard-")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                frames=dataset.frames,
                observations=dataset.observations,
                planner_values=dataset.planner_values,
                acceptable=dataset.acceptable,
                immediate_safe=dataset.immediate_safe,
                pair_sign=dataset.pair_sign,
                pair_weight=dataset.pair_weight,
                best_action=dataset.best_action,
                behavior_action=dataset.behavior_action,
                episode_seed=dataset.episode_seed,
                step_index=dataset.step_index,
                source=dataset.source,
            )
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256(temporary)
        path = root / f"{digest}.npz"
        if not path.exists():
            os.replace(temporary, path)
        metadata_path = path.with_suffix(".json")
        _atomic_json(
            metadata_path,
            {
                "actionOrder": dataset.action_order,
                "partition": partition,
                "rows": len(dataset.frames),
                "sha256": digest,
                "version": 1,
            },
        )
        shard = DatasetShard(
            path=path,
            sha256=digest,
            rows=len(dataset.frames),
            partition=partition,  # type: ignore[arg-type]
        )
        load_dataset_shard(shard)
        return shard
    finally:
        temporary.unlink(missing_ok=True)


def load_dataset_shard(shard: DatasetShard) -> TransitionDataset:
    if not shard.path.is_file() or _sha256(shard.path) != shard.sha256:
        raise ValueError("dataset shard hash mismatch")
    metadata_path = shard.path.with_suffix(".json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("dataset shard metadata is missing or corrupt") from error
    if (
        metadata.get("sha256") != shard.sha256
        or metadata.get("rows") != shard.rows
        or metadata.get("partition") != shard.partition
        or tuple(metadata.get("actionOrder", ())) != ACTION_NAMES
    ):
        raise ValueError("dataset shard metadata does not match its reference")
    try:
        with np.load(shard.path, allow_pickle=False) as archive:
            dataset = TransitionDataset(
                frames=archive["frames"],
                observations=archive["observations"],
                planner_values=archive["planner_values"],
                acceptable=archive["acceptable"],
                immediate_safe=archive["immediate_safe"],
                pair_sign=archive["pair_sign"],
                pair_weight=archive["pair_weight"],
                best_action=archive["best_action"],
                behavior_action=archive["behavior_action"],
                episode_seed=archive["episode_seed"],
                step_index=archive["step_index"],
                source=archive["source"],
                action_order=ACTION_NAMES,
            )
    except (OSError, ValueError, KeyError) as error:
        raise ValueError("dataset shard content is corrupt") from error
    dataset.validate()
    if len(dataset.frames) != shard.rows:
        raise ValueError("dataset shard row count mismatch")
    return dataset
