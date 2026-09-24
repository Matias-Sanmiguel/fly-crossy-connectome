from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from .checkpoint import validate_checkpoint
from .env import WORLD_VERSION, create_game, observe, step_game
from .expert_planner import PLANNER_VERSION, plan_action
from .models import SettledPopulationFixedGraphPolicy
from .schema import (
    ACTION_ORDER,
    OBSERVATION_INPUT_SIZE,
    OBSERVATION_VERSION,
    flatten_observation,
)


DAGGER_VERSION = "dagger-neural-rollout-v1"


def _load_student(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[SettledPopulationFixedGraphPolicy, Any]:
    saved = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = validate_checkpoint(
        saved,
        expected_controller="connectome",
        expected_environment_version=WORLD_VERSION,
        expected_observation_size=OBSERVATION_INPUT_SIZE,
    )
    if checkpoint.connectome_interface != "population-settled":
        raise ValueError("DAgger requires a population-settled student checkpoint.")
    if checkpoint.graph is None:
        raise ValueError("Student checkpoint graph is missing.")

    model = SettledPopulationFixedGraphPolicy(
        checkpoint.graph,
        checkpoint.observation_size,
        checkpoint.actions,
    ).to(device)
    model.load_state_dict(checkpoint.state_dict)
    model.eval()
    return model, checkpoint


def _load_base_dataset(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    dataset_path = directory / "dataset.npz"
    expected_hash = manifest["dataset"]["sha256"]
    actual_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("Base dataset hash does not match its manifest.")

    if manifest["environmentVersion"] != WORLD_VERSION:
        raise ValueError("Base dataset environment version mismatch.")
    if manifest["observationVersion"] != OBSERVATION_VERSION:
        raise ValueError("Base dataset observation version mismatch.")
    if manifest["observationSize"] != OBSERVATION_INPUT_SIZE:
        raise ValueError("Base dataset observation width mismatch.")
    if manifest["actionOrder"] != [action.value for action in ACTION_ORDER]:
        raise ValueError("Base dataset action order mismatch.")

    with np.load(dataset_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}

    required = {
        "observations",
        "actions",
        "episode_offsets",
        "seeds",
        "final_scores",
        "terminal_reasons",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"Base dataset is missing arrays: {sorted(missing)}")

    return manifest, arrays


@torch.no_grad()
def _collect_episode(
    *,
    model: SettledPopulationFixedGraphPolicy,
    graph_node_count: int,
    device: torch.device,
    seed: str,
    depth: int,
    max_steps: int,
) -> dict[str, Any]:
    state = create_game(seed)
    hidden = torch.zeros(
        1,
        graph_node_count,
        dtype=torch.float32,
        device=device,
    )
    action_index = {
        action: index
        for index, action in enumerate(ACTION_ORDER)
    }

    observations: list[np.ndarray] = []
    expert_actions: list[int] = []
    student_actions: list[int] = []
    nodes_expanded = 0
    cache_hits = 0

    while state.terminal is None and len(expert_actions) < max_steps:
        flat = flatten_observation(observe(state)).astype(
            np.float32,
            copy=False,
        )
        values = torch.from_numpy(flat).to(
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)

        logits, _, hidden = model(values, hidden)
        student_index = int(torch.argmax(logits, dim=1).item())
        student_action = ACTION_ORDER[student_index]

        decision = plan_action(state, depth=depth)
        expert_action = decision.action
        nodes_expanded += decision.nodes_expanded
        cache_hits += decision.cache_hits

        observations.append(flat)
        expert_actions.append(action_index[expert_action])
        student_actions.append(student_index)

        state = step_game(state, student_action).state

    return {
        "seed": seed,
        "observations": np.stack(observations, axis=0),
        "expertActions": np.asarray(expert_actions, dtype=np.int64),
        "studentActions": np.asarray(student_actions, dtype=np.int64),
        "score": float(state.score),
        "terminal": state.terminal,
        "nodesExpanded": nodes_expanded,
        "cacheHits": cache_hits,
    }


def collect_dagger_round(
    *,
    base_dataset: Path,
    checkpoint: Path,
    output: Path,
    episodes: int,
    max_steps: int,
    depth: int,
    seed_prefix: str,
    device_name: str,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    if depth < 1:
        raise ValueError("depth must be positive.")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else device_name
        if device_name != "auto"
        else "cpu"
    )

    base_manifest, base = _load_base_dataset(base_dataset)
    model, validated = _load_student(checkpoint, device)

    new_observations: list[np.ndarray] = []
    new_expert_actions: list[np.ndarray] = []
    new_student_actions: list[np.ndarray] = []
    new_lengths: list[int] = []
    new_original_seeds: list[str] = []
    new_scores: list[float] = []
    new_terminals: list[str] = []

    confusion = np.zeros(
        (len(ACTION_ORDER), len(ACTION_ORDER)),
        dtype=np.int64,
    )
    expert_counts: Counter[str] = Counter()
    student_counts: Counter[str] = Counter()
    disagreement_counts: Counter[str] = Counter()
    terminal_counts: Counter[str] = Counter()
    total_nodes = 0
    total_cache_hits = 0
    disagreement_total = 0

    for episode in range(episodes):
        seed = f"{seed_prefix}:{episode:04d}"
        rollout = _collect_episode(
            model=model,
            graph_node_count=validated.graph.node_count,
            device=device,
            seed=seed,
            depth=depth,
            max_steps=max_steps,
        )

        expert_actions = rollout["expertActions"]
        student_actions = rollout["studentActions"]
        disagreements = expert_actions != student_actions

        new_observations.append(rollout["observations"])
        new_expert_actions.append(expert_actions)
        new_student_actions.append(student_actions)
        new_lengths.append(len(expert_actions))
        new_original_seeds.append(seed)
        new_scores.append(float(rollout["score"]))
        new_terminals.append(rollout["terminal"] or "")

        total_nodes += int(rollout["nodesExpanded"])
        total_cache_hits += int(rollout["cacheHits"])
        disagreement_total += int(disagreements.sum())

        for expert_index, student_index in zip(
            expert_actions.tolist(),
            student_actions.tolist(),
            strict=True,
        ):
            confusion[expert_index, student_index] += 1
            expert_name = ACTION_ORDER[expert_index].value
            student_name = ACTION_ORDER[student_index].value
            expert_counts[expert_name] += 1
            student_counts[student_name] += 1
            if expert_index != student_index:
                disagreement_counts[expert_name] += 1

        terminal_counts[rollout["terminal"] or "step-limit"] += 1

        print(
            f"[dagger] episode {episode + 1:04d}/{episodes:04d} "
            f"steps={len(expert_actions):3d} "
            f"score={rollout['score']:6.1f} "
            f"disagree={float(disagreements.mean()):.3f} "
            f"terminal={rollout['terminal'] or 'step-limit'}",
            flush=True,
        )

    dagger_observations = np.concatenate(new_observations, axis=0).astype(
        np.float32,
        copy=False,
    )
    dagger_expert_actions = np.concatenate(new_expert_actions, axis=0).astype(
        np.int64,
        copy=False,
    )
    dagger_student_actions = np.concatenate(new_student_actions, axis=0).astype(
        np.int64,
        copy=False,
    )

    # Standard training arrays: old expert trajectories + new student-visited
    # trajectories, both labelled by the offline planner.
    observations = np.concatenate(
        [base["observations"].astype(np.float32, copy=False), dagger_observations],
        axis=0,
    )
    actions = np.concatenate(
        [base["actions"].astype(np.int64, copy=False), dagger_expert_actions],
        axis=0,
    )

    base_offsets = base["episode_offsets"].astype(np.int64, copy=False)
    offsets = list(base_offsets.tolist())
    current = int(offsets[-1])
    for length in new_lengths:
        current += int(length)
        offsets.append(current)
    episode_offsets = np.asarray(offsets, dtype=np.int64)

    # Checkpoint validation expects every concrete world seed to share one root
    # prefix. Keep original seed provenance in manifest, and use canonical
    # aggregate aliases in the training file.
    aggregate_root = f"{seed_prefix}-aggregate"
    base_episode_count = len(base_offsets) - 1
    canonical_seeds = [
        f"{aggregate_root}:base:{index:04d}"
        for index in range(base_episode_count)
    ] + [
        f"{aggregate_root}:dagger:{index:04d}"
        for index in range(episodes)
    ]

    final_scores = np.concatenate(
        [
            base["final_scores"].astype(np.float32, copy=False),
            np.asarray(new_scores, dtype=np.float32),
        ]
    )
    terminal_reasons = np.concatenate(
        [
            base["terminal_reasons"].astype(str),
            np.asarray(new_terminals),
        ]
    )

    # Optional audit arrays are aligned with all samples. Old expert samples
    # have studentAction=-1 and source=0; DAgger samples use source=1.
    base_sample_count = len(base["actions"])
    student_actions = np.concatenate(
        [
            np.full(base_sample_count, -1, dtype=np.int64),
            dagger_student_actions,
        ]
    )
    source = np.concatenate(
        [
            np.zeros(base_sample_count, dtype=np.uint8),
            np.ones(len(dagger_student_actions), dtype=np.uint8),
        ]
    )
    disagreement = np.concatenate(
        [
            np.zeros(base_sample_count, dtype=np.uint8),
            (dagger_student_actions != dagger_expert_actions).astype(np.uint8),
        ]
    )

    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "dataset.npz"
    np.savez(
        dataset_path,
        observations=observations,
        actions=actions,
        episode_offsets=episode_offsets,
        seeds=np.asarray(canonical_seeds),
        final_scores=final_scores,
        terminal_reasons=terminal_reasons,
        student_actions=student_actions,
        source=source,
        disagreement=disagreement,
    )
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    confusion_payload = {
        ACTION_ORDER[expert].value: {
            ACTION_ORDER[student].value: int(confusion[expert, student])
            for student in range(len(ACTION_ORDER))
        }
        for expert in range(len(ACTION_ORDER))
    }

    dagger_samples = len(dagger_expert_actions)
    base_action_counts = Counter(
        ACTION_ORDER[int(index)].value
        for index in base["actions"].tolist()
    )
    aggregate_action_counts = base_action_counts + expert_counts

    manifest = {
        "version": 2,
        "environmentVersion": WORLD_VERSION,
        "observationVersion": OBSERVATION_VERSION,
        "observationSize": OBSERVATION_INPUT_SIZE,
        "plannerVersion": PLANNER_VERSION,
        "plannerDepth": depth,
        "seedPrefix": aggregate_root,
        "episodes": int(len(episode_offsets) - 1),
        "maxStepsPerEpisode": max(
            int(base_manifest.get("maxStepsPerEpisode", 0)),
            max_steps,
        ),
        "samples": int(len(actions)),
        "actionOrder": [action.value for action in ACTION_ORDER],
        "actionCounts": dict(sorted(aggregate_action_counts.items())),
        "dataset": {
            "path": dataset_path.name,
            "sha256": dataset_hash,
        },
        "aggregation": {
            "version": DAGGER_VERSION,
            "baseDataset": str(base_dataset),
            "baseDatasetSha256": base_manifest["dataset"]["sha256"],
            "baseSamples": int(base_sample_count),
            "baseEpisodes": int(base_episode_count),
            "studentCheckpoint": str(checkpoint),
            "studentCheckpointSha256": checkpoint_hash,
            "studentInternalSteps": (
                SettledPopulationFixedGraphPolicy.INTERNAL_STEPS
            ),
            "daggerSeedPrefix": seed_prefix,
            "daggerEpisodes": episodes,
            "daggerSamples": int(dagger_samples),
            "daggerMeanScore": mean(new_scores),
            "daggerBestScore": max(new_scores),
            "daggerMeanLength": mean(new_lengths),
            "daggerTerminalReasons": dict(sorted(terminal_counts.items())),
            "expertActionCounts": dict(sorted(expert_counts.items())),
            "studentActionCounts": dict(sorted(student_counts.items())),
            "expertActionDisagreementCounts": dict(
                sorted(disagreement_counts.items())
            ),
            "disagreementRate": (
                disagreement_total / max(1, dagger_samples)
            ),
            "confusionExpertRowsStudentColumns": confusion_payload,
            "meanSearchNodesPerDecision": (
                total_nodes / max(1, dagger_samples)
            ),
            "meanCacheHitsPerDecision": (
                total_cache_hits / max(1, dagger_samples)
            ),
            "originalBaseSeedPrefix": base_manifest.get("seedPrefix"),
            "originalDaggerSeeds": new_original_seeds,
        },
    }

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect DAgger trajectories: the settled MaleCNS drives the game, "
            "the offline planner labels each visited state, and the labels are "
            "aggregated with the original expert dataset."
        )
    )
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed-prefix", required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    result = collect_dagger_round(
        base_dataset=args.base_dataset,
        checkpoint=args.checkpoint,
        output=args.output,
        episodes=args.episodes,
        max_steps=args.max_steps,
        depth=args.depth,
        seed_prefix=args.seed_prefix,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
