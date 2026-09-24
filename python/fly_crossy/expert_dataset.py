from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .env import WORLD_VERSION, create_game, observe, step_game
from .expert_planner import PLANNER_VERSION, plan_action
from .schema import (
    ACTION_ORDER,
    OBSERVATION_INPUT_SIZE,
    OBSERVATION_VERSION,
    flatten_observation,
)


DATASET_VERSION = 1


def _episode_seed(seed_prefix: str, index: int) -> str:
    return f"{seed_prefix}:{index:04d}"


def _generate_episode(
    specification: tuple[int, str, int, int],
) -> dict[str, Any]:
    index, seed_prefix, depth, max_steps = specification
    seed = _episode_seed(seed_prefix, index)
    state = create_game(seed)

    observations: list[np.ndarray] = []
    actions: list[int] = []
    nodes_expanded = 0
    cache_hits = 0
    action_index = {
        action: index
        for index, action in enumerate(ACTION_ORDER)
    }

    while state.terminal is None and len(actions) < max_steps:
        observations.append(
            flatten_observation(observe(state)).astype(np.float32, copy=False)
        )
        decision = plan_action(state, depth=depth)
        nodes_expanded += decision.nodes_expanded
        cache_hits += decision.cache_hits
        actions.append(action_index[decision.action])
        state = step_game(state, decision.action).state

    return {
        "seed": seed,
        "observations": np.stack(observations, axis=0),
        "actions": np.asarray(actions, dtype=np.int64),
        "score": float(state.score),
        "terminal": state.terminal,
        "nodesExpanded": nodes_expanded,
        "cacheHits": cache_hits,
    }


def _iter_episodes(
    specifications: list[tuple[int, str, int, int]],
    workers: int,
) -> Iterable[dict[str, Any]]:
    if workers <= 1:
        for specification in specifications:
            yield _generate_episode(specification)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(
            _generate_episode,
            specifications,
            chunksize=1,
        )


def generate_dataset(
    *,
    output: Path,
    episodes: int,
    depth: int,
    max_steps: int,
    seed_prefix: str,
    workers: int,
) -> dict[str, Any]:
    if episodes < 2:
        raise ValueError("Expert dataset needs at least two episodes.")
    if depth < 1:
        raise ValueError("Planner depth must be at least one.")
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    if workers < 1:
        raise ValueError("workers must be positive.")

    output.mkdir(parents=True, exist_ok=True)
    specifications = [
        (index, seed_prefix, depth, max_steps)
        for index in range(episodes)
    ]

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    offsets = [0]
    seeds: list[str] = []
    scores: list[float] = []
    terminals: list[str] = []
    total_nodes = 0
    total_cache_hits = 0
    action_counts: Counter[str] = Counter()

    for index, episode in enumerate(
        _iter_episodes(specifications, workers),
        start=1,
    ):
        episode_observations = episode["observations"]
        episode_actions = episode["actions"]
        observations.append(episode_observations)
        actions.append(episode_actions)
        offsets.append(offsets[-1] + len(episode_actions))
        seeds.append(episode["seed"])
        scores.append(episode["score"])
        terminals.append(episode["terminal"] or "")
        total_nodes += int(episode["nodesExpanded"])
        total_cache_hits += int(episode["cacheHits"])
        for action_index in episode_actions:
            action_counts[ACTION_ORDER[int(action_index)].value] += 1

        print(
            f"[dataset] episode {index:04d}/{episodes:04d} "
            f"steps={len(episode_actions):3d} "
            f"score={episode['score']:6.1f} "
            f"terminal={episode['terminal'] or 'step-limit'}",
            flush=True,
        )

    all_observations = np.concatenate(observations, axis=0).astype(
        np.float32,
        copy=False,
    )
    all_actions = np.concatenate(actions, axis=0).astype(
        np.int64,
        copy=False,
    )
    episode_offsets = np.asarray(offsets, dtype=np.int64)

    dataset_path = output / "dataset.npz"
    np.savez(
        dataset_path,
        observations=all_observations,
        actions=all_actions,
        episode_offsets=episode_offsets,
        seeds=np.asarray(seeds),
        final_scores=np.asarray(scores, dtype=np.float32),
        terminal_reasons=np.asarray(terminals),
    )
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    manifest = {
        "version": DATASET_VERSION,
        "environmentVersion": WORLD_VERSION,
        "observationVersion": OBSERVATION_VERSION,
        "observationSize": OBSERVATION_INPUT_SIZE,
        "plannerVersion": PLANNER_VERSION,
        "plannerDepth": depth,
        "seedPrefix": seed_prefix,
        "episodes": episodes,
        "maxStepsPerEpisode": max_steps,
        "samples": int(len(all_actions)),
        "actionOrder": [action.value for action in ACTION_ORDER],
        "actionCounts": dict(sorted(action_counts.items())),
        "meanScore": float(np.mean(scores)),
        "bestScore": float(np.max(scores)),
        "terminalReasons": dict(
            sorted(Counter(reason or "step-limit" for reason in terminals).items())
        ),
        "meanSearchNodesPerDecision": total_nodes / max(1, len(all_actions)),
        "meanCacheHitsPerDecision": total_cache_hits / max(1, len(all_actions)),
        "dataset": {
            "path": dataset_path.name,
            "sha256": dataset_hash,
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
            "Generate complete expert trajectories for recurrent MaleCNS "
            "imitation learning."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed-prefix", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    manifest = generate_dataset(
        output=args.output,
        episodes=args.episodes,
        depth=args.depth,
        max_steps=args.max_steps,
        seed_prefix=args.seed_prefix,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    _main()
