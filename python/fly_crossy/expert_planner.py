from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from statistics import mean
from typing import Sequence

from .env import GameState, create_game, step_game
from .schema import Action


PLANNER_VERSION = "receding-horizon-v2"
PLANNER_ACTION_ORDER: tuple[Action, ...] = (
    Action.FORWARD,
    Action.LEFT,
    Action.RIGHT,
    Action.WAIT,
    Action.BACKWARD,
)


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    action: Action
    principal_variation: tuple[Action, ...]
    value: tuple[float, ...]
    nodes_expanded: int
    cache_hits: int


def _safe_action_count(state: GameState) -> int:
    if state.terminal is not None:
        return 0
    return sum(
        step_game(state, action).state.terminal is None
        for action in PLANNER_ACTION_ORDER
    )


def _leaf_value(
    state: GameState,
    *,
    root_score: float,
    root_row: int,
) -> tuple[float, ...]:
    score_gain = state.score - root_score
    row_gain = state.fly.row - root_row

    if state.terminal is not None:
        return (
            0.0,
            score_gain,
            row_gain,
            -abs(state.fly.column),
            0.0,
            0.0,
        )

    mobility = _safe_action_count(state)
    forward_safe = float(
        step_game(state, Action.FORWARD).state.terminal is None
    )
    return (
        1.0,
        score_gain,
        row_gain,
        float(mobility),
        forward_safe,
        -abs(state.fly.column),
    )


def _search_key(
    state: GameState,
    remaining: int,
) -> tuple[int, int, int, float, float, str | None]:
    return (
        remaining,
        state.step,
        state.fly.row,
        round(float(state.fly.column), 9),
        round(float(state.score), 9),
        state.terminal,
    )


def plan_action(
    state: GameState,
    *,
    depth: int = 4,
) -> PlannerDecision:
    if depth < 1:
        raise ValueError("Planner depth must be at least one.")
    if state.terminal is not None:
        raise ValueError("Cannot plan from a terminal state.")

    root_score = state.score
    root_row = state.fly.row
    nodes_expanded = 0
    cache_hits = 0
    cache: dict[
        tuple[int, int, int, float, float, str | None],
        tuple[tuple[float, ...], tuple[Action, ...]],
    ] = {}

    def search(
        candidate: GameState,
        remaining: int,
    ) -> tuple[tuple[float, ...], tuple[Action, ...]]:
        nonlocal nodes_expanded, cache_hits

        key = _search_key(candidate, remaining)
        cached = cache.get(key)
        if cached is not None:
            cache_hits += 1
            return cached

        nodes_expanded += 1
        if candidate.terminal is not None or remaining == 0:
            result = (
                _leaf_value(
                    candidate,
                    root_score=root_score,
                    root_row=root_row,
                ),
                (),
            )
            cache[key] = result
            return result

        best_value: tuple[float, ...] | None = None
        best_path: tuple[Action, ...] = ()

        for action in PLANNER_ACTION_ORDER:
            next_state = step_game(candidate, action).state
            value, suffix = search(next_state, remaining - 1)
            path = (action, *suffix)
            if best_value is None or value > best_value:
                best_value = value
                best_path = path

        assert best_value is not None
        result = (best_value, best_path)
        cache[key] = result
        return result

    value, path = search(state, depth)
    if not path:
        raise RuntimeError("Planner produced no action.")

    return PlannerDecision(
        action=path[0],
        principal_variation=path,
        value=value,
        nodes_expanded=nodes_expanded,
        cache_hits=cache_hits,
    )


def evaluate_expert(
    *,
    episodes: int,
    depth: int,
    max_steps: int,
    seed_prefix: str,
) -> dict[str, object]:
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")

    scores: list[float] = []
    lengths: list[int] = []
    terminals: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    reached_step_limit = 0
    total_nodes = 0
    total_cache_hits = 0

    for episode in range(episodes):
        state = create_game(f"{seed_prefix}:{episode:04d}")
        steps = 0

        while state.terminal is None and steps < max_steps:
            decision = plan_action(state, depth=depth)
            total_nodes += decision.nodes_expanded
            total_cache_hits += decision.cache_hits
            actions[decision.action.value] += 1
            state = step_game(state, decision.action).state
            steps += 1

        scores.append(float(state.score))
        lengths.append(steps)
        if state.terminal is None:
            reached_step_limit += 1
        else:
            terminals[state.terminal] += 1

    ordered_scores = sorted(scores)
    decisions = max(1, sum(lengths))
    return {
        "plannerVersion": PLANNER_VERSION,
        "episodes": episodes,
        "depth": depth,
        "maxSteps": max_steps,
        "meanScore": mean(scores),
        "medianScore": ordered_scores[len(ordered_scores) // 2],
        "bestScore": max(scores),
        "meanLength": mean(lengths),
        "reachedStepLimit": reached_step_limit,
        "terminalReasons": dict(sorted(terminals.items())),
        "actions": dict(sorted(actions.items())),
        "meanSearchNodesPerDecision": total_nodes / decisions,
        "meanCacheHitsPerDecision": total_cache_hits / decisions,
    }


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the training-only receding-horizon expert. "
            "It is never exported to the browser."
        )
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed-prefix", default="expert-planner-v2")
    args = parser.parse_args(argv)

    print(
        json.dumps(
            evaluate_expert(
                episodes=args.episodes,
                depth=args.depth,
                max_steps=args.max_steps,
                seed_prefix=args.seed_prefix,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
