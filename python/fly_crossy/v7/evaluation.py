from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np

from fly_crossy.env import create_game, observe, step_game
from fly_crossy.schema import ACTION_ORDER, flatten_observation
from fly_crossy.v2.crossy_camera import render_crossy_neural_frame
from fly_crossy.v2.preference_distill import planner_action_preferences
from fly_crossy.v4.train_expo_specialist import resize_rgb

from .contracts import ProfileName, profile_budget
from .dataset import BehaviorPolicy
from .physical_gate import ActionGate


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    seed: str
    steps: int
    score: float
    success: bool
    progress_qualified: bool
    terminal_reason: str | None
    fatal_actions: int
    labelled_states: int
    actions: tuple[int, ...]
    requested_actions: tuple[int, ...] = ()
    effective_actions: tuple[int, ...] = ()
    physical_outcomes: tuple[str, ...] = ()
    physical_failures: tuple[str, ...] = ()
    physical_confirmed: int = 0
    physical_failed: int = 0
    physical_waited: int = 0


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    episodes: int
    successes: int
    progress_qualified: int
    mean_score: float
    median_score: float
    mean_survival_steps: float
    fatal_action_rate: float
    action_distribution: tuple[float, ...]
    physical_confirmed: int = 0
    physical_failed: int = 0
    physical_waited: int = 0


@dataclass(frozen=True, slots=True)
class PairedComparison:
    seeds: tuple[str, ...]
    wins: int
    ties: int
    losses: int
    win_fraction: float


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    next_action: Literal["promote", "continue", "reject"]
    predicates: Mapping[str, bool]
    reasons: tuple[str, ...]


def _validate_episodes(episodes: Sequence[EpisodeMetrics]) -> None:
    if not episodes:
        raise ValueError("episode collection must not be empty")
    seeds = [episode.seed for episode in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError("episode collection contains duplicate seeds")
    for episode in episodes:
        if episode.steps < 0 or episode.labelled_states < 0 or episode.fatal_actions < 0:
            raise ValueError("episode counters must be non-negative")
        if episode.fatal_actions > episode.labelled_states:
            raise ValueError("fatal action count exceeds labelled states")
        if not np.isfinite(episode.score):
            raise ValueError("episode score must be finite")
        if any(action < 0 or action >= len(ACTION_ORDER) for action in episode.actions):
            raise ValueError("episode contains an invalid action")
        requested = episode.requested_actions or episode.actions
        effective = episode.effective_actions or episode.actions
        if len(requested) != len(episode.actions) or len(effective) != len(episode.actions):
            raise ValueError("episode physical action histories must match actions")
        if episode.physical_outcomes and len(episode.physical_outcomes) != len(episode.actions):
            raise ValueError("episode physical outcomes must match actions")
        if any(action < 0 or action >= len(ACTION_ORDER) for action in (*requested, *effective)):
            raise ValueError("episode contains an invalid physical action")


def aggregate_metrics(episodes: Sequence[EpisodeMetrics]) -> AggregateMetrics:
    _validate_episodes(episodes)
    scores = np.asarray([episode.score for episode in episodes], dtype=np.float64)
    actions = np.asarray(
        [action for episode in episodes for action in episode.actions], dtype=np.int64
    )
    counts = np.bincount(actions, minlength=len(ACTION_ORDER)) if len(actions) else np.zeros(5)
    total_actions = int(counts.sum())
    distribution = tuple(
        float(count / total_actions) if total_actions else 0.0 for count in counts
    )
    labelled = sum(episode.labelled_states for episode in episodes)
    fatal = sum(episode.fatal_actions for episode in episodes)
    return AggregateMetrics(
        episodes=len(episodes),
        successes=sum(episode.success for episode in episodes),
        progress_qualified=sum(episode.progress_qualified for episode in episodes),
        mean_score=float(scores.mean()),
        median_score=float(np.median(scores)),
        mean_survival_steps=float(np.mean([episode.steps for episode in episodes])),
        fatal_action_rate=float(fatal / labelled) if labelled else 0.0,
        action_distribution=distribution,
        physical_confirmed=sum(episode.physical_confirmed for episode in episodes),
        physical_failed=sum(episode.physical_failed for episode in episodes),
        physical_waited=sum(episode.physical_waited for episode in episodes),
    )


def robust_key(metrics: AggregateMetrics) -> tuple[float, ...]:
    return (
        float(metrics.successes),
        float(metrics.progress_qualified),
        metrics.mean_score,
        metrics.median_score,
        metrics.mean_survival_steps,
        -metrics.fatal_action_rate,
    )


def _episode_key(episode: EpisodeMetrics) -> tuple[float, ...]:
    fatal_rate = episode.fatal_actions / max(1, episode.labelled_states)
    return (
        float(episode.success),
        float(episode.progress_qualified),
        episode.score,
        float(episode.steps),
        -fatal_rate,
    )


def paired_comparison(
    predecessor: Sequence[EpisodeMetrics],
    candidate: Sequence[EpisodeMetrics],
) -> PairedComparison:
    _validate_episodes(predecessor)
    _validate_episodes(candidate)
    old = {episode.seed: episode for episode in predecessor}
    new = {episode.seed: episode for episode in candidate}
    if set(old) != set(new):
        raise ValueError("paired episode seed sets do not match")
    seeds = tuple(sorted(old))
    wins = ties = losses = 0
    for seed in seeds:
        old_key = _episode_key(old[seed])
        new_key = _episode_key(new[seed])
        if new_key > old_key:
            wins += 1
        elif new_key < old_key:
            losses += 1
        else:
            ties += 1
    return PairedComparison(seeds, wins, ties, losses, wins / len(seeds))


def _action_distribution_healthy(
    predecessor: AggregateMetrics,
    candidate: AggregateMetrics,
) -> bool:
    if max(candidate.action_distribution, default=0.0) > 0.85:
        return False
    for old_share, new_share in zip(
        predecessor.action_distribution, candidate.action_distribution, strict=True
    ):
        if old_share >= 0.10 and new_share < 0.02:
            return False
    return True


def decide_promotion(
    *,
    profile: ProfileName,
    predecessor: Sequence[EpisodeMetrics],
    candidate: Sequence[EpisodeMetrics],
    planner_agreement: float,
    agreement_threshold: float,
    final_test_passed: bool,
) -> PromotionDecision:
    profile_budget(profile)
    if not np.isfinite(planner_agreement) or not 0.0 <= planner_agreement <= 1.0:
        raise ValueError("planner agreement must be finite and in [0, 1]")
    if not 0.0 <= agreement_threshold <= 1.0:
        raise ValueError("agreement threshold must be in [0, 1]")
    old = aggregate_metrics(predecessor)
    new = aggregate_metrics(candidate)
    paired = paired_comparison(predecessor, candidate)
    required_mean = old.mean_score + 0.10 * max(abs(old.mean_score), 1e-6)
    predicates = {
        "profileCanPromote": profile != "smoke",
        "sufficientPairedSeeds": len(paired.seeds) >= 3,
        "meanScoreAtLeast10PercentBetter": new.mean_score >= required_mean,
        "medianScoreNoRegression": new.median_score >= old.median_score,
        "robustTupleImproved": robust_key(new) > robust_key(old),
        "pairedWinsAtLeast60Percent": paired.win_fraction >= 0.60,
        "successCountNoRegression": new.successes >= old.successes,
        "plannerAgreementPassed": planner_agreement >= agreement_threshold,
        "actionDistributionHealthy": _action_distribution_healthy(old, new),
        "finalTestPassed": bool(final_test_passed),
    }
    reasons = tuple(name for name, passed in predicates.items() if not passed)
    if all(predicates.values()):
        next_action: Literal["promote", "continue", "reject"] = "promote"
    elif (
        not predicates["finalTestPassed"]
        or not predicates["successCountNoRegression"]
        or not predicates["robustTupleImproved"]
    ):
        next_action = "reject"
    else:
        next_action = "continue"
    return PromotionDecision(next_action, predicates, reasons)


def evaluate_policy(
    policy: BehaviorPolicy,
    seeds: Sequence[str],
    *,
    max_steps: int,
    planner_depth: int,
    action_gate: ActionGate | None = None,
) -> tuple[EpisodeMetrics, ...]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be non-empty and unique")
    if max_steps <= 0 or planner_depth <= 0:
        raise ValueError("evaluation limits must be positive")
    gate = action_gate
    results: list[EpisodeMetrics] = []
    for seed in seeds:
        policy.reset()
        if gate is not None:
            gate.reset_episode()
        state = create_game(seed)
        actions: list[int] = []
        effective_actions: list[int] = []
        physical_outcomes: list[str] = []
        physical_failures: list[str] = []
        fatal_actions = 0
        steps = 0
        while state.terminal is None and steps < max_steps:
            frame = np.rint(
                resize_rgb(render_crossy_neural_frame(state)) * 255.0
            ).clip(0, 255).astype(np.uint8)
            observation = flatten_observation(observe(state))
            _, _, immediate_safe = planner_action_preferences(
                state, depth=planner_depth
            )
            action = int(policy.act(state, frame, observation))
            if action < 0 or action >= len(ACTION_ORDER):
                raise ValueError("policy returned an invalid action")
            fatal_actions += int(not immediate_safe[action])
            actions.append(action)
            if gate is None:
                effective_action = action
            else:
                gated = gate.execute(action)
                effective_action = gated.effective_index
                physical_outcomes.append(gated.outcome)
                if gated.outcome == "failed":
                    physical_failures.append(
                        gated.failure_reason or "physical-action-failed"
                    )
            effective_actions.append(effective_action)
            state = step_game(state, ACTION_ORDER[effective_action]).state
            steps += 1
        progress_qualified = float(state.score) > 0.0
        survived = state.terminal is None and steps == max_steps
        results.append(
            EpisodeMetrics(
                seed=seed,
                steps=steps,
                score=float(state.score),
                success=bool(survived and progress_qualified),
                progress_qualified=progress_qualified,
                terminal_reason=state.terminal,
                fatal_actions=fatal_actions,
                labelled_states=steps,
                actions=tuple(actions),
                requested_actions=tuple(actions),
                effective_actions=tuple(effective_actions),
                physical_outcomes=tuple(physical_outcomes),
                physical_failures=tuple(physical_failures),
                physical_confirmed=physical_outcomes.count("confirmed"),
                physical_failed=physical_outcomes.count("failed"),
                physical_waited=physical_outcomes.count("waited"),
            )
        )
    return tuple(results)
