from __future__ import annotations

from dataclasses import replace

import pytest

from fly_crossy.v7.evaluation import (
    AggregateMetrics,
    EpisodeMetrics,
    aggregate_metrics,
    decide_promotion,
    paired_comparison,
    robust_key,
    evaluate_policy,
)


def _episode(seed: str, score: float, *, action: int = 0) -> EpisodeMetrics:
    return EpisodeMetrics(
        seed=seed,
        steps=100,
        score=score,
        success=True,
        progress_qualified=True,
        terminal_reason=None,
        fatal_actions=0,
        labelled_states=100,
        actions=(action,) * 100,
    )


def _suite(scores: list[float], *, actions: list[int] | None = None):
    actions = actions or [index % 5 for index in range(len(scores))]
    return tuple(
        _episode(f"seed-{index}", score, action=actions[index])
        for index, score in enumerate(scores)
    )


def test_aggregate_and_robust_key_have_exact_order() -> None:
    episodes = (
        _episode("a", 2.0, action=0),
        replace(
            _episode("b", 4.0, action=1),
            success=False,
            progress_qualified=False,
            steps=50,
            fatal_actions=2,
            labelled_states=10,
            actions=(1,) * 10,
        ),
    )
    metrics = aggregate_metrics(episodes)
    assert metrics.episodes == 2
    assert metrics.successes == 1
    assert metrics.progress_qualified == 1
    assert metrics.mean_score == 3.0
    assert metrics.median_score == 3.0
    assert metrics.mean_survival_steps == 75.0
    assert metrics.fatal_action_rate == pytest.approx(2 / 110)
    assert sum(metrics.action_distribution) == pytest.approx(1.0)
    assert robust_key(metrics) == (1.0, 1.0, 3.0, 3.0, 75.0, pytest.approx(-2 / 110))


def test_one_seed_improvement_is_diagnostic_only_and_cannot_promote() -> None:
    decision = decide_promotion(
        profile="80",
        predecessor=_suite([10.0]),
        candidate=_suite([12.0]),
        planner_agreement=0.95,
        agreement_threshold=0.8,
        final_test_passed=True,
    )
    assert decision.next_action == "continue"
    assert not decision.predicates["sufficientPairedSeeds"]


@pytest.mark.parametrize("wins,accepted", [(59, False), (60, True)])
def test_paired_win_fraction_requires_sixty_percent(wins: int, accepted: bool) -> None:
    predecessor = _suite([10.0] * 100)
    candidate = _suite([12.0] * wins + [9.9] * (100 - wins))
    decision = decide_promotion(
        profile="80",
        predecessor=predecessor,
        candidate=candidate,
        planner_agreement=0.95,
        agreement_threshold=0.8,
        final_test_passed=True,
    )
    assert decision.predicates["pairedWinsAtLeast60Percent"] is accepted
    assert (decision.next_action == "promote") is accepted


@pytest.mark.parametrize(
    "mutation,predicate",
    [
        ("mean", "meanScoreAtLeast10PercentBetter"),
        ("median", "medianScoreNoRegression"),
        ("success", "successCountNoRegression"),
        ("agreement", "plannerAgreementPassed"),
        ("collapse", "actionDistributionHealthy"),
        ("final", "finalTestPassed"),
    ],
)
def test_every_promotion_gate_is_authoritative(mutation: str, predicate: str) -> None:
    predecessor = list(_suite([10.0] * 10))
    candidate = list(_suite([12.0] * 10))
    agreement = 0.9
    final = True
    if mutation == "mean":
        candidate = list(_suite([10.9] * 10))
    elif mutation == "median":
        predecessor = list(_suite([10.0] * 10))
        candidate = list(_suite([22.0] * 4 + [9.0] * 6))
    elif mutation == "success":
        candidate[0] = replace(candidate[0], success=False)
    elif mutation == "agreement":
        agreement = 0.799
    elif mutation == "collapse":
        candidate = list(_suite([12.0] * 10, actions=[0] * 10))
    elif mutation == "final":
        final = False

    decision = decide_promotion(
        profile="80",
        predecessor=predecessor,
        candidate=candidate,
        planner_agreement=agreement,
        agreement_threshold=0.8,
        final_test_passed=final,
    )
    assert not decision.predicates[predicate]
    assert decision.next_action != "promote"
    if mutation in {"success", "final"}:
        assert decision.next_action == "reject"


def test_action_distribution_gate_rejects_missing_predecessor_actions() -> None:
    predecessor_actions = (0,) * 34 + (2,) * 33 + (3,) * 33
    candidate_actions = (0,) * 60 + (4,) * 40
    predecessor = tuple(
        replace(_episode(f"seed-{index}", 10.0), actions=predecessor_actions)
        for index in range(3)
    )
    candidate = tuple(
        replace(_episode(f"seed-{index}", 12.0), actions=candidate_actions)
        for index in range(3)
    )

    decision = decide_promotion(
        profile="80",
        predecessor=predecessor,
        candidate=candidate,
        planner_agreement=0.95,
        agreement_threshold=0.8,
        final_test_passed=True,
    )

    assert max(aggregate_metrics(candidate).action_distribution) <= 0.85
    assert not decision.predicates["actionDistributionHealthy"]
    assert decision.next_action != "promote"


def test_action_distribution_gate_rejects_single_action_dominance_above_85_percent() -> None:
    predecessor_actions = (0,) * 34 + (2,) * 33 + (3,) * 33
    candidate_actions = (0,) * 86 + (4,) * 14
    predecessor = tuple(
        replace(_episode(f"seed-{index}", 10.0), actions=predecessor_actions)
        for index in range(3)
    )
    candidate = tuple(
        replace(_episode(f"seed-{index}", 12.0), actions=candidate_actions)
        for index in range(3)
    )

    decision = decide_promotion(
        profile="80",
        predecessor=predecessor,
        candidate=candidate,
        planner_agreement=0.95,
        agreement_threshold=0.8,
        final_test_passed=True,
    )

    assert not decision.predicates["actionDistributionHealthy"]


def test_teacher_and_student_agreement_thresholds_can_be_expressed_exactly() -> None:
    predecessor = _suite([10.0] * 5)
    candidate = _suite([12.0] * 5)
    teacher = decide_promotion(
        profile="80",
        predecessor=predecessor,
        candidate=candidate,
        planner_agreement=0.899,
        agreement_threshold=0.9,
        final_test_passed=True,
    )
    student = decide_promotion(
        profile="80",
        predecessor=predecessor,
        candidate=candidate,
        planner_agreement=0.799,
        agreement_threshold=0.8,
        final_test_passed=True,
    )
    assert teacher.next_action == "continue"
    assert student.next_action == "continue"


def test_smoke_never_promotes_even_when_all_metrics_pass() -> None:
    decision = decide_promotion(
        profile="smoke",
        predecessor=_suite([10.0] * 5),
        candidate=_suite([12.0] * 5),
        planner_agreement=1.0,
        agreement_threshold=0.8,
        final_test_passed=True,
    )
    assert decision.next_action == "continue"
    assert not decision.predicates["profileCanPromote"]


def test_pairing_rejects_duplicate_or_mismatched_seed_sets() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        paired_comparison(_suite([1.0, 2.0]), (_episode("seed-0", 3.0),) * 2)
    with pytest.raises(ValueError, match="seed"):
        paired_comparison(_suite([1.0]), (_episode("different", 2.0),))


def test_aggregate_rejects_empty_or_malformed_action_indices() -> None:
    with pytest.raises(ValueError):
        aggregate_metrics(())
    with pytest.raises(ValueError):
        aggregate_metrics((replace(_episode("a", 1.0), actions=(5,)),))


class _WaitPolicy:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def act(self, state, frame, observation) -> int:
        assert frame.shape == (24, 48, 3)
        assert observation.shape == (517,)
        return 0


class _Gate:
    def __init__(self, *, failed: bool = False) -> None:
        self.failed = failed
        self.resets = 0

    def reset_episode(self) -> None:
        self.resets += 1

    def execute(self, requested_index: int):
        from fly_crossy.v7.physical_gate import GatedAction

        if self.failed and requested_index != 4:
            return GatedAction(requested_index, 4, "failed", "contact-disabled")
        outcome = "waited" if requested_index == 4 else "confirmed"
        return GatedAction(requested_index, requested_index, outcome)


def test_evaluate_policy_runs_closed_loop_and_labels_every_action() -> None:
    policy = _WaitPolicy()
    gate = _Gate()
    episodes = evaluate_policy(
        policy,
        ("unit-v7-validation-000", "unit-v7-validation-001"),
        max_steps=2,
        planner_depth=1,
        action_gate=gate,
    )
    assert policy.resets == 2
    assert tuple(episode.seed for episode in episodes) == (
        "unit-v7-validation-000",
        "unit-v7-validation-001",
    )
    assert all(episode.labelled_states == len(episode.actions) for episode in episodes)
    assert all(episode.requested_actions == episode.actions for episode in episodes)
    assert all(episode.effective_actions == episode.actions for episode in episodes)
    assert all(episode.physical_confirmed == 2 for episode in episodes)


def test_evaluate_policy_defaults_to_logical_execution_without_physics() -> None:
    episodes = evaluate_policy(
        _WaitPolicy(),
        ("unit-v7-validation-logical-default",),
        max_steps=3,
        planner_depth=1,
    )
    episode = episodes[0]
    assert episode.requested_actions == episode.actions
    assert episode.effective_actions == episode.actions
    assert episode.physical_outcomes == ()
    assert episode.physical_failures == ()
    assert episode.physical_confirmed == 0
    assert episode.physical_failed == 0
    assert episode.physical_waited == 0


def test_evaluation_reports_failed_physical_actions_as_effective_waits() -> None:
    episodes = evaluate_policy(
        _WaitPolicy(),
        ("unit-v7-validation-physical-failure",),
        max_steps=3,
        planner_depth=1,
        action_gate=_Gate(failed=True),
    )
    episode = episodes[0]
    assert episode.requested_actions == (0, 0, 0)
    assert episode.effective_actions == (4, 4, 4)
    assert episode.physical_failed == 3
    assert episode.physical_confirmed == 0
    assert episode.physical_failures == ("contact-disabled",) * 3


def test_physical_evaluation_remains_deterministic() -> None:
    kwargs = {
        "max_steps": 3,
        "planner_depth": 1,
        "action_gate": _Gate(),
    }
    first = evaluate_policy(
        _WaitPolicy(), ("unit-v7-validation-deterministic",), **kwargs
    )
    kwargs["action_gate"] = _Gate()
    second = evaluate_policy(
        _WaitPolicy(), ("unit-v7-validation-deterministic",), **kwargs
    )
    assert first == second
