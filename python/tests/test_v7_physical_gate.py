from __future__ import annotations

from types import SimpleNamespace

from fly_crossy.biomechanics.world import WorldActionResult
from fly_crossy.env import create_game, step_game
from fly_crossy.schema import ACTION_ORDER
from fly_crossy.v7.physical_gate import BiomechanicalActionGate


class _FakeWorld:
    def __init__(self, outcome: str = "confirmed") -> None:
        self.config = SimpleNamespace(decision_timeout_seconds=1.5)
        self.ready = True
        self.requires_reset = False
        self.failure_reason = None
        self.outcome = outcome
        self.run_calls = 0
        self.recovery_steps = 0
        self.started_while_ready: list[bool] = []

    def run(self, intention, limit_seconds=1.5) -> WorldActionResult:
        assert limit_seconds == self.config.decision_timeout_seconds
        self.started_while_ready.append(self.ready)
        self.run_calls += 1
        self.ready = False
        if self.outcome == "failed":
            self.failure_reason = "contact-timeout"
        return WorldActionResult(
            intention_id=intention.intention_id,
            outcome=self.outcome,
            action=intention.action if self.outcome == "confirmed" else "wait",
            requested_action=intention.action,
            completion_time=float(self.run_calls),
        )

    def step(self):
        self.recovery_steps += 1
        self.ready = True
        self.failure_reason = None
        return ()

    def reset(self) -> None:
        self.ready = True
        self.requires_reset = False
        self.failure_reason = None


def test_confirmed_direction_advances_game_with_requested_action() -> None:
    world = _FakeWorld("confirmed")
    gate = BiomechanicalActionGate(world)
    state = create_game("unit-v7-physical-confirmed")

    gated = gate.execute(0)
    advanced = step_game(state, ACTION_ORDER[gated.effective_index]).state

    assert gated.requested_index == 0
    assert gated.effective_index == 0
    assert gated.outcome == "confirmed"
    assert advanced.fly.row == state.fly.row + 1


def test_failed_physical_contact_turns_direction_into_wait() -> None:
    world = _FakeWorld("failed")
    gated = BiomechanicalActionGate(world).execute(0)

    assert gated.requested_index == 0
    assert gated.effective_index == 4
    assert gated.outcome == "failed"
    assert gated.failure_reason == "contact-timeout"


def test_wait_never_enters_the_physical_world() -> None:
    world = _FakeWorld()
    gated = BiomechanicalActionGate(world).execute(4)

    assert gated.outcome == "waited"
    assert gated.effective_index == 4
    assert world.run_calls == 0
    assert world.recovery_steps == 0


def test_next_intention_starts_only_after_previous_motion_is_neutral() -> None:
    world = _FakeWorld()
    gate = BiomechanicalActionGate(world)

    gate.execute(0)
    assert world.ready
    gate.execute(2)

    assert world.started_while_ready == [True, True]
    assert world.recovery_steps == 2
