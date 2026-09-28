"""Non-differentiable biomechanical authority gate for V7 actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol

from fly_crossy.biomechanics.motor import MotorIntention
from fly_crossy.biomechanics.recovery import recover_world
from fly_crossy.biomechanics.world import BiomechanicalWorld, WorldActionResult
from fly_crossy.schema import ACTION_ORDER, Action


PhysicalOutcome = Literal["confirmed", "waited", "failed"]


class PhysicalWorld(Protocol):
    ready: bool
    requires_reset: bool
    failure_reason: str | None
    config: "PhysicalWorldConfig"

    def run(
        self, intention: MotorIntention, limit_seconds: float = 1.5
    ) -> WorldActionResult: ...

    def step(self) -> tuple[object, ...]: ...

    def reset(self) -> None: ...


class PhysicalWorldConfig(Protocol):
    decision_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class GatedAction:
    requested_index: int
    effective_index: int
    outcome: PhysicalOutcome
    failure_reason: str | None = None


class ActionGate(Protocol):
    def reset_episode(self) -> None: ...

    def execute(self, requested_index: int) -> GatedAction: ...


def _load_default_world() -> BiomechanicalWorld:
    root = Path(__file__).resolve().parents[3]
    return BiomechanicalWorld.load(
        root / "data/manifests/flybody-v1.json",
        root / "configs/biomechanics-v1.json",
        root / "data/calibration/keyboard-reach-v1.json",
    )


class BiomechanicalActionGate:
    """Execute canonical V7 actions through FlyBody before advancing the game."""

    def __init__(
        self,
        world: PhysicalWorld | None = None,
        *,
        world_factory: Callable[[], PhysicalWorld] = _load_default_world,
    ) -> None:
        self._world = world
        self._world_factory = world_factory
        self._sequence = 0

    @property
    def world(self) -> PhysicalWorld:
        if self._world is None:
            self._world = self._world_factory()
        return self._world

    def reset_episode(self) -> None:
        if self._world is None:
            return
        if not self._world.ready:
            recover_world(self._world)
        self._world.reset()
        if not self._world.ready:
            raise RuntimeError("biomechanical reset failed at episode boundary")

    def execute(self, requested_index: int) -> GatedAction:
        if requested_index < 0 or requested_index >= len(ACTION_ORDER):
            raise ValueError("requested action is outside canonical ACTION_ORDER")

        requested = ACTION_ORDER[requested_index]
        wait_index = ACTION_ORDER.index(Action.WAIT)
        if requested is Action.WAIT:
            return GatedAction(requested_index, wait_index, "waited")

        world = self.world
        if not world.ready:
            recover_world(world)
        if not world.ready:
            raise RuntimeError("biomechanical world did not recover before intention")

        self._sequence += 1
        result = world.run(
            MotorIntention(f"i-v7gate{self._sequence:012d}", requested.value),
            limit_seconds=float(world.config.decision_timeout_seconds),
        )
        if result.requested_action != requested.value:
            raise RuntimeError("biomechanical result does not match requested action")
        if result.outcome == "confirmed" and result.action != requested.value:
            raise RuntimeError("confirmed biomechanical result changed the requested action")
        failure_reason = world.failure_reason if result.outcome == "failed" else None
        effective_index = requested_index if result.outcome == "confirmed" else wait_index

        if not world.ready:
            recover_world(world)
        if not world.ready:
            raise RuntimeError("biomechanical world did not recover after intention")

        return GatedAction(
            requested_index=requested_index,
            effective_index=effective_index,
            outcome=result.outcome,
            failure_reason=failure_reason,
        )


__all__ = [
    "ActionGate",
    "BiomechanicalActionGate",
    "GatedAction",
    "PhysicalOutcome",
]
