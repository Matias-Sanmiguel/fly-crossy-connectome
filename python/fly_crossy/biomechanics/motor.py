"""Shared closed-loop 59-signal controller for physical keyboard presses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from fly_crossy.protocol import Action, KeyName

from .body import FlyBodyModel
from .calibration import CalibrationArtifact
from .config import KeyboardConfig
from .contact import ContactOutcome, LegName


class MotorPhase(StrEnum):
    NEUTRAL = "neutral"
    TARGETING = "targeting"
    REACHING = "reaching"
    PRESSING = "pressing"
    CONFIRMED = "confirmed"
    RETRACTING = "retracting"
    SETTLING = "settling"
    FAILED = "failed"


ACTION_TARGETS: Mapping[Action, tuple[LegName, KeyName]] = MappingProxyType(
    {
        "forward": ("front_left", "W"),
        "backward": ("front_right", "S"),
        "left": ("middle_left", "A"),
        "right": ("middle_right", "D"),
    }
)

_LEGAL_TRANSITIONS: Mapping[MotorPhase, frozenset[MotorPhase]] = MappingProxyType(
    {
        MotorPhase.NEUTRAL: frozenset((MotorPhase.TARGETING,)),
        MotorPhase.TARGETING: frozenset((MotorPhase.REACHING, MotorPhase.FAILED)),
        MotorPhase.REACHING: frozenset((MotorPhase.PRESSING, MotorPhase.FAILED)),
        MotorPhase.PRESSING: frozenset((MotorPhase.CONFIRMED, MotorPhase.FAILED)),
        MotorPhase.CONFIRMED: frozenset((MotorPhase.RETRACTING,)),
        MotorPhase.FAILED: frozenset((MotorPhase.RETRACTING,)),
        MotorPhase.RETRACTING: frozenset((MotorPhase.SETTLING, MotorPhase.FAILED)),
        MotorPhase.SETTLING: frozenset((MotorPhase.NEUTRAL, MotorPhase.FAILED)),
    }
)

POSE_TOLERANCE_RADIANS = 1e-6
MAX_POSITION_RATE_RADIANS_PER_SECOND = 20.0


class MotorBusy(RuntimeError):
    """Raised when a second intention would violate single-flight execution."""


@dataclass(frozen=True, slots=True)
class MotorIntention:
    intention_id: str
    action: Action

    def __post_init__(self) -> None:
        if not isinstance(self.intention_id, str) or not self.intention_id:
            raise ValueError("intention_id must be a non-empty string")
        if self.action not in ("forward", "backward", "left", "right", "wait"):
            raise ValueError(f"unknown motor action {self.action!r}")


@dataclass(frozen=True, slots=True)
class MotorTarget:
    intention_id: str
    action: Action
    leg: LegName
    key: KeyName


@dataclass(frozen=True, slots=True)
class BodyState:
    """One immutable proprioceptive sample plus optional contact-gate event."""

    action: np.ndarray
    contact: ContactOutcome | None = None
    stable: bool = True

    def __post_init__(self) -> None:
        values = np.asarray(self.action, dtype=np.float64)
        if values.shape != (59,) or not np.isfinite(values).all():
            raise ValueError("body action must contain exactly 59 finite values")
        if not isinstance(self.stable, bool):
            raise ValueError("stable must be boolean")
        if self.contact is not None and not isinstance(self.contact, ContactOutcome):
            raise TypeError("contact must be a ContactOutcome or None")
        copied = np.array(values, dtype=np.float64, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "action", copied)


@dataclass(frozen=True, slots=True)
class MotorCommand:
    values: np.ndarray
    phase: MotorPhase
    intention_id: str | None

    def __post_init__(self) -> None:
        copied = np.array(self.values, dtype=np.float64, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "values", copied)


class MotorController:
    """Drive one calibrated mapped leg while preserving every other joint."""

    def __init__(
        self,
        body: FlyBodyModel,
        config: KeyboardConfig,
        calibration: CalibrationArtifact,
    ) -> None:
        if not isinstance(body, FlyBodyModel):
            raise TypeError("body must be a FlyBodyModel")
        if not isinstance(config, KeyboardConfig):
            raise TypeError("config must be a KeyboardConfig")
        if not isinstance(calibration, CalibrationArtifact):
            raise TypeError("calibration must be a CalibrationArtifact")
        if calibration.action_order != body.joint_names:
            raise ValueError("calibration action order does not match FlyBody")
        self._body = body
        self._config = config
        self._calibration = calibration
        self._phase = MotorPhase.NEUTRAL
        self._active: MotorTarget | None = None
        self._elapsed = 0.0

    @property
    def phase(self) -> MotorPhase:
        return self._phase

    @property
    def active(self) -> MotorTarget | None:
        return self._active

    def request(self, intention: MotorIntention) -> MotorTarget:
        if not isinstance(intention, MotorIntention):
            raise TypeError("intention must be a MotorIntention")
        if self._active is not None or self._phase is not MotorPhase.NEUTRAL:
            active_id = self._active.intention_id if self._active else "recovery"
            raise MotorBusy(f"motor is already processing {active_id}")
        if intention.action not in ACTION_TARGETS:
            raise ValueError("motor requests must contain a directional action")
        leg, key = ACTION_TARGETS[intention.action]
        target = MotorTarget(intention.intention_id, intention.action, leg, key)
        self._active = target
        self._elapsed = 0.0
        self._transition(MotorPhase.TARGETING)
        return target

    def update(self, state: BodyState, dt: float) -> MotorCommand:
        if isinstance(dt, bool) or not isinstance(dt, (int, float)):
            raise ValueError("dt must be finite and positive")
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not isinstance(state, BodyState):
            raise TypeError("state must be a BodyState")
        current = self._body.validate_action(state.action)
        self._validate_contact(state.contact)

        if self._phase is MotorPhase.NEUTRAL:
            return self._command(current, self._calibration.neutral_pose, dt)

        active = self._active
        if active is None:  # pragma: no cover - private invariant
            raise RuntimeError("non-neutral motor has no active intention")
        trajectory = self._calibration.trajectories[active.action]
        self._elapsed += dt
        entered_failed = False

        if not state.stable and self._phase not in (
            MotorPhase.FAILED,
            MotorPhase.CONFIRMED,
        ):
            self._transition(MotorPhase.FAILED)
            entered_failed = True
        elif (
            state.contact is not None
            and state.contact.kind not in ("pending", "confirmed")
            and self._phase in (
                MotorPhase.TARGETING,
                MotorPhase.REACHING,
                MotorPhase.PRESSING,
            )
        ):
            self._transition(MotorPhase.FAILED)
            entered_failed = True
        elif (
            self._elapsed > self._config.decision_timeout_seconds
            and self._phase
            not in (MotorPhase.FAILED, MotorPhase.RETRACTING, MotorPhase.SETTLING)
        ):
            self._transition(MotorPhase.FAILED)
            entered_failed = True

        if entered_failed:
            desired = trajectory.retract_pose
        elif self._phase is MotorPhase.TARGETING:
            self._transition(MotorPhase.REACHING)
            desired = trajectory.pre_key_pose
        elif self._phase is MotorPhase.REACHING:
            if self._at_pose(current, trajectory.pre_key_pose):
                self._transition(MotorPhase.PRESSING)
                desired = trajectory.press_pose
            else:
                desired = trajectory.pre_key_pose
        elif self._phase is MotorPhase.PRESSING:
            if state.contact is not None and state.contact.kind != "pending":
                if state.contact.kind == "confirmed":
                    self._transition(MotorPhase.CONFIRMED)
                    desired = trajectory.press_pose
                else:
                    self._transition(MotorPhase.FAILED)
                    desired = trajectory.retract_pose
            else:
                desired = trajectory.press_pose
        elif self._phase is MotorPhase.CONFIRMED:
            self._transition(MotorPhase.RETRACTING)
            desired = trajectory.retract_pose
        elif self._phase is MotorPhase.FAILED:
            self._transition(MotorPhase.RETRACTING)
            desired = trajectory.retract_pose
        elif self._phase is MotorPhase.RETRACTING:
            if self._at_pose(current, trajectory.retract_pose):
                self._transition(MotorPhase.SETTLING)
                desired = self._calibration.neutral_pose
            else:
                desired = trajectory.retract_pose
        elif self._phase is MotorPhase.SETTLING:
            if self._at_pose(current, self._calibration.neutral_pose):
                self._transition(MotorPhase.NEUTRAL)
                self._active = None
                self._elapsed = 0.0
            desired = self._calibration.neutral_pose
        else:  # pragma: no cover - enum exhaustiveness
            raise RuntimeError(f"unsupported motor phase {self._phase}")

        return self._command(current, desired, dt)

    def _validate_contact(self, outcome: ContactOutcome | None) -> None:
        if outcome is None:
            return
        active = self._active
        if active is None or outcome.intention_id != active.intention_id:
            raise ValueError("contact outcome intention does not match active intention")
        if outcome.kind == "confirmed":
            if outcome.key != active.key:
                raise ValueError("confirmed contact key does not match active intention")
            if self._phase is not MotorPhase.PRESSING:
                raise ValueError("confirmed contact is illegal outside pressing")

    def _transition(self, target: MotorPhase) -> None:
        if target not in _LEGAL_TRANSITIONS[self._phase]:
            raise RuntimeError(f"illegal motor transition {self._phase} -> {target}")
        self._phase = target

    @staticmethod
    def _at_pose(current: np.ndarray, desired: tuple[float, ...]) -> bool:
        return bool(
            np.max(np.abs(current[:45] - np.asarray(desired[:45])))
            <= POSE_TOLERANCE_RADIANS
        )

    def _command(
        self, current: np.ndarray, desired_values: tuple[float, ...], dt: float
    ) -> MotorCommand:
        desired = self._body.validate_action(desired_values)
        command = np.array(desired, copy=True)
        maximum_step = MAX_POSITION_RATE_RADIANS_PER_SECOND * dt
        correction = desired[:45] - current[:45]
        command[:45] = current[:45] + np.clip(
            correction, -maximum_step, maximum_step
        )
        # Tendons are explicit calibrated controls and adhesion is an explicit
        # six-bit leg mask; neither is interpolated as though it were an angle.
        command[45:59] = desired[45:59]
        validated = self._body.validate_action(command)
        intention_id = self._active.intention_id if self._active else None
        return MotorCommand(validated, self._phase, intention_id)


__all__ = [
    "ACTION_TARGETS",
    "BodyState",
    "MotorBusy",
    "MotorCommand",
    "MotorController",
    "MotorIntention",
    "MotorPhase",
    "MotorTarget",
]
