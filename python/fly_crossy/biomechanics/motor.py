"""Shared closed-loop 59-signal controller for physical keyboard presses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from types import MappingProxyType
from typing import Mapping

import mujoco
import numpy as np

from fly_crossy.protocol import Action, KeyName

from .body import FlyBodyModel
from .calibration import ACTIVE_POSITION_INDICES, CalibrationArtifact
from .config import KeyboardConfig
from .contact import ContactOutcome, LegName
from .mapping import ACTION_TARGETS


class MotorPhase(StrEnum):
    NEUTRAL = "neutral"
    TARGETING = "targeting"
    REACHING = "reaching"
    PRESSING = "pressing"
    CONFIRMED = "confirmed"
    RETRACTING = "retracting"
    SETTLING = "settling"
    FAILED = "failed"


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

MAX_POSITION_RATE_RADIANS_PER_SECOND = 20.0


class MotorBusy(RuntimeError):
    """Raised when a second intention would violate single-flight execution."""


class MotorResetRequired(RuntimeError):
    """Raised when recovery failed and a coordinated reset is required."""


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

    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    site_positions: Mapping[LegName, tuple[float, float, float]]
    contact: ContactOutcome | None = None
    stable: bool = True

    def __post_init__(self) -> None:
        positions = np.asarray(self.joint_positions, dtype=np.float64)
        velocities = np.asarray(self.joint_velocities, dtype=np.float64)
        if positions.shape != (45,) or not np.isfinite(positions).all():
            raise ValueError("body state must contain exactly 45 joint positions")
        if velocities.shape != (45,) or not np.isfinite(velocities).all():
            raise ValueError("body state must contain exactly 45 joint velocities")
        sites = dict(self.site_positions)
        if set(sites) != {
            "front_left",
            "front_right",
            "middle_left",
            "middle_right",
            "hind_left",
            "hind_right",
        }:
            raise ValueError("body state must contain all six tarsus site positions")
        immutable_sites: dict[LegName, tuple[float, float, float]] = {}
        for leg, value in sites.items():
            vector = np.asarray(value, dtype=np.float64)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError("tarsus site positions must be finite 3-vectors")
            immutable_sites[leg] = tuple(float(item) for item in vector)
        if not isinstance(self.stable, bool):
            raise ValueError("stable must be boolean")
        if self.contact is not None and not isinstance(self.contact, ContactOutcome):
            raise TypeError("contact must be a ContactOutcome or None")
        copied_positions = np.array(positions, dtype=np.float64, copy=True)
        copied_velocities = np.array(velocities, dtype=np.float64, copy=True)
        copied_positions.setflags(write=False)
        copied_velocities.setflags(write=False)
        object.__setattr__(self, "joint_positions", copied_positions)
        object.__setattr__(self, "joint_velocities", copied_velocities)
        object.__setattr__(self, "site_positions", MappingProxyType(immutable_sites))

    @classmethod
    def from_mujoco(
        cls,
        body: FlyBodyModel,
        data: mujoco.MjData,
        *,
        contact: ContactOutcome | None = None,
        stable: bool = True,
    ) -> "BodyState":
        if not isinstance(body, FlyBodyModel) or not isinstance(data, mujoco.MjData):
            raise TypeError("from_mujoco requires FlyBodyModel and MjData")
        positions = np.empty(45, dtype=np.float64)
        velocities = np.empty(45, dtype=np.float64)
        for actuator_index in range(45):
            joint_id = int(body.model.actuator_trnid[actuator_index, 0])
            positions[actuator_index] = data.qpos[body.model.jnt_qposadr[joint_id]]
            velocities[actuator_index] = data.qvel[body.model.jnt_dofadr[joint_id]]
        sites = {
            leg: tuple(float(value) / 1_000.0 for value in data.site_xpos[site_id])
            for leg, site_id in body.leg_sites.items()
        }
        return cls(positions, velocities, sites, contact=contact, stable=stable)


@dataclass(frozen=True, slots=True)
class MotorCommand:
    values: np.ndarray
    phase: MotorPhase
    intention_id: str | None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (59,) or not np.isfinite(values).all():
            raise ValueError("motor command must contain exactly 59 finite values")
        copied = np.array(values, dtype=np.float64, copy=True)
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
        self._decision_elapsed = 0.0
        self._recovery_elapsed = 0.0
        self._requires_reset = False
        self._failure_reason: str | None = None
        self._site_targets = self._build_site_targets()

    @property
    def phase(self) -> MotorPhase:
        return self._phase

    @property
    def active(self) -> MotorTarget | None:
        return self._active

    @property
    def requires_reset(self) -> bool:
        return self._requires_reset

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    def reset(self) -> None:
        self._phase = MotorPhase.NEUTRAL
        self._active = None
        self._decision_elapsed = 0.0
        self._recovery_elapsed = 0.0
        self._requires_reset = False
        self._failure_reason = None

    def request(self, intention: MotorIntention) -> MotorTarget:
        if not isinstance(intention, MotorIntention):
            raise TypeError("intention must be a MotorIntention")
        if self._requires_reset:
            raise MotorResetRequired(
                f"motor reset required after {self._failure_reason or 'recovery failure'}"
            )
        if self._active is not None or self._phase is not MotorPhase.NEUTRAL:
            active_id = self._active.intention_id if self._active else "recovery"
            raise MotorBusy(f"motor is already processing {active_id}")
        if intention.action not in ACTION_TARGETS:
            raise ValueError("motor requests must contain a directional action")
        leg, key = ACTION_TARGETS[intention.action]
        target = MotorTarget(intention.intention_id, intention.action, leg, key)
        self._active = target
        self._decision_elapsed = 0.0
        self._recovery_elapsed = 0.0
        self._failure_reason = None
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
        self._validate_contact(state.contact)
        current = state.joint_positions

        if self._phase is MotorPhase.NEUTRAL:
            return self._command(current, self._calibration.neutral_pose, dt)

        active = self._active
        if active is None:  # pragma: no cover - private invariant
            raise RuntimeError("non-neutral motor has no active intention")
        trajectory = self._calibration.trajectories[active.action]
        if self._requires_reset:
            return self._command(
                current, self._without_adhesion(self._calibration.neutral_pose), dt
            )
        if self._phase in (
            MotorPhase.TARGETING,
            MotorPhase.REACHING,
            MotorPhase.PRESSING,
        ):
            self._decision_elapsed += dt
        elif self._phase in (
            MotorPhase.CONFIRMED,
            MotorPhase.FAILED,
            MotorPhase.RETRACTING,
            MotorPhase.SETTLING,
        ):
            self._recovery_elapsed += dt
        entered_failed = False

        if (
            self._phase in (MotorPhase.RETRACTING, MotorPhase.SETTLING)
            and self._recovery_elapsed > self._config.decision_timeout_seconds
        ):
            self._halt("recovery-timeout")
            return self._command(
                current, self._without_adhesion(self._calibration.neutral_pose), dt
            )

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
            self._decision_elapsed > self._config.decision_timeout_seconds
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
            if self._at_target(
                state,
                active.leg,
                trajectory.pre_key_pose,
                self._site_targets[active.action]["pre_key"],
            ):
                self._transition(MotorPhase.PRESSING)
                desired = trajectory.press_pose
            else:
                desired = trajectory.pre_key_pose
        elif self._phase is MotorPhase.PRESSING:
            if state.contact is not None and state.contact.kind != "pending":
                if state.contact.kind == "confirmed":
                    self._transition(MotorPhase.CONFIRMED)
                    desired = self._without_adhesion(trajectory.press_pose)
                else:
                    self._transition(MotorPhase.FAILED)
                    desired = trajectory.retract_pose
            else:
                desired = trajectory.press_pose
        elif self._phase is MotorPhase.CONFIRMED:
            self._transition(MotorPhase.RETRACTING)
            self._recovery_elapsed = 0.0
            desired = trajectory.retract_pose
        elif self._phase is MotorPhase.FAILED:
            self._transition(MotorPhase.RETRACTING)
            desired = trajectory.retract_pose
        elif self._phase is MotorPhase.RETRACTING:
            if self._at_target(
                state,
                active.leg,
                trajectory.retract_pose,
                self._site_targets[active.action]["retract"],
            ):
                self._transition(MotorPhase.SETTLING)
                desired = self._calibration.neutral_pose
            else:
                desired = trajectory.retract_pose
        elif self._phase is MotorPhase.SETTLING:
            if self._at_target(
                state,
                active.leg,
                self._calibration.neutral_pose,
                self._site_targets[active.action]["neutral"],
            ):
                self._transition(MotorPhase.NEUTRAL)
                self._active = None
                self._decision_elapsed = 0.0
                self._recovery_elapsed = 0.0
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

    def _at_target(
        self,
        state: BodyState,
        leg: LegName,
        desired: tuple[float, ...],
        desired_site: tuple[float, float, float],
    ) -> bool:
        indices = ACTIVE_POSITION_INDICES[leg]
        position_error = np.max(
            np.abs(
                state.joint_positions[list(indices)]
                - np.asarray(desired)[list(indices)]
            )
        )
        velocity = np.max(np.abs(state.joint_velocities[list(indices)]))
        site_error = np.linalg.norm(
            np.asarray(state.site_positions[leg]) - np.asarray(desired_site)
        )
        return bool(
            position_error <= self._calibration.position_tolerance_radians
            and velocity
            <= self._calibration.velocity_tolerance_radians_per_second
            and site_error <= self._calibration.site_tolerance_meters
        )

    def _build_site_targets(
        self,
    ) -> dict[str, dict[str, tuple[float, float, float]]]:
        targets: dict[str, dict[str, tuple[float, float, float]]] = {}
        for action, trajectory in self._calibration.trajectories.items():
            targets[action] = {
                "pre_key": self._site_for_pose(trajectory.leg, trajectory.pre_key_pose),
                "retract": self._site_for_pose(trajectory.leg, trajectory.retract_pose),
                "neutral": self._site_for_pose(
                    trajectory.leg, self._calibration.neutral_pose
                ),
            }
        return targets

    def _site_for_pose(
        self, leg: LegName, pose: tuple[float, ...]
    ) -> tuple[float, float, float]:
        data = mujoco.MjData(self._body.model)
        for actuator_index in range(45):
            joint_id = int(self._body.model.actuator_trnid[actuator_index, 0])
            data.qpos[self._body.model.jnt_qposadr[joint_id]] = pose[actuator_index]
        mujoco.mj_forward(self._body.model, data)
        return tuple(
            float(value) / 1_000.0
            for value in data.site_xpos[self._body.leg_sites[leg]]
        )

    @staticmethod
    def _without_adhesion(values: tuple[float, ...]) -> tuple[float, ...]:
        return (*values[:53], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _halt(self, reason: str) -> None:
        self._phase = MotorPhase.FAILED
        self._requires_reset = True
        self._failure_reason = reason

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
    "MotorResetRequired",
    "MotorTarget",
]
