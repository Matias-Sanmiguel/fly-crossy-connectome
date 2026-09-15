"""Unified MuJoCo world joining FlyBody, keyboard, motor control, and contact gating."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Literal, Mapping

from flygym.compose import ActuatorType, ContactParams, KinematicPosePreset
from flygym.compose.fly import FlyBody
from flygym.flybody import (
    FlyBodyActuatedDOFPreset,
    FlyBodyAxisOrder,
    FlyBodyJointPreset,
    FlyBodySkeleton,
)
import mujoco
import numpy as np

from fly_crossy.protocol import Action, KeyName

from .body import ACTION_SIZE, LEG_SITE_TARGETS, FlyBodyModel
from .calibration import CalibrationArtifact, load_calibration
from .config import KEY_NAMES, KeyboardConfig
from .contact import ContactGate, ContactOutcome, ContactSample, KeyContact, LegName
from .keyboard import build_keyboard_mjcf, resolve_key_ids
from .mapping import ACTION_TARGETS
from .motor import BodyState, MotorController, MotorIntention, MotorPhase

NATIVE_TIMESTEP_SECONDS = 0.0001
MAX_SNAPSHOT_HZ = 30.0
_MODEL_LENGTHS_PER_METER = 1_000.0
_MODEL_FORCES_PER_NEWTON = 1_000_000.0
KEYBOARD_PREFIX = "keyboard_"


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    simulation_time: float
    body_position: tuple[float, float, float]
    body_quaternion: tuple[float, float, float, float]
    joint_positions: tuple[float, ...]
    key_travel: Mapping[KeyName, float]
    motor_phase: MotorPhase


@dataclass(frozen=True, slots=True)
class WorldActionResult:
    intention_id: str
    outcome: Literal["confirmed", "waited", "failed"]
    action: Action
    requested_action: Action
    completion_time: float


WorldEvent = WorldSnapshot | WorldActionResult


def _build_fly_spec() -> tuple[
    mujoco.MjSpec,
    Mapping[LegName, tuple[str, ...]],
]:
    """Build FlyBody and retain the real tarsus collision geom names."""
    fly = FlyBody(name="flybody")

    skeleton = FlyBodySkeleton(
        axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
        joint_preset=FlyBodyJointPreset.ALL_BIOLOGICAL,
    )

    neutral_pose = KinematicPosePreset.FLYBODY_NEUTRAL
    fly.add_joints(skeleton, neutral_pose)

    all_dofs = list(skeleton.iter_jointdofs())

    head_dofs = [
        dof
        for dof in all_dofs
        if dof.parent.name == "c_thorax"
        and dof.child.name == "c_head"
    ]

    leg_dofs = skeleton.get_actuated_dofs_from_preset(
        FlyBodyActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )

    if len(head_dofs) != 3 or len(leg_dofs) != 42:
        raise ValueError("FlyGym FlyBody DoF composition changed")

    fly.add_actuators(
        [*head_dofs, *leg_dofs],
        ActuatorType.POSITION,
        neutral_input=neutral_pose,
        kp=100,
    )

    fly.add_tendons()
    fly.add_tendon_actuators()
    fly.add_leg_adhesion(add_labrum=False)

    tarsus_geom_names: dict[LegName, tuple[str, ...]] = {}

    for stable_name, (site_name, short_leg) in LEG_SITE_TARGETS.items():
        tarsus = fly.mjcf_root.body(f"{short_leg}_tarsus5")

        if tarsus is None:
            raise ValueError(
                f"FlyGym FlyBody is missing {short_leg}_tarsus5"
            )

        tarsus.add_site(
            name=site_name,
            pos=(0.0, 0.0, 0.0),
        )

        segment = fly.BODY_SEGMENT_CLASS(
            f"{short_leg}_tarsus5"
        )

        geom_names = tuple(
            geom.name
            for geom in fly.bodyseg_to_mjcfgeom[segment]
        )

        if not geom_names:
            raise ValueError(
                f"FlyGym FlyBody has no collision geometry for "
                f"{short_leg}_tarsus5"
            )

        tarsus_geom_names[stable_name] = geom_names

    return (
        fly.mjcf_root.copy(),
        MappingProxyType(tarsus_geom_names),
    )

def _add_keyboard_contact_pairs(
    spec: mujoco.MjSpec,
    tarsus_geom_names: Mapping[LegName, tuple[str, ...]],
) -> None:
    """Make every terminal tarsus physically collide with every keyboard key.

    FlyBody mesh geoms intentionally use contype=conaffinity=0, so their
    environmental contacts must be declared explicitly.
    """
    params = ContactParams()
    pair_index = 0

    for leg, geom_names in tarsus_geom_names.items():
        for key in KEY_NAMES:
            key_geom_name = (
                f"{KEYBOARD_PREFIX}"
                f"key_{key.lower()}_cap"
            )

            for tarsus_geom_name in geom_names:
                spec.add_pair(
                    name=f"keyboard_contact_{pair_index}",
                    geomname1=tarsus_geom_name,
                    geomname2=key_geom_name,
                    friction=params.get_friction_tuple(),
                    solref=params.get_solref_tuple(),
                    solimp=params.get_solimp_tuple(),
                    margin=params.margin,
                )

                pair_index += 1

    if pair_index == 0:
        raise ValueError(
            "no FlyBody-to-keyboard contact pairs were created"
        )

def _rebind_body(model: mujoco.MjModel, reference: FlyBodyModel) -> FlyBodyModel:
    """Resolve all FlyBody IDs again after the unified world is compiled."""
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
        for index in range(model.nu)
    )
    if model.nu != ACTION_SIZE or actuator_names != reference.joint_names:
        raise ValueError("unified world changed the pinned 59-signal actuator order")

    actuator_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in actuator_names
    )
    if any(actuator_id < 0 for actuator_id in actuator_ids):
        raise ValueError("unified world is missing a required FlyBody actuator")

    lower_limits = np.array(model.actuator_ctrlrange[actuator_ids, 0], copy=True)
    upper_limits = np.array(model.actuator_ctrlrange[actuator_ids, 1], copy=True)
    if not np.array_equal(lower_limits, reference.lower_limits) or not np.array_equal(
        upper_limits, reference.upper_limits
    ):
        raise ValueError("unified world changed the pinned actuator limits")
    lower_limits.setflags(write=False)
    upper_limits.setflags(write=False)

    leg_sites: dict[str, int] = {}
    for stable_name, (site_name, _) in LEG_SITE_TARGETS.items():
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"unified world is missing required site {site_name}")
        leg_sites[stable_name] = site_id
    if len(set(leg_sites.values())) != len(LEG_SITE_TARGETS):
        raise ValueError("unified world FlyBody leg sites must be unique")

    return FlyBodyModel(
        model=model,
        joint_names=reference.joint_names,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        leg_sites=MappingProxyType(leg_sites),
        actuator_ids=actuator_ids,
        action_groups=reference.action_groups,
    )


def _compose_model(config: KeyboardConfig, reference: FlyBodyModel) -> FlyBodyModel:
    fly_spec, tarsus_geom_names = _build_fly_spec()
    keyboard_spec = mujoco.MjSpec.from_string(build_keyboard_mjcf(config))

    attachment = fly_spec.worldbody.add_frame(name="fly_keyboard_attachment")
    
    fly_spec.attach(
    keyboard_spec,
    frame=attachment,
    prefix=KEYBOARD_PREFIX,
    )
    
    _add_keyboard_contact_pairs(
    fly_spec,
    tarsus_geom_names,
    )
    
    model = fly_spec.compile()

    if abs(float(model.opt.timestep) - NATIVE_TIMESTEP_SECONDS) > 1e-12:
        raise ValueError(
            "unified FlyBody world must preserve the native 0.0001 s MuJoCo timestep"
        )
    return _rebind_body(model, reference)


class BiomechanicalWorld:
    """Authoritative physical path from one high-level action to a key press."""

    def __init__(
        self,
        body: FlyBodyModel,
        config: KeyboardConfig,
        calibration: CalibrationArtifact,
        *,
        contacts_enabled: bool = True,
    ) -> None:
        if not isinstance(body, FlyBodyModel):
            raise TypeError("body must be a FlyBodyModel")
        if not isinstance(config, KeyboardConfig):
            raise TypeError("config must be a KeyboardConfig")
        if not isinstance(calibration, CalibrationArtifact):
            raise TypeError("calibration must be a CalibrationArtifact")
        if not isinstance(contacts_enabled, bool):
            raise TypeError("contacts_enabled must be boolean")

        self.body = body
        self.model = body.model
        self.config = config
        self.calibration = calibration
        self.key_ids = resolve_key_ids(
            self.model,
            prefix=KEYBOARD_PREFIX,
        )
        self.data = mujoco.MjData(self.model)
        self._contacts_enabled = contacts_enabled

        outer_dt = 1.0 / config.physics_hz
        self._native_substeps = round(outer_dt / self.model.opt.timestep)
        if self._native_substeps < 1 or abs(
            self._native_substeps * self.model.opt.timestep - outer_dt
        ) > 1e-12:
            raise ValueError("physicsHz must be exactly representable by native substeps")
        self._motor_stride = config.physics_hz // config.motor_hz
        self._snapshot_stride = max(1, int(np.ceil(config.physics_hz / MAX_SNAPSHOT_HZ)))

        self._tarsus_geoms = self._resolve_tarsus_geoms()
        self._thorax_body_id = self._resolve_thorax_body()
        self._motor = MotorController(body, config, calibration)
        self._gate = ContactGate(config)
        self._active: MotorIntention | None = None
        self._pending_contact: ContactOutcome | None = None
        self._result_emitted = False
        self._wait_elapsed = 0.0
        self._outer_tick = 0
        self.reset()

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        config_path: str | Path,
        calibration_path: str | Path,
        *,
        contacts_enabled: bool = True,
    ) -> "BiomechanicalWorld":
        manifest = Path(manifest_path)
        config_file = Path(config_path)
        calibration_file = Path(calibration_path)
        reference = FlyBodyModel.load(manifest)
        config = KeyboardConfig.load(config_file)
        body = _compose_model(config, reference)
        calibration = load_calibration(
            calibration_file,
            body,
            config,
            manifest,
            config_file,
        )
        return cls(body, config, calibration, contacts_enabled=contacts_enabled)

    @property
    def ready(self) -> bool:
        return self._active is None and self._motor.phase is MotorPhase.NEUTRAL

    @property
    def motor_phase(self) -> MotorPhase:
        return self._motor.phase
    
    @property
    def requires_reset(self) -> bool:
        return self._motor.requires_reset


    @property
    def failure_reason(self) -> str | None:
        return self._motor.failure_reason

    @property
    def native_substeps_per_tick(self) -> int:
        return self._native_substeps

    @property
    def motor_outer_tick_stride(self) -> int:
        return self._motor_stride

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._motor = MotorController(self.body, self.config, self.calibration)
        self._gate = ContactGate(self.config)
        self._active = None
        self._pending_contact = None
        self._result_emitted = False
        self._wait_elapsed = 0.0
        self._outer_tick = 0

        for actuator_index in range(45):
            joint_id = int(self.model.actuator_trnid[actuator_index, 0])
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[qpos_address] = self.calibration.neutral_pose[actuator_index]
        self.data.ctrl[list(self.body.actuator_ids)] = np.asarray(
            self.calibration.neutral_pose, dtype=np.float64
        )
        mujoco.mj_forward(self.model, self.data)

    def start_intention(self, intention: MotorIntention) -> None:
        if not isinstance(intention, MotorIntention):
            raise TypeError("intention must be a MotorIntention")
        if not self.ready:
            raise RuntimeError("biomechanical world is still processing the previous intention")

        self._active = intention
        self._pending_contact = None
        self._result_emitted = False
        self._wait_elapsed = 0.0
        if intention.action in ACTION_TARGETS:
            target = self._motor.request(intention)
            self._gate.begin(intention.intention_id, target.key)
        elif intention.action != "wait":
            raise ValueError(f"unsupported action {intention.action!r}")

    def step(self) -> tuple[WorldEvent, ...]:
        events: list[WorldEvent] = []
        active = self._active

        if active is not None and active.action == "wait":
            self._wait_elapsed += 1.0 / self.config.physics_hz
        elif active is not None and self._outer_tick % self._motor_stride == 0:
            contact = self._pending_contact

            target_touched = False

            if active is not None and active.action in ACTION_TARGETS:
                expected_leg, expected_key = ACTION_TARGETS[active.action]

                physical_sample = self._contact_sample()

                intended_key_state = next(
                    item
                    for item in physical_sample.contacts
                    if item.key == expected_key
                )

                target_touched = (
                    intended_key_state.touching_tarsi
                    == (expected_leg,)
                )

            state = BodyState.from_mujoco(
                self.body,
                self.data,
                contact=contact,
                target_touched=target_touched,
                stable=self._is_stable(),
            )
            command = self._motor.update(state, 1.0 / self.config.motor_hz)
            self.data.ctrl[list(self.body.actuator_ids)] = command.values
            if contact is not None:
                self._pending_contact = None

        for _ in range(self._native_substeps):
            mujoco.mj_step(self.model, self.data)
        self._outer_tick += 1

        if active is not None and active.action in ACTION_TARGETS:
            outcome = self._gate.sample(
                self._contact_sample(),
                confirmation_enabled=(
                    self._motor.phase is MotorPhase.PRESSING
                ),
            )
            if outcome is not None and outcome.kind != "pending":
                if outcome.kind == "confirmed" and self._motor.phase is not MotorPhase.PRESSING:
                    # A real press occurred outside the permitted motor phase. It stays a
                    # physical event, but cannot become a directional game action.
                    self._pending_contact = ContactOutcome(
                        "cancelled", outcome.intention_id, outcome.key
                    )
                    if not self._result_emitted:
                        events.append(self._emit_result("failed", "wait"))
                else:
                    self._pending_contact = outcome
                    if outcome.kind == "confirmed":
                        if not self._result_emitted:
                            events.append(self._emit_result("confirmed", active.action))
                    elif not self._result_emitted:
                        events.append(self._emit_result("failed", "wait"))

        if (
            active is not None
            and active.action == "wait"
            and not self._result_emitted
            and self._wait_elapsed >= 1.0 / self.config.motor_hz
        ):
            events.append(self._emit_result("waited", "wait"))

        if self._outer_tick % self._snapshot_stride == 0:
            events.append(self.snapshot())

        if self._active is not None and self._result_emitted:
            if self._active.action == "wait":
                self._active = None
            elif self._motor.phase is MotorPhase.NEUTRAL:
                self._active = None
                self._pending_contact = None

        return tuple(events)

    def step_until_event(self, limit_seconds: float = 1.5) -> Iterator[WorldEvent]:
        if limit_seconds <= 0:
            raise ValueError("limit_seconds must be positive")
        limit_ticks = int(np.ceil(limit_seconds * self.config.physics_hz))
        for _ in range(limit_ticks):
            for event in self.step():
                yield event
                if isinstance(event, WorldActionResult):
                    return

    def run(self, intention: MotorIntention, limit_seconds: float = 1.5) -> WorldActionResult:
        self.start_intention(intention)
        for event in self.step_until_event(limit_seconds):
            if isinstance(event, WorldActionResult):
                return event
        if not self._result_emitted:
            result = self._emit_result("failed", "wait")
            cancelled = self._gate.cancel()
            if cancelled is not None:
                self._pending_contact = cancelled
            return result
        raise RuntimeError("world emitted a terminal result without returning it")

    def snapshot(self) -> WorldSnapshot:
        body_position = tuple(
            float(value) / _MODEL_LENGTHS_PER_METER
            for value in self.data.xpos[self._thorax_body_id]
        )
        body_quaternion = tuple(
            float(value) for value in self.data.xquat[self._thorax_body_id]
        )
        key_travel = MappingProxyType(
            {key: self._key_travel_meters(key) for key in KEY_NAMES}
        )
        return WorldSnapshot(
            simulation_time=float(self.data.time),
            body_position=body_position,
            body_quaternion=body_quaternion,
            joint_positions=tuple(float(value) for value in self._position_state()),
            key_travel=key_travel,
            motor_phase=self._motor.phase,
        )

    def _emit_result(
        self,
        outcome: Literal["confirmed", "waited", "failed"],
        action: Action,
    ) -> WorldActionResult:
        active = self._active
        if active is None:
            raise RuntimeError("cannot emit a result without an active intention")
        if self._result_emitted:
            raise RuntimeError("an intention may emit exactly one terminal result")
        self._result_emitted = True
        return WorldActionResult(
            intention_id=active.intention_id,
            outcome=outcome,
            action=action,
            requested_action=active.action,
            completion_time=float(self.data.time),
        )

    def _contact_sample(self) -> ContactSample:
        touching_by_key = self._touching_tarsi()
        contacts: list[KeyContact] = []
        for key in KEY_NAMES:
            ids = self.key_ids[key]
            sensor_address = int(
                self.model.sensor_adr[ids.force_sensor_id]
            )

            sensor_dim = int(
                self.model.sensor_dim[ids.force_sensor_id]
            )

            if sensor_dim != 3:
                raise ValueError(
                    "keyboard force sensor must expose a 3D force vector"
                )

            force_vector = np.asarray(
                self.data.sensordata[
                    sensor_address : sensor_address + sensor_dim
                ],
                dtype=np.float64,
            )

            # The key slide axis is vertical. The MuJoCo force sensor is
            # expressed in the site's local frame, so the Z component is
            # the physically relevant key-normal load.
            normal_force_model = abs(float(force_vector[2]))
            contacts.append(
                KeyContact(
                    key=key,
                    travel=self._key_travel_meters(key),
                    normal_force=normal_force_model / _MODEL_FORCES_PER_NEWTON,
                    touching_tarsi=(
                        touching_by_key[key] if self._contacts_enabled else ()
                    ),
                )
            )
        return ContactSample(time=float(self.data.time), contacts=tuple(contacts))

    def _touching_tarsi(self) -> Mapping[KeyName, tuple[LegName, ...]]:
        result: dict[KeyName, set[LegName]] = {key: set() for key in KEY_NAMES}
        key_by_geom = {ids.geom_id: key for key, ids in self.key_ids.items()}
        leg_by_geom = {
            geom_id: leg
            for leg, geom_ids in self._tarsus_geoms.items()
            for geom_id in geom_ids
        }
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = int(contact.geom1)
            second = int(contact.geom2)
            if first in key_by_geom and second in leg_by_geom:
                result[key_by_geom[first]].add(leg_by_geom[second])
            elif second in key_by_geom and first in leg_by_geom:
                result[key_by_geom[second]].add(leg_by_geom[first])
        return MappingProxyType(
            {key: tuple(sorted(legs)) for key, legs in result.items()}
        )

    def _resolve_tarsus_geoms(self) -> Mapping[LegName, tuple[int, ...]]:
        resolved: dict[LegName, tuple[int, ...]] = {}
        for leg, site_id in self.body.leg_sites.items():
            body_id = int(self.model.site_bodyid[site_id])
            geom_ids = tuple(
                index
                for index in range(self.model.ngeom)
                if int(self.model.geom_bodyid[index]) == body_id
            )
            if not geom_ids:
                raise ValueError(f"unified world has no collision geometry for {leg}")
            resolved[leg] = geom_ids
        return MappingProxyType(resolved)

    def _resolve_thorax_body(self) -> int:
        for name in ("c_thorax", "flybody/c_thorax"):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return body_id
        raise ValueError("unified world is missing the FlyBody thorax")

    def _key_travel_meters(self, key: KeyName) -> float:
        joint_id = self.key_ids[key].joint_id
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        return max(
            0.0,
            float(self.data.qpos[qpos_address]) / _MODEL_LENGTHS_PER_METER,
        )

    def _position_state(self) -> np.ndarray:
        positions = np.empty(45, dtype=np.float64)
        for actuator_index in range(45):
            joint_id = int(self.model.actuator_trnid[actuator_index, 0])
            positions[actuator_index] = self.data.qpos[self.model.jnt_qposadr[joint_id]]
        return positions

    def _is_stable(self) -> bool:
        return bool(
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
            and np.isfinite(self.data.ctrl).all()
        )


__all__ = [
    "BiomechanicalWorld",
    "MAX_SNAPSHOT_HZ",
    "NATIVE_TIMESTEP_SECONDS",
    "WorldActionResult",
    "WorldEvent",
    "WorldSnapshot",
]
