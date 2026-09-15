"""Native-MjSpec construction of six physical, spring-loaded keys."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import mujoco

from .config import KEY_NAMES, KeyboardConfig


MODEL_LENGTHS_PER_METER = 1_000.0
MODEL_MASSES_PER_KILOGRAM = 1_000.0
MODEL_FORCES_PER_NEWTON = 1_000_000.0
MODEL_LINEAR_COEFFICIENTS_PER_SI = 1_000.0


def _scaled(value: float, factor: float, quantity: str) -> float:
    converted = float(value) * factor
    if not math.isfinite(converted):
        raise ValueError(f"{quantity} must convert to a finite model value")
    return converted


def meters_to_model_length(value: float) -> float:
    """Convert public metres to FlyGym/MuJoCo model millimetres."""
    return _scaled(value, MODEL_LENGTHS_PER_METER, "length")


def kilograms_to_model_mass(value: float) -> float:
    """Convert public kilograms to FlyGym/MuJoCo model grams."""
    return _scaled(value, MODEL_MASSES_PER_KILOGRAM, "mass")


def newtons_to_model_force(value: float) -> float:
    """Convert newtons to g*mm/s^2."""
    return _scaled(value, MODEL_FORCES_PER_NEWTON, "force")


def linear_stiffness_to_model(value: float) -> float:
    """Convert N/m to (g*mm/s^2)/mm."""
    return _scaled(value, MODEL_LINEAR_COEFFICIENTS_PER_SI, "stiffness")


def linear_damping_to_model(value: float) -> float:
    """Convert N*s/m to (g*mm/s^2)/(mm/s)."""
    return _scaled(value, MODEL_LINEAR_COEFFICIENTS_PER_SI, "damping")


@dataclass(frozen=True, slots=True)
class KeyIds:
    """Resolved MuJoCo IDs for one physical key."""

    joint_id: int
    geom_id: int
    force_sensor_id: int
    contact_sensor_id: int

    @property
    def sensor_id(self) -> int:
        """Compatibility name for the required force sensor ID."""
        return self.force_sensor_id


def _stem(name: str) -> str:
    return f"key_{name.lower()}"


def build_keyboard_mjcf(config: KeyboardConfig) -> str:
    """Build standalone MJCF using MuJoCo 3.9's native ``MjSpec`` API.

    The public configuration stays in SI. Conversion happens only while values
    cross into the FlyGym millimetre-gram-second model convention here.
    """
    spec = mujoco.MjSpec()
    spec.modelname = "fly_keyboard"
    spec.option.timestep = 1.0 / config.physics_hz
    spec.option.gravity = (0.0, 0.0, -9_810.0)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    travel = meters_to_model_length(config.key_travel_meters)
    half_extents = tuple(
        meters_to_model_length(value) for value in config.key_half_extents_meters
    )
    mass = kilograms_to_model_mass(config.key_mass_kilograms)
    stiffness = linear_stiffness_to_model(
        config.return_stiffness_newtons_per_meter
    )
    damping = linear_damping_to_model(
        config.return_damping_newton_seconds_per_meter
    )

    for name in KEY_NAMES:
        stem = _stem(name)
        center = tuple(meters_to_model_length(value) for value in config.keys[name].center)
        body = spec.worldbody.add_body(name=stem, pos=center)
        body.add_joint(
            name=f"{stem}_travel",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=(0.0, 0.0, -1.0),
            limited=True,
            range=(0.0, travel),
            stiffness=stiffness,
            damping=damping,
            solref_limit=(0.005, 1.0),
            solimp_limit=(0.99, 0.999, 0.001, 0.5, 2.0),
        )
        body.add_geom(
            name=f"{stem}_cap",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half_extents,
            mass=mass,
        )
        body.add_site(
            name=f"{stem}_site",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half_extents,
        )
        spec.add_sensor(
            name=f"{stem}_force",
            type=mujoco.mjtSensor.mjSENS_FORCE,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=f"{stem}_site",
        )
        spec.add_sensor(
            name=f"{stem}_contact",
            type=mujoco.mjtSensor.mjSENS_TOUCH,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=f"{stem}_site",
        )

    # Compilation here rejects invalid native references before the XML is
    # exposed for later composition.
    spec.compile()
    return spec.to_xml()


def _required_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise ValueError(f"keyboard model is missing {name}")
    return object_id


def resolve_key_ids(
    model: mujoco.MjModel,
    *,
    prefix: str = "",
) -> Mapping[str, KeyIds]:
    """Resolve and validate the stable six-key MuJoCo identity contract.

    ``prefix`` is used when the keyboard spec is attached to another MuJoCo
    model and its elements are deliberately namespaced.
    """
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")

    resolved: dict[str, KeyIds] = {}

    for name in KEY_NAMES:
        stem = f"{prefix}{_stem(name)}"

        ids = KeyIds(
            joint_id=_required_id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{stem}_travel",
            ),
            geom_id=_required_id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{stem}_cap",
            ),
            force_sensor_id=_required_id(
                model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                f"{stem}_force",
            ),
            contact_sensor_id=_required_id(
                model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                f"{stem}_contact",
            ),
        )

        if (
            model.sensor_type[ids.force_sensor_id]
            != mujoco.mjtSensor.mjSENS_FORCE
        ):
            raise ValueError(f"{stem}_force is not a force sensor")

        if (
            model.sensor_type[ids.contact_sensor_id]
            != mujoco.mjtSensor.mjSENS_TOUCH
        ):
            raise ValueError(f"{stem}_contact is not a touch sensor")

        resolved[name] = ids

    for field in (
        "joint_id",
        "geom_id",
        "force_sensor_id",
        "contact_sensor_id",
    ):
        values = {getattr(ids, field) for ids in resolved.values()}
        if len(values) != len(KEY_NAMES):
            raise ValueError(f"keyboard {field} values must be unique")

    return MappingProxyType(resolved)


__all__ = [
    "KEY_NAMES",
    "KeyIds",
    "build_keyboard_mjcf",
    "kilograms_to_model_mass",
    "linear_damping_to_model",
    "linear_stiffness_to_model",
    "meters_to_model_length",
    "newtons_to_model_force",
    "resolve_key_ids",
]
