"""Immutable adapter for the pinned 59-signal FlyBody motor contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Sequence

from flygym import assets_dir
from flygym.compose import ActuatorType, KinematicPosePreset
from flygym.compose.fly import FlyBody
from flygym.flybody import (
    FlyBodyActuatedDOFPreset,
    FlyBodyAxisOrder,
    FlyBodyJointPreset,
    FlyBodySkeleton,
)
import mujoco
import numpy as np


ACTION_SIZE = 59
EXPECTED_GROUP_COUNTS = {"position": 45, "tendon": 8, "adhesion": 6}
IMMUTABLE_MANIFEST_FIELDS = {
    "source": "https://github.com/TuragaLab/flybody",
    "integrationSource": "https://github.com/NeLy-EPFL/flygym",
    "model": "flybody-compatible",
    "flygymVersion": "2.1.0",
    "mujocoVersion": "3.9.0",
    "upstreamCommit": "ca65a510c2afe6ac61c51df4f274c8d190c2f95f",
    "modelFile": "model/flybody/fruitfly.xml",
    "licenseFile": "public/data/flybody/LICENSE",
    "noticeFile": "public/data/flybody/NOTICE.md",
}
LEG_SITE_TARGETS = {
    "front_left": ("front_left_tarsus", "lf"),
    "front_right": ("front_right_tarsus", "rf"),
    "middle_left": ("middle_left_tarsus", "lm"),
    "middle_right": ("middle_right_tarsus", "rm"),
    "hind_left": ("hind_left_tarsus", "lh"),
    "hind_right": ("hind_right_tarsus", "rh"),
}
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_COMMIT = re.compile(r"[0-9a-fA-F]{40}\Z")


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid FlyBody manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("invalid FlyBody manifest: expected an object")

    for field, expected in IMMUTABLE_MANIFEST_FIELDS.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"immutable manifest field {field} must equal {expected!r}"
            )

    checksum = manifest.get("modelSha256")
    if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
        raise ValueError("modelSha256 must be a 64-character hexadecimal checksum")
    commit = manifest.get("upstreamCommit")
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ValueError("upstreamCommit must be a 40-character hexadecimal commit")
    if manifest.get("schemaVersion") != 1:
        raise ValueError("unsupported FlyBody manifest schemaVersion")
    if manifest.get("actionSize") != ACTION_SIZE:
        raise ValueError(f"FlyBody manifest actionSize must be {ACTION_SIZE}")
    if manifest.get("actionGroups") != EXPECTED_GROUP_COUNTS:
        raise ValueError(
            "FlyBody actionGroups must contain 45 position, 8 tendon, and "
            "6 adhesion signals"
        )
    signals = manifest.get("actionSignals")
    if (
        not isinstance(signals, list)
        or len(signals) != ACTION_SIZE
        or not all(isinstance(item, str) and item for item in signals)
        or len(set(signals)) != ACTION_SIZE
    ):
        raise ValueError("actionSignals must contain 59 unique non-empty names")
    required_sites = manifest.get("requiredSites")
    expected_sites = [site for site, _ in LEG_SITE_TARGETS.values()]
    if required_sites != expected_sites:
        raise ValueError("requiredSites does not match the stable six-leg site contract")
    return manifest


def _verify_runtime_and_source(manifest: Mapping[str, object]) -> Path:
    expected_flygym = manifest.get("flygymVersion")
    expected_mujoco = manifest.get("mujocoVersion")
    if version("flygym") != expected_flygym:
        raise ValueError(
            f"FlyGym version mismatch: expected {expected_flygym}, "
            f"found {version('flygym')}"
        )
    if version("mujoco") != expected_mujoco:
        raise ValueError(
            f"MuJoCo version mismatch: expected {expected_mujoco}, "
            f"found {version('mujoco')}"
        )

    relative_model_file = manifest.get("modelFile")
    if relative_model_file != "model/flybody/fruitfly.xml":
        raise ValueError("modelFile must name FlyGym's bundled FlyBody source XML")
    model_path = (assets_dir / relative_model_file).resolve()
    expected_root = assets_dir.resolve()
    if not model_path.is_relative_to(expected_root) or not model_path.is_file():
        raise ValueError("FlyBody modelFile is unavailable outside the pinned package")
    actual_checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_checksum.lower() != str(manifest["modelSha256"]).lower():
        raise ValueError(
            "FlyBody model checksum mismatch: the installed source is not the "
            "pinned model"
        )
    return model_path


def _build_flygym_model() -> tuple[mujoco.MjModel, tuple[str, ...]]:
    """Build the native-MjSpec FlyBody composition matching legacy walking actions."""
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
        if dof.parent.name == "c_thorax" and dof.child.name == "c_head"
    ]
    leg_dofs = skeleton.get_actuated_dofs_from_preset(
        FlyBodyActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    if len(head_dofs) != 3 or len(leg_dofs) != 42:
        raise ValueError(
            "FlyGym FlyBody DoF composition changed: expected 3 head and 42 "
            "active-leg position controls"
        )
    fly.add_actuators(
        [*head_dofs, *leg_dofs],
        ActuatorType.POSITION,
        neutral_input=neutral_pose,
        kp=100,
    )
    fly.add_tendons()
    fly.add_tendon_actuators()
    fly.add_leg_adhesion(add_labrum=False)

    if len(fly.get_actuated_jointdofs_order(ActuatorType.POSITION)) != 45:
        raise ValueError("FlyGym FlyBody did not produce 45 position controls")
    if len(fly.get_actuated_jointdofs_order(ActuatorType.TENDON)) != 8:
        raise ValueError("FlyGym FlyBody did not produce 8 tendon controls")
    if len(fly.leg_to_adhesionactuator) != 6:
        raise ValueError("FlyGym FlyBody did not produce 6 leg adhesion controls")

    for site_name, leg in LEG_SITE_TARGETS.values():
        tarsus = fly.mjcf_root.body(f"{leg}_tarsus5")
        if tarsus is None:
            raise ValueError(f"FlyGym FlyBody is missing {leg}_tarsus5")
        tarsus.add_site(name=site_name, pos=(0.0, 0.0, 0.0))

    model, _ = fly.compile()
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
        for index in range(model.nu)
    )
    return model, actuator_names


@dataclass(frozen=True, slots=True)
class FlyBodyModel:
    """Validated, immutable view of the project's 59 motor signals."""

    model: mujoco.MjModel
    joint_names: tuple[str, ...]
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    leg_sites: Mapping[str, int]
    actuator_ids: tuple[int, ...]
    action_groups: Mapping[str, tuple[int, int]]

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @classmethod
    def load(cls, manifest_path: str | Path) -> "FlyBodyModel":
        manifest = _load_manifest(Path(manifest_path))
        _verify_runtime_and_source(manifest)
        model, actuator_names = _build_flygym_model()

        expected_names = tuple(manifest["actionSignals"])
        if model.nu != ACTION_SIZE or actuator_names != expected_names:
            raise ValueError(
                "compiled FlyBody actuator ordering does not match the pinned "
                "59-signal contract"
            )
        if len(set(actuator_names)) != ACTION_SIZE:
            raise ValueError("compiled FlyBody actuator names must be unique")

        actuator_ids = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in actuator_names
        )
        if any(actuator_id < 0 for actuator_id in actuator_ids):
            raise ValueError("compiled FlyBody is missing a required actuator")
        lower_limits = np.array(model.actuator_ctrlrange[actuator_ids, 0], copy=True)
        upper_limits = np.array(model.actuator_ctrlrange[actuator_ids, 1], copy=True)
        if (
            not np.isfinite(lower_limits).all()
            or not np.isfinite(upper_limits).all()
            or np.any(lower_limits >= upper_limits)
        ):
            raise ValueError("compiled FlyBody has invalid or non-finite joint limits")
        lower_limits.setflags(write=False)
        upper_limits.setflags(write=False)

        leg_sites = {}
        for stable_name, (site_name, _) in LEG_SITE_TARGETS.items():
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                raise ValueError(f"compiled FlyBody is missing required site {site_name}")
            leg_sites[stable_name] = site_id
        if len(set(leg_sites.values())) != len(LEG_SITE_TARGETS):
            raise ValueError("compiled FlyBody leg sites must be unique")

        action_groups = MappingProxyType(
            {"position": (0, 45), "tendon": (45, 53), "adhesion": (53, 59)}
        )
        return cls(
            model=model,
            joint_names=actuator_names,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            leg_sites=MappingProxyType(leg_sites),
            actuator_ids=actuator_ids,
            action_groups=action_groups,
        )

    def validate_action(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        action = np.asarray(values, dtype=np.float64)
        if action.shape != (ACTION_SIZE,):
            raise ValueError(f"action must contain exactly {ACTION_SIZE} values")
        if not np.isfinite(action).all():
            raise ValueError("action values must be finite")
        outside = (action < self.lower_limits) | (action > self.upper_limits)
        if np.any(outside):
            first = int(np.flatnonzero(outside)[0])
            raise ValueError(
                "action exceeds joint limit for "
                f"{self.joint_names[first]} at index {first}"
            )
        validated = np.array(action, dtype=np.float64, copy=True)
        validated.setflags(write=False)
        return validated
