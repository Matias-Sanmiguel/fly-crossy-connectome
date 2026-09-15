"""Deterministic FK/Jacobian calibration for the physical fly keyboard."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import mujoco
import numpy as np

from .body import ACTION_SIZE, FlyBodyModel
from .config import KeyboardConfig
from .keyboard import meters_to_model_length


ACTION_TARGETS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "forward": ("front_left", "W"),
        "backward": ("front_right", "S"),
        "left": ("middle_left", "A"),
        "right": ("middle_right", "D"),
    }
)
ACTIVE_POSITION_INDICES: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "front_left": tuple(range(3, 10)),
        "middle_left": tuple(range(10, 17)),
        "hind_left": tuple(range(17, 24)),
        "front_right": tuple(range(24, 31)),
        "middle_right": tuple(range(31, 38)),
        "hind_right": tuple(range(38, 45)),
    }
)
ADHESION_INDEX: Mapping[str, int] = MappingProxyType(
    {
        "front_left": 53,
        "middle_left": 54,
        "hind_left": 55,
        "front_right": 56,
        "middle_right": 57,
        "hind_right": 58,
    }
)

SCHEMA_VERSION = 1
MAX_TARGET_ERROR_METERS = 0.00012
PRE_KEY_CLEARANCE_METERS = 0.00025
IK_DAMPING = 0.05
IK_MAX_STEP_RADIANS = 0.08
IK_MAX_ITERATIONS = 500
IK_CONVERGENCE_MODEL_UNITS = 1e-5


def _immutable_pose(values: Sequence[float], field: str) -> tuple[float, ...]:
    pose = tuple(float(value) for value in values)
    if len(pose) != ACTION_SIZE:
        raise ValueError(f"{field} must contain exactly 59 values")
    if not all(math.isfinite(value) for value in pose):
        raise ValueError(f"{field} must contain only finite values")
    return pose


@dataclass(frozen=True, slots=True)
class CalibratedTrajectory:
    action: str
    leg: str
    key: str
    pre_key_pose: tuple[float, ...]
    press_pose: tuple[float, ...]
    retract_pose: tuple[float, ...]
    target_error_meters: float


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    schema_version: int
    manifest_sha256: str
    model_sha256: str
    mesh_inventory_sha256: str
    config_sha256: str
    action_order: tuple[str, ...]
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]
    neutral_pose: tuple[float, ...]
    signal_semantics: Mapping[str, str]
    trajectories: Mapping[str, CalibratedTrajectory]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_hashes(path: Path) -> tuple[str, str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        model_hash = document["modelSha256"]
        mesh_hash = document["meshAssets"]["inventorySha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"unable to read calibration manifest hashes: {exc}") from exc
    if not isinstance(model_hash, str) or not isinstance(mesh_hash, str):
        raise ValueError("calibration manifest hashes must be strings")
    return model_hash, mesh_hash


def _neutral_pose(body: FlyBodyModel) -> np.ndarray:
    """Derive the action neutral from the compiled model, never a hand pose."""
    data = mujoco.MjData(body.model)
    mujoco.mj_forward(body.model, data)
    neutral = np.zeros(ACTION_SIZE, dtype=np.float64)
    for actuator_index in range(45):
        joint_id = int(body.model.actuator_trnid[actuator_index, 0])
        qpos_address = int(body.model.jnt_qposadr[joint_id])
        neutral[actuator_index] = data.qpos[qpos_address]
    # Tendon signals use their compiled zero-rest inputs. Adhesion is binary
    # and remains off in neutral; the active leg alone is enabled while reaching.
    neutral[45:53] = 0.0
    neutral[53:59] = 0.0
    return body.validate_action(neutral)


def _key_targets(
    config: KeyboardConfig, key: str
) -> tuple[np.ndarray, np.ndarray]:
    cap_top = np.array(config.keys[key].center, dtype=np.float64) * 1_000.0
    cap_top[2] += meters_to_model_length(config.key_half_extents_meters[2])
    pre_target = np.array(cap_top, copy=True)
    pre_target[2] += meters_to_model_length(PRE_KEY_CLEARANCE_METERS)
    press_target = np.array(cap_top, copy=True)
    press_target[2] -= meters_to_model_length(config.minimum_travel_meters)
    return pre_target, press_target


def _set_position_pose(
    body: FlyBodyModel, data: mujoco.MjData, pose: Sequence[float]
) -> None:
    for actuator_index in range(45):
        joint_id = int(body.model.actuator_trnid[actuator_index, 0])
        qpos_address = int(body.model.jnt_qposadr[joint_id])
        data.qpos[qpos_address] = pose[actuator_index]


def _solve_leg_pose(
    body: FlyBodyModel,
    seed: Sequence[float],
    leg: str,
    target_model: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Solve one leg with bounded damped least squares on the real MuJoCo FK."""
    pose = np.array(seed, dtype=np.float64, copy=True)
    data = mujoco.MjData(body.model)
    active = ACTIVE_POSITION_INDICES[leg]
    joint_ids = tuple(int(body.model.actuator_trnid[index, 0]) for index in active)
    qpos_addresses = tuple(int(body.model.jnt_qposadr[joint]) for joint in joint_ids)
    dof_addresses = tuple(int(body.model.jnt_dofadr[joint]) for joint in joint_ids)
    site_id = body.leg_sites[leg]
    _set_position_pose(body, data, pose)

    for _ in range(IK_MAX_ITERATIONS):
        mujoco.mj_forward(body.model, data)
        error = target_model - data.site_xpos[site_id]
        if float(np.linalg.norm(error)) <= IK_CONVERGENCE_MODEL_UNITS:
            break
        jacobian = np.zeros((3, body.model.nv), dtype=np.float64)
        angular = np.zeros((3, body.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(body.model, data, jacobian, angular, site_id)
        reduced = jacobian[:, dof_addresses]
        normal = reduced @ reduced.T + (IK_DAMPING**2) * np.eye(3)
        delta = reduced.T @ np.linalg.solve(normal, error)
        maximum = float(np.max(np.abs(delta)))
        if maximum > IK_MAX_STEP_RADIANS:
            delta *= IK_MAX_STEP_RADIANS / maximum
        for offset, (action_index, qpos_address) in enumerate(
            zip(active, qpos_addresses, strict=True)
        ):
            value = data.qpos[qpos_address] + delta[offset]
            value = min(
                max(value, body.lower_limits[action_index] + 1e-7),
                body.upper_limits[action_index] - 1e-7,
            )
            data.qpos[qpos_address] = value
            pose[action_index] = value

    mujoco.mj_forward(body.model, data)
    error_meters = float(
        np.linalg.norm(target_model - data.site_xpos[site_id]) / 1_000.0
    )
    if error_meters > MAX_TARGET_ERROR_METERS:
        raise ValueError(
            f"{leg} target is unreachable: error {error_meters:.12g} meters"
        )
    validated = body.validate_action(pose)
    return validated, error_meters


def generate_calibration(
    body: FlyBodyModel,
    config: KeyboardConfig,
    manifest_path: str | Path,
    config_path: str | Path,
) -> CalibrationArtifact:
    manifest = Path(manifest_path)
    config_file = Path(config_path)
    model_hash, mesh_hash = _manifest_hashes(manifest)
    neutral = _neutral_pose(body)
    trajectories: dict[str, CalibratedTrajectory] = {}

    for action, (leg, key) in ACTION_TARGETS.items():
        pre_target, press_target = _key_targets(config, key)
        pre_key, _ = _solve_leg_pose(body, neutral, leg, pre_target)
        press, target_error = _solve_leg_pose(body, pre_key, leg, press_target)
        press = np.array(press, copy=True)
        press[ADHESION_INDEX[leg]] = 1.0
        press = body.validate_action(press)
        retract, _ = _solve_leg_pose(body, press, leg, pre_target)
        retract = np.array(retract, copy=True)
        retract[53:59] = 0.0
        retract = body.validate_action(retract)
        trajectories[action] = CalibratedTrajectory(
            action=action,
            leg=leg,
            key=key,
            pre_key_pose=_immutable_pose(pre_key, "preKeyPose"),
            press_pose=_immutable_pose(press, "pressPose"),
            retract_pose=_immutable_pose(retract, "retractPose"),
            target_error_meters=target_error,
        )

    return CalibrationArtifact(
        schema_version=SCHEMA_VERSION,
        manifest_sha256=_sha256(manifest),
        model_sha256=model_hash,
        mesh_inventory_sha256=mesh_hash,
        config_sha256=_sha256(config_file),
        action_order=tuple(body.joint_names),
        lower_limits=tuple(float(value) for value in body.lower_limits),
        upper_limits=tuple(float(value) for value in body.upper_limits),
        neutral_pose=_immutable_pose(neutral, "neutralPose"),
        signal_semantics=MappingProxyType(
            {
                "position": "45 joint-angle targets in radians",
                "tendon": "8 dimensionless compiled tendon controls; zero-rest preserved",
                "adhesion": "6 binary controls; only the active mapped leg is enabled",
            }
        ),
        trajectories=MappingProxyType(trajectories),
    )


def _artifact_document(artifact: CalibrationArtifact) -> dict[str, object]:
    return {
        "schemaVersion": artifact.schema_version,
        "manifestSha256": artifact.manifest_sha256,
        "modelSha256": artifact.model_sha256,
        "meshInventorySha256": artifact.mesh_inventory_sha256,
        "configSha256": artifact.config_sha256,
        "actionOrder": list(artifact.action_order),
        "limits": {
            "lower": list(artifact.lower_limits),
            "upper": list(artifact.upper_limits),
        },
        "neutralPose": list(artifact.neutral_pose),
        "signalSemantics": dict(artifact.signal_semantics),
        "trajectories": {
            action: {
                "leg": trajectory.leg,
                "key": trajectory.key,
                "preKeyPose": list(trajectory.pre_key_pose),
                "pressPose": list(trajectory.press_pose),
                "retractPose": list(trajectory.retract_pose),
                "targetErrorMeters": trajectory.target_error_meters,
            }
            for action, trajectory in artifact.trajectories.items()
        },
    }


def write_calibration(path: str | Path, artifact: CalibrationArtifact) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _artifact_document(artifact),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    output.write_text(payload + "\n", encoding="utf-8")


def load_calibration(
    path: str | Path,
    body: FlyBodyModel,
    config: KeyboardConfig,
    manifest_path: str | Path,
    config_path: str | Path,
) -> CalibrationArtifact:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read calibration artifact: {exc}") from exc
    if not isinstance(document, dict) or document.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unsupported calibration schemaVersion")

    manifest = Path(manifest_path)
    config_file = Path(config_path)
    model_hash, mesh_hash = _manifest_hashes(manifest)
    expected_hashes = {
        "manifestSha256": (_sha256(manifest), "manifest hash"),
        "modelSha256": (model_hash, "model hash"),
        "meshInventorySha256": (mesh_hash, "mesh inventory hash"),
        "configSha256": (_sha256(config_file), "config hash"),
    }
    for field, (expected, label) in expected_hashes.items():
        if document.get(field) != expected:
            raise ValueError(f"calibration {label} does not match loaded inputs")

    action_order = tuple(document.get("actionOrder", ()))
    if action_order != body.joint_names:
        raise ValueError("calibration action order does not match FlyBody")
    limits = document.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("calibration limits are missing")
    lower = _immutable_pose(limits.get("lower", ()), "lower limits")
    upper = _immutable_pose(limits.get("upper", ()), "upper limits")
    if not np.array_equal(lower, body.lower_limits) or not np.array_equal(
        upper, body.upper_limits
    ):
        raise ValueError("calibration joint limits do not match FlyBody")
    neutral = _immutable_pose(document.get("neutralPose", ()), "neutralPose")
    body.validate_action(neutral)

    semantics = document.get("signalSemantics")
    if not isinstance(semantics, dict) or set(semantics) != {
        "position",
        "tendon",
        "adhesion",
    } or not all(isinstance(value, str) and value for value in semantics.values()):
        raise ValueError("calibration signal semantics are incomplete")

    raw_trajectories = document.get("trajectories")
    if not isinstance(raw_trajectories, dict) or set(raw_trajectories) != set(
        ACTION_TARGETS
    ):
        raise ValueError("calibration trajectories must contain four actions")
    trajectories: dict[str, CalibratedTrajectory] = {}
    for action, (expected_leg, expected_key) in ACTION_TARGETS.items():
        raw = raw_trajectories[action]
        if not isinstance(raw, dict):
            raise ValueError(f"calibration trajectory {action} is invalid")
        if raw.get("leg") != expected_leg or raw.get("key") != expected_key:
            raise ValueError(f"calibration trajectory {action} mapping is invalid")
        poses = {
            field: _immutable_pose(raw.get(json_name, ()), json_name)
            for field, json_name in (
                ("pre_key_pose", "preKeyPose"),
                ("press_pose", "pressPose"),
                ("retract_pose", "retractPose"),
            )
        }
        for pose in poses.values():
            body.validate_action(pose)
        error = raw.get("targetErrorMeters")
        if (
            isinstance(error, bool)
            or not isinstance(error, (int, float))
            or not math.isfinite(float(error))
            or not 0 <= float(error) <= MAX_TARGET_ERROR_METERS
        ):
            raise ValueError(f"calibration trajectory {action} target error is invalid")

        data = mujoco.MjData(body.model)
        _set_position_pose(body, data, poses["press_pose"])
        mujoco.mj_forward(body.model, data)
        _, target = _key_targets(config, expected_key)
        observed_error = float(
            np.linalg.norm(target - data.site_xpos[body.leg_sites[expected_leg]])
            / 1_000.0
        )
        if not math.isclose(observed_error, float(error), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"calibration trajectory {action} FK target error mismatch")
        trajectories[action] = CalibratedTrajectory(
            action=action,
            leg=expected_leg,
            key=expected_key,
            target_error_meters=float(error),
            **poses,
        )

    return CalibrationArtifact(
        schema_version=SCHEMA_VERSION,
        manifest_sha256=str(document["manifestSha256"]),
        model_sha256=str(document["modelSha256"]),
        mesh_inventory_sha256=str(document["meshInventorySha256"]),
        config_sha256=str(document["configSha256"]),
        action_order=action_order,
        lower_limits=lower,
        upper_limits=upper,
        neutral_pose=neutral,
        signal_semantics=MappingProxyType(dict(semantics)),
        trajectories=MappingProxyType(trajectories),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = args.manifest
    if manifest is None:
        manifest = Path(__file__).resolve().parents[3] / "data/manifests/flybody-v1.json"
    body = FlyBodyModel.load(manifest)
    config = KeyboardConfig.load(args.config)
    artifact = generate_calibration(body, config, manifest, args.config)
    write_calibration(args.output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI subprocess
    raise SystemExit(main())


__all__ = [
    "ACTIVE_POSITION_INDICES",
    "ADHESION_INDEX",
    "CalibrationArtifact",
    "CalibratedTrajectory",
    "generate_calibration",
    "load_calibration",
    "main",
    "write_calibration",
]
