from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from fly_crossy.biomechanics.body import FlyBodyModel
from fly_crossy.biomechanics.calibration import CalibrationArtifact, load_calibration
from fly_crossy.biomechanics.config import KeyboardConfig
from fly_crossy.biomechanics.contact import ContactOutcome
from fly_crossy.biomechanics.motor import (
    ACTION_TARGETS,
    BodyState,
    MotorBusy,
    MotorController,
    MotorIntention,
    MotorPhase,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "data" / "manifests" / "flybody-v1.json"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "biomechanics-v1.json"
CALIBRATION_PATH = REPOSITORY_ROOT / "data" / "calibration" / "keyboard-reach-v1.json"


@pytest.fixture(scope="session")
def body() -> FlyBodyModel:
    return FlyBodyModel.load(MANIFEST_PATH)


@pytest.fixture(scope="session")
def config() -> KeyboardConfig:
    return KeyboardConfig.load(CONFIG_PATH)


@pytest.fixture(scope="session")
def calibration(body: FlyBodyModel, config: KeyboardConfig) -> CalibrationArtifact:
    return load_calibration(CALIBRATION_PATH, body, config, MANIFEST_PATH, CONFIG_PATH)


@pytest.fixture
def motor(
    body: FlyBodyModel,
    config: KeyboardConfig,
    calibration: CalibrationArtifact,
) -> MotorController:
    return MotorController(body, config, calibration)


def intention(identifier: str, action: str) -> MotorIntention:
    return MotorIntention(identifier, action)


def state(
    calibration: CalibrationArtifact,
    *,
    pose: str = "neutral",
    contact: ContactOutcome | None = None,
    stable: bool = True,
) -> BodyState:
    values = calibration.neutral_pose if pose == "neutral" else getattr(
        calibration.trajectories["forward"], f"{pose}_pose"
    )
    return BodyState(values, contact=contact, stable=stable)


@pytest.mark.parametrize(
    ("action", "leg", "key"),
    (
        ("forward", "front_left", "W"),
        ("backward", "front_right", "S"),
        ("left", "middle_left", "A"),
        ("right", "middle_right", "D"),
    ),
)
def test_action_mapping_uses_the_fixed_leg_and_key_contract(
    action: str, leg: str, key: str, motor: MotorController
) -> None:
    target = motor.request(intention(f"i-{action}", action))

    assert ACTION_TARGETS[action] == (leg, key)
    assert (target.leg, target.key) == (leg, key)
    assert motor.phase is MotorPhase.TARGETING


def test_wait_never_creates_a_physical_key_target(motor: MotorController) -> None:
    with pytest.raises(ValueError, match="directional"):
        motor.request(intention("i-wait", "wait"))


def test_motor_cannot_accept_two_intentions(motor: MotorController) -> None:
    motor.request(intention("i-1", "forward"))

    with pytest.raises(MotorBusy, match="i-1"):
        motor.request(intention("i-2", "left"))


def test_successful_lifecycle_has_no_skipped_phase(
    motor: MotorController, calibration: CalibrationArtifact
) -> None:
    motor.request(intention("i-success", "forward"))
    observed = [motor.phase]

    command = motor.update(state(calibration), 0.01)
    observed.append(command.phase)
    command = motor.update(state(calibration, pose="pre_key"), 0.01)
    observed.append(command.phase)
    command = motor.update(state(calibration, pose="press"), 0.01)
    observed.append(command.phase)
    confirmed = ContactOutcome("confirmed", "i-success", "W")
    command = motor.update(state(calibration, pose="press", contact=confirmed), 0.01)
    observed.append(command.phase)
    command = motor.update(state(calibration, pose="press"), 0.01)
    observed.append(command.phase)
    command = motor.update(state(calibration, pose="retract"), 0.01)
    observed.append(command.phase)
    command = motor.update(state(calibration), 0.01)
    observed.append(command.phase)
    command = motor.update(state(calibration), 0.01)
    observed.append(command.phase)

    assert observed == [
        MotorPhase.TARGETING,
        MotorPhase.REACHING,
        MotorPhase.PRESSING,
        MotorPhase.PRESSING,
        MotorPhase.CONFIRMED,
        MotorPhase.RETRACTING,
        MotorPhase.SETTLING,
        MotorPhase.NEUTRAL,
        MotorPhase.NEUTRAL,
    ]
    assert command.intention_id is None


@pytest.mark.parametrize(
    "outcome",
    (
        ContactOutcome("wrong-key", "i-fail", "S"),
        ContactOutcome("ambiguous", "i-fail", None),
        ContactOutcome("unsafe-force", "i-fail", "W"),
        ContactOutcome("timed-out", "i-fail", None),
    ),
)
def test_terminal_contact_failure_retracts_before_settling(
    outcome: ContactOutcome,
    motor: MotorController,
    calibration: CalibrationArtifact,
) -> None:
    motor.request(intention("i-fail", "forward"))
    motor.update(state(calibration), 0.01)
    motor.update(state(calibration, pose="pre_key"), 0.01)

    failed = motor.update(state(calibration, pose="press", contact=outcome), 0.01)
    retracting = motor.update(state(calibration, pose="press"), 0.01)
    settling = motor.update(state(calibration, pose="retract"), 0.01)
    neutral = motor.update(state(calibration), 0.01)

    assert [failed.phase, retracting.phase, settling.phase, neutral.phase] == [
        MotorPhase.FAILED,
        MotorPhase.RETRACTING,
        MotorPhase.SETTLING,
        MotorPhase.NEUTRAL,
    ]


def test_instability_enters_failed_without_skipping_retraction(
    motor: MotorController, calibration: CalibrationArtifact
) -> None:
    motor.request(intention("i-unstable", "left"))

    failed = motor.update(state(calibration, stable=False), 0.01)
    retracting = motor.update(state(calibration), 0.01)

    assert failed.phase is MotorPhase.FAILED
    assert retracting.phase is MotorPhase.RETRACTING


def test_wrong_key_during_reaching_fails_immediately(
    motor: MotorController, calibration: CalibrationArtifact
) -> None:
    motor.request(intention("i-early-contact", "forward"))
    motor.update(state(calibration), 0.01)

    failed = motor.update(
        state(
            calibration,
            contact=ContactOutcome("wrong-key", "i-early-contact", "S"),
        ),
        0.01,
    )

    assert failed.phase is MotorPhase.FAILED


def test_instability_while_retracting_uses_the_legal_recovery_failure_edge(
    motor: MotorController, calibration: CalibrationArtifact
) -> None:
    motor.request(intention("i-recovery", "forward"))
    motor.update(state(calibration), 0.01)
    motor.update(state(calibration, pose="pre_key"), 0.01)
    motor.update(
        state(
            calibration,
            pose="press",
            contact=ContactOutcome("confirmed", "i-recovery", "W"),
        ),
        0.01,
    )
    assert motor.update(state(calibration, pose="press"), 0.01).phase is MotorPhase.RETRACTING

    failed = motor.update(state(calibration, pose="press", stable=False), 0.01)

    assert failed.phase is MotorPhase.FAILED


def test_phase_timeout_enters_failed_and_rejects_invalid_dt(
    motor: MotorController, calibration: CalibrationArtifact, config: KeyboardConfig
) -> None:
    motor.request(intention("i-timeout", "right"))
    with pytest.raises(ValueError, match="finite and positive"):
        motor.update(state(calibration), float("nan"))
    with pytest.raises(ValueError, match="finite and positive"):
        motor.update(state(calibration), 0.0)

    failed = motor.update(
        state(calibration), config.decision_timeout_seconds + 0.001
    )

    assert failed.phase is MotorPhase.FAILED


def test_foreign_or_impossible_contact_outcome_is_rejected_without_state_mutation(
    motor: MotorController, calibration: CalibrationArtifact
) -> None:
    motor.request(intention("i-owned", "forward"))
    motor.update(state(calibration), 0.01)
    previous_phase = motor.phase

    with pytest.raises(ValueError, match="intention"):
        motor.update(
            state(
                calibration,
                pose="pre_key",
                contact=ContactOutcome("confirmed", "i-foreign", "W"),
            ),
            0.01,
        )

    assert motor.phase is previous_phase


def test_every_motor_command_is_a_copied_validated_59_vector(
    motor: MotorController, calibration: CalibrationArtifact, body: FlyBodyModel
) -> None:
    motor.request(intention("i-shape", "backward"))
    body_values = np.array(calibration.neutral_pose, copy=True)
    command = motor.update(BodyState(body_values), 0.01)
    body_values[:] = body.upper_limits

    assert command.values.shape == (59,)
    assert command.values.flags.writeable is False
    assert np.all(command.values >= body.lower_limits)
    assert np.all(command.values <= body.upper_limits)
    assert command.values[45:59].tolist() == [0] * 14


def test_calibration_is_real_reachable_and_keeps_inactive_position_joints_neutral(
    body: FlyBodyModel, config: KeyboardConfig, calibration: CalibrationArtifact
) -> None:
    assert calibration.action_order == body.joint_names
    assert calibration.model_sha256 == json.loads(MANIFEST_PATH.read_text())["modelSha256"]
    assert calibration.mesh_inventory_sha256 == json.loads(MANIFEST_PATH.read_text())["meshAssets"]["inventorySha256"]
    assert calibration.config_sha256 == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()

    leg_position_indices = {
        "front_left": range(3, 10),
        "middle_left": range(10, 17),
        "front_right": range(24, 31),
        "middle_right": range(31, 38),
    }
    for action, (leg, key) in ACTION_TARGETS.items():
        trajectory = calibration.trajectories[action]
        assert (trajectory.leg, trajectory.key) == (leg, key)
        assert trajectory.target_error_meters <= 0.00012
        for pose in (
            trajectory.pre_key_pose,
            trajectory.press_pose,
            trajectory.retract_pose,
        ):
            assert len(pose) == 59
            body.validate_action(pose)
        inactive = set(range(45)) - set(leg_position_indices[leg])
        assert np.array(trajectory.press_pose)[sorted(inactive)] == pytest.approx(
            np.array(calibration.neutral_pose)[sorted(inactive)], abs=0
        )
        assert trajectory.pre_key_pose[45:53] == calibration.neutral_pose[45:53]
        assert trajectory.press_pose[45:53] == calibration.neutral_pose[45:53]
        assert trajectory.retract_pose[45:53] == calibration.neutral_pose[45:53]
        active_adhesion = [0.0] * 6
        active_adhesion[{"front_left": 0, "middle_left": 1, "front_right": 3, "middle_right": 4}[leg]] = 1.0
        assert list(trajectory.pre_key_pose[53:59]) == [0.0] * 6
        assert list(trajectory.press_pose[53:59]) == active_adhesion
        assert list(trajectory.retract_pose[53:59]) == [0.0] * 6

        data = mujoco.MjData(body.model)
        for actuator_index in range(45):
            joint_id = int(body.model.actuator_trnid[actuator_index, 0])
            qpos_address = int(body.model.jnt_qposadr[joint_id])
            data.qpos[qpos_address] = trajectory.press_pose[actuator_index]
        mujoco.mj_forward(body.model, data)
        target_model = np.array(config.keys[key].center) * 1_000
        target_model[2] += config.key_half_extents_meters[2] * 1_000
        target_model[2] -= config.minimum_travel_meters * 1_000
        actual_model = data.site_xpos[body.leg_sites[leg]]
        assert np.linalg.norm(actual_model - target_model) / 1_000 == pytest.approx(
            trajectory.target_error_meters, abs=1e-12
        )


def test_calibration_value_objects_and_arrays_are_immutable(
    calibration: CalibrationArtifact,
) -> None:
    with pytest.raises((FrozenInstanceError, AttributeError)):
        calibration.schema_version = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        calibration.trajectories["forward"] = calibration.trajectories["left"]  # type: ignore[index]
    with pytest.raises(TypeError):
        calibration.neutral_pose[0] = 0.0  # type: ignore[index]


def test_calibration_cli_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    command = [
        sys.executable,
        "-m",
        "fly_crossy.biomechanics.calibration",
        "--config",
        str(CONFIG_PATH),
        "--manifest",
        str(MANIFEST_PATH),
    ]
    subprocess.run([*command, "--output", str(first)], cwd=REPOSITORY_ROOT / "python", check=True)
    subprocess.run([*command, "--output", str(second)], cwd=REPOSITORY_ROOT / "python", check=True)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == CALIBRATION_PATH.read_bytes()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("model_hash", "model hash"),
        ("config_hash", "config hash"),
        ("order", "action order"),
        ("shape", "59"),
        ("bounds", "joint limit"),
    ),
)
def test_loader_rejects_stale_malformed_or_out_of_bounds_artifacts(
    mutation: str,
    message: str,
    tmp_path: Path,
    body: FlyBodyModel,
    config: KeyboardConfig,
) -> None:
    payload = json.loads(CALIBRATION_PATH.read_text())
    if mutation == "model_hash":
        payload["modelSha256"] = "0" * 64
    elif mutation == "config_hash":
        payload["configSha256"] = "0" * 64
    elif mutation == "order":
        payload["actionOrder"][0], payload["actionOrder"][1] = payload["actionOrder"][1], payload["actionOrder"][0]
    elif mutation == "shape":
        payload["neutralPose"].pop()
    elif mutation == "bounds":
        payload["neutralPose"][0] = float(body.upper_limits[0] + 1)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        load_calibration(path, body, config, MANIFEST_PATH, CONFIG_PATH)
