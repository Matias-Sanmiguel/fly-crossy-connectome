from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from fly_crossy.biomechanics.motor import MotorIntention, MotorPhase
from fly_crossy.biomechanics.world import (
    BiomechanicalWorld,
    NATIVE_TIMESTEP_SECONDS,
    WorldActionResult,
    WorldSnapshot,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "data" / "manifests" / "flybody-v1.json"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "biomechanics-v1.json"
CALIBRATION_PATH = REPOSITORY_ROOT / "data" / "calibration" / "keyboard-reach-v1.json"


@pytest.fixture(scope="module")
def world() -> BiomechanicalWorld:
    return BiomechanicalWorld.load(MANIFEST_PATH, CONFIG_PATH, CALIBRATION_PATH)


def test_unified_world_contains_flybody_and_six_keyboard_keys(
    world: BiomechanicalWorld,
) -> None:
    assert world.model.nu == 59
    assert tuple(world.key_ids) == ("W", "A", "S", "D", "SPACE_LEFT", "SPACE_RIGHT")
    assert len(set(world.body.leg_sites.values())) == 6
    assert all(ids.geom_id >= 0 for ids in world.key_ids.values())
    assert all(ids.joint_id >= 0 for ids in world.key_ids.values())


def test_unified_world_preserves_native_timestep_and_scheduler(
    world: BiomechanicalWorld,
) -> None:
    assert world.model.opt.timestep == pytest.approx(NATIVE_TIMESTEP_SECONDS)
    assert world.native_substeps_per_tick == 20
    assert world.motor_outer_tick_stride == 5


def test_wait_completes_without_pressing_any_directional_key(
    world: BiomechanicalWorld,
) -> None:
    world.reset()
    result = world.run(MotorIntention("i-00000001", "wait"), limit_seconds=0.2)

    assert result.outcome == "waited"
    assert result.action == "wait"
    assert result.requested_action == "wait"
    assert world.motor_phase is MotorPhase.NEUTRAL


@pytest.mark.parametrize(
    ("action", "key"),
    (
        ("forward", "W"),
        ("backward", "S"),
        ("left", "A"),
        ("right", "D"),
    ),
)
def test_direction_is_emitted_only_after_real_key_contact(
    world: BiomechanicalWorld,
    action: str,
    key: str,
) -> None:
    world.reset()
    result = world.run(MotorIntention(f"i-{action:0<8}", action), limit_seconds=1.5)

    assert result.outcome == "confirmed"
    assert result.action == action
    assert result.requested_action == action
    assert world.snapshot().key_travel[key] >= world.config.release_travel_meters


def test_disabling_contact_evidence_can_never_emit_requested_direction() -> None:
    world = BiomechanicalWorld.load(
        MANIFEST_PATH,
        CONFIG_PATH,
        CALIBRATION_PATH,
        contacts_enabled=False,
    )

    result = world.run(MotorIntention("i-disabled1", "forward"), limit_seconds=1.6)

    assert result.outcome == "failed"
    assert result.action == "wait"
    assert result.requested_action == "forward"


def test_one_intention_emits_exactly_one_terminal_result(
    world: BiomechanicalWorld,
) -> None:
    world.reset()
    world.start_intention(MotorIntention("i-single01", "forward"))
    terminals: list[WorldActionResult] = []

    for _ in range(round(2.0 * world.config.physics_hz)):
        for event in world.step():
            if isinstance(event, WorldActionResult):
                terminals.append(event)
        if terminals and world.ready:
            break

    assert len(terminals) == 1


def test_snapshots_are_capped_below_or_equal_to_30_hz(
    world: BiomechanicalWorld,
) -> None:
    world.reset()
    snapshots: list[WorldSnapshot] = []
    seconds = 1.0

    for _ in range(round(seconds * world.config.physics_hz)):
        snapshots.extend(
            event for event in world.step() if isinstance(event, WorldSnapshot)
        )

    assert len(snapshots) <= 30
    assert all(
        later.simulation_time > earlier.simulation_time
        for earlier, later in zip(snapshots, snapshots[1:], strict=False)
    )


def test_all_unified_ids_remain_resolvable_after_compilation(
    world: BiomechanicalWorld,
) -> None:
    for actuator_name in world.body.joint_names:
        assert mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        ) >= 0
