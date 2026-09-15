from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
from pydantic import ValidationError

from fly_crossy.biomechanics.config import KeyboardConfig
from fly_crossy.biomechanics.keyboard import (
    KEY_NAMES,
    build_keyboard_mjcf,
    kilograms_to_model_mass,
    linear_damping_to_model,
    linear_stiffness_to_model,
    meters_to_model_length,
    newtons_to_model_force,
    resolve_key_ids,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "biomechanics-v1.json"


@pytest.fixture(scope="module")
def config() -> KeyboardConfig:
    return KeyboardConfig.load(CONFIG_PATH)


@pytest.fixture(scope="module")
def keyboard_model(config: KeyboardConfig) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(build_keyboard_mjcf(config))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    return model


def _config_document() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_keyboard_compiles_and_resolves_six_unique_physical_keys(
    config: KeyboardConfig, keyboard_model: mujoco.MjModel
) -> None:
    key_ids = resolve_key_ids(keyboard_model)

    assert tuple(key_ids) == KEY_NAMES
    assert len({ids.joint_id for ids in key_ids.values()}) == 6
    assert len({ids.geom_id for ids in key_ids.values()}) == 6
    assert len({ids.force_sensor_id for ids in key_ids.values()}) == 6
    assert len({ids.contact_sensor_id for ids in key_ids.values()}) == 6
    assert all(
        keyboard_model.sensor_type[ids.force_sensor_id]
        == mujoco.mjtSensor.mjSENS_FORCE
        for ids in key_ids.values()
    )
    assert all(
        keyboard_model.sensor_type[ids.contact_sensor_id]
        == mujoco.mjtSensor.mjSENS_TOUCH
        for ids in key_ids.values()
    )

    expected_range = (0.0, meters_to_model_length(config.key_travel_meters))
    for ids in key_ids.values():
        assert tuple(keyboard_model.jnt_range[ids.joint_id]) == pytest.approx(
            expected_range
        )
        body_id = int(keyboard_model.geom_bodyid[ids.geom_id])
        assert keyboard_model.body_mass[body_id] > 0
        assert np.all(np.isfinite(keyboard_model.body_inertia[body_id]))
        assert np.all(keyboard_model.body_inertia[body_id] > 0)


def test_public_si_values_are_explicitly_converted_at_the_mjspec_boundary(
    config: KeyboardConfig, keyboard_model: mujoco.MjModel
) -> None:
    assert meters_to_model_length(0.0005) == pytest.approx(0.5)
    assert kilograms_to_model_mass(0.00001) == pytest.approx(0.01)
    assert newtons_to_model_force(0.001) == pytest.approx(1000.0)
    assert linear_stiffness_to_model(2.0) == pytest.approx(2000.0)
    assert linear_damping_to_model(0.004) == pytest.approx(4.0)

    ids = resolve_key_ids(keyboard_model)["W"]
    dof_id = int(keyboard_model.jnt_dofadr[ids.joint_id])
    body_id = int(keyboard_model.geom_bodyid[ids.geom_id])
    assert keyboard_model.opt.timestep == pytest.approx(1 / config.physics_hz)
    assert tuple(keyboard_model.geom_size[ids.geom_id]) == pytest.approx(
        tuple(meters_to_model_length(value) for value in config.key_half_extents_meters)
    )
    assert keyboard_model.body_mass[body_id] == pytest.approx(
        kilograms_to_model_mass(config.key_mass_kilograms)
    )
    assert keyboard_model.jnt_stiffness[ids.joint_id] == pytest.approx(
        linear_stiffness_to_model(config.return_stiffness_newtons_per_meter)
    )
    assert keyboard_model.dof_damping[dof_id] == pytest.approx(
        linear_damping_to_model(config.return_damping_newton_seconds_per_meter)
    )


def test_a_real_mujoco_key_travels_and_returns_below_the_release_threshold(
    config: KeyboardConfig, keyboard_model: mujoco.MjModel
) -> None:
    key_ids = resolve_key_ids(keyboard_model)
    ids = key_ids["W"]
    data = mujoco.MjData(keyboard_model)
    body_id = int(keyboard_model.geom_bodyid[ids.geom_id])
    qpos_id = int(keyboard_model.jnt_qposadr[ids.joint_id])

    data.xfrc_applied[body_id, 2] = -newtons_to_model_force(0.001)
    for _ in range(round(0.1 * config.physics_hz)):
        mujoco.mj_step(keyboard_model, data)
    pressed_travel_meters = float(data.qpos[qpos_id]) / 1000.0

    data.xfrc_applied[:] = 0
    for _ in range(round(0.25 * config.physics_hz)):
        mujoco.mj_step(keyboard_model, data)
    released_travel_meters = float(data.qpos[qpos_id]) / 1000.0

    assert pressed_travel_meters >= config.minimum_travel_meters
    assert released_travel_meters < config.release_travel_meters
    assert all(
        float(data.qpos[keyboard_model.jnt_qposadr[other.joint_id]])
        < meters_to_model_length(config.release_travel_meters)
        for name, other in key_ids.items()
        if name != "W"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda raw: raw.update({"surprise": 1}), "Extra inputs"),
        (lambda raw: raw["keys"]["W"].update({"color": "red"}), "Extra inputs"),
        (lambda raw: raw["keys"].pop("W"), "keys"),
        (lambda raw: raw["keys"].update({"Q": {"center": [0.0, 0.0, 0.0]}}), "keys"),
        (
            lambda raw: raw["keys"]["W"].update(
                {"center": raw["keys"]["S"]["center"]}
            ),
            "unique",
        ),
        (
            lambda raw: raw.update(
                {"releaseTravelMeters": raw["minimumTravelMeters"]}
            ),
            "releaseTravelMeters",
        ),
        (
            lambda raw: raw.update(
                {"maximumForceNewtons": raw["minimumForceNewtons"]}
            ),
            "maximumForceNewtons",
        ),
        (lambda raw: raw.update({"minimumForceNewtons": math.nan}), "finite"),
        (
            lambda raw: raw.update({"returnDampingNewtonSecondsPerMeter": True}),
            "valid number",
        ),
        (lambda raw: raw.update({"physicsHz": 0}), "greater than 0"),
    ),
)
def test_keyboard_config_rejects_invalid_or_ambiguous_physics(
    mutate: object, message: str
) -> None:
    raw = _config_document()
    mutate(raw)

    with pytest.raises(ValidationError, match=message):
        KeyboardConfig.model_validate(raw)


def test_keyboard_config_rejects_distinct_centers_whose_caps_overlap() -> None:
    raw = _config_document()
    raw["keys"]["W"]["center"] = [0.0, 0.0, 0.0]
    raw["keys"]["A"]["center"] = [0.0004, 0.0004, 0.0]

    with pytest.raises(ValidationError, match="overlap"):
        KeyboardConfig.model_validate(raw)


def test_keyboard_config_allows_caps_to_touch_at_an_edge_with_float_tolerance() -> None:
    raw = _config_document()
    half_width = raw["keyHalfExtentsMeters"][0]
    spacing = 2 * half_width - 5e-13
    for index, name in enumerate(KEY_NAMES):
        raw["keys"][name]["center"] = [index * 0.0018, 0.0, 0.0]
    raw["keys"]["A"]["center"] = [spacing, 0.0, 0.0]

    config = KeyboardConfig.model_validate(raw)

    assert config.keys["A"].center[0] == pytest.approx(spacing)


@pytest.mark.parametrize("invalid_version", (True, 1.0))
def test_schema_version_requires_the_exact_integer_one(
    invalid_version: object,
) -> None:
    raw = _config_document()
    raw["schemaVersion"] = invalid_version

    with pytest.raises(ValidationError, match="schemaVersion"):
        KeyboardConfig.model_validate(raw)


def test_validated_key_mapping_and_nested_centers_are_deeply_immutable() -> None:
    raw = _config_document()
    config = KeyboardConfig.model_validate(raw)
    original_center = config.keys["W"].center

    with pytest.raises(TypeError):
        config.keys["W"] = config.keys["A"]
    with pytest.raises(ValidationError, match="frozen"):
        config.keys["W"].center = (9.0, 9.0, 9.0)
    with pytest.raises(TypeError):
        config.keys["W"].center[0] = 9.0

    raw["keys"]["W"]["center"][0] = 9.0
    assert config.keys["W"].center == original_center


def test_deeply_immutable_config_still_serializes_to_the_public_json_schema() -> None:
    raw = _config_document()
    config = KeyboardConfig.model_validate(raw)

    assert json.loads(config.model_dump_json(by_alias=True)) == raw
