from __future__ import annotations

import math
from pathlib import Path

import pytest

from fly_crossy.biomechanics.config import KEY_NAMES, KeyboardConfig
from fly_crossy.biomechanics.contact import (
    ContactGate,
    ContactGateBusy,
    ContactSample,
    KeyContact,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "biomechanics-v1.json"

EXPECTED_LEGS = {
    "W": "front_left",
    "A": "middle_left",
    "S": "front_right",
    "D": "middle_right",
    "SPACE_LEFT": "hind_left",
    "SPACE_RIGHT": "hind_right",
}


@pytest.fixture
def config() -> KeyboardConfig:
    return KeyboardConfig.load(CONFIG_PATH)


@pytest.fixture
def gate(config: KeyboardConfig) -> ContactGate:
    return ContactGate(config)


def frame(
    time: float,
    *contacts: tuple[str, float, float, tuple[str, ...]],
) -> ContactSample:
    overrides = {
        key: (travel, force, tarsi) for key, travel, force, tarsi in contacts
    }
    return ContactSample(
        time=time,
        contacts=tuple(
            KeyContact(
                key=key,
                travel=overrides.get(key, (0.0, 0.0, ()))[0],
                normal_force=overrides.get(key, (0.0, 0.0, ()))[1],
                touching_tarsi=overrides.get(key, (0.0, 0.0, ()))[2],
            )
            for key in KEY_NAMES
        ),
    )


def valid_contact(
    key: str,
    *,
    travel: float = 0.0004,
    force: float = 0.0002,
    leg: str | None = None,
) -> tuple[str, float, float, tuple[str, ...]]:
    return (key, travel, force, (leg or EXPECTED_LEGS[key],))


def test_press_needs_continuous_force_travel_correct_leg_and_debounce(
    gate: ContactGate,
) -> None:
    gate.begin("i-1", "W")

    assert gate.sample(frame(0.000, valid_contact("W"))).kind == "pending"
    assert gate.sample(frame(0.019, valid_contact("W"))).kind == "pending"
    outcome = gate.sample(frame(0.021, valid_contact("W")))

    assert outcome.kind == "confirmed"
    assert outcome.intention_id == "i-1"
    assert outcome.key == "W"
    assert outcome.is_directional_confirmation is True


@pytest.mark.parametrize(
    "invalid",
    (
        ("W", 0.0, 0.0002, ("front_left",)),
        ("W", 0.0004, 0.0, ("front_left",)),
        ("W", 0.0004, 0.0002, ("front_right",)),
    ),
    ids=("force-without-travel", "travel-without-force", "wrong-tarsus"),
)
def test_invalid_physical_evidence_resets_the_continuous_debounce(
    gate: ContactGate,
    invalid: tuple[str, float, float, tuple[str, ...]],
) -> None:
    gate.begin("i-2", "W")
    assert gate.sample(frame(0.000, valid_contact("W"))).kind == "pending"
    assert gate.sample(frame(0.019, invalid)).kind == "pending"
    assert gate.sample(frame(0.021, valid_contact("W"))).kind == "pending"

    assert gate.sample(frame(0.042, valid_contact("W"))).kind == "confirmed"


def test_wrong_key_is_a_terminal_failure_and_never_confirms_requested_key(
    gate: ContactGate,
) -> None:
    gate.begin("i-3", "W")

    outcome = gate.sample(frame(0.030, valid_contact("S")))

    assert outcome.kind == "wrong-key"
    assert outcome.intention_id == "i-3"
    assert outcome.key == "S"
    assert outcome.is_directional_confirmation is False
    assert gate.sample(frame(0.040, valid_contact("W"))) is None


def test_two_touched_keys_in_one_frame_are_deterministically_ambiguous(
    gate: ContactGate,
) -> None:
    gate.begin("i-4", "W")

    first_order = gate.sample(
        frame(0.000, valid_contact("W"), valid_contact("A"))
    )

    assert first_order.kind == "ambiguous"
    assert first_order.key is None
    assert gate.sample(
        frame(0.010, valid_contact("A"), valid_contact("W"))
    ) is None


def test_force_above_the_safe_maximum_is_terminal(gate: ContactGate) -> None:
    gate.begin("i-5", "W")

    outcome = gate.sample(
        frame(0.000, valid_contact("W", force=0.02001))
    )

    assert outcome.kind == "unsafe-force"
    assert outcome.key == "W"
    assert gate.sample(frame(0.030, valid_contact("W"))) is None


def test_timeout_is_terminal_and_cannot_later_confirm(gate: ContactGate) -> None:
    gate.begin("i-6", "W")
    assert gate.sample(frame(0.000)).kind == "pending"

    outcome = gate.sample(frame(1.500))

    assert outcome.kind == "timed-out"
    assert outcome.key is None
    assert gate.sample(frame(1.501, valid_contact("W"))) is None


def test_cancel_emits_one_terminal_outcome_and_is_idempotent(
    gate: ContactGate,
) -> None:
    gate.begin("i-7", "D")

    outcome = gate.cancel()

    assert outcome.kind == "cancelled"
    assert outcome.intention_id == "i-7"
    assert outcome.key is None
    assert gate.cancel() is None
    assert gate.sample(frame(0.010, valid_contact("D"))) is None
    with pytest.raises(ValueError, match="already completed"):
        gate.begin("i-7", "D")


def test_gate_rejects_a_second_live_intention(gate: ContactGate) -> None:
    gate.begin("i-8", "W")

    with pytest.raises(ContactGateBusy, match="i-8"):
        gate.begin("i-9", "A")


def test_confirmed_key_must_release_before_a_new_intention_can_rearm(
    gate: ContactGate,
) -> None:
    gate.begin("i-10", "W")
    assert gate.sample(frame(0.000, valid_contact("W"))).kind == "pending"
    assert gate.sample(frame(0.021, valid_contact("W"))).kind == "confirmed"

    gate.begin("i-11", "W")
    assert gate.sample(frame(0.030, valid_contact("W"))).kind == "pending"
    assert gate.sample(frame(0.040)).kind == "pending"
    assert gate.sample(frame(0.050, valid_contact("W"))).kind == "pending"
    assert gate.sample(frame(0.069, valid_contact("W"))).kind == "pending"
    assert gate.sample(frame(0.071, valid_contact("W"))).kind == "confirmed"


def test_release_can_be_observed_silently_after_the_terminal_event(
    gate: ContactGate,
) -> None:
    gate.begin("i-12", "A")
    assert gate.sample(frame(0.000, valid_contact("A"))).kind == "pending"
    assert gate.sample(frame(0.021, valid_contact("A"))).kind == "confirmed"
    assert gate.sample(frame(0.030)) is None

    gate.begin("i-13", "A")
    assert gate.sample(frame(0.040, valid_contact("A"))).kind == "pending"
    assert gate.sample(frame(0.061, valid_contact("A"))).kind == "confirmed"


@pytest.mark.parametrize(
    ("key", "leg"),
    tuple(EXPECTED_LEGS.items()),
)
def test_each_key_requires_its_mapped_leg(
    gate: ContactGate, key: str, leg: str
) -> None:
    gate.begin(f"i-map-{key.lower()}", key)
    assert gate.sample(frame(0.000, valid_contact(key, leg=leg))).kind == "pending"

    outcome = gate.sample(frame(0.021, valid_contact(key, leg=leg)))

    assert outcome.kind == "confirmed"
    assert outcome.key == key


@pytest.mark.parametrize(
    ("space_key", "wrong_leg"),
    (("SPACE_LEFT", "front_left"), ("SPACE_RIGHT", "middle_right")),
)
def test_space_zones_reject_non_hind_legs_and_never_report_directional_success(
    gate: ContactGate, space_key: str, wrong_leg: str
) -> None:
    gate.begin(f"i-space-wrong-{space_key.lower()}", space_key)
    assert gate.sample(
        frame(0.000, valid_contact(space_key, leg=wrong_leg))
    ).kind == "pending"
    assert gate.cancel().kind == "cancelled"

    gate.begin(f"i-space-right-{space_key.lower()}", space_key)
    assert gate.sample(frame(0.100)).kind == "pending"
    assert gate.sample(frame(0.110, valid_contact(space_key))).kind == "pending"
    outcome = gate.sample(frame(0.131, valid_contact(space_key)))

    assert outcome.kind == "confirmed"
    assert outcome.key == space_key
    assert outcome.is_directional_confirmation is False


def test_wrong_tarsus_cannot_satisfy_a_directional_key(gate: ContactGate) -> None:
    gate.begin("i-14", "W")

    assert gate.sample(
        frame(0.000, valid_contact("W", leg="hind_left"))
    ).kind == "pending"
    assert gate.sample(
        frame(0.100, valid_contact("W", leg="hind_left"))
    ).kind == "pending"


@pytest.mark.parametrize("bad_time", (math.nan, math.inf, -math.inf))
def test_sample_timestamp_must_be_finite(bad_time: float) -> None:
    with pytest.raises(ValueError, match="time must be finite"):
        frame(bad_time)


def test_sample_timestamps_must_be_monotonic(gate: ContactGate) -> None:
    gate.begin("i-15", "W")
    assert gate.sample(frame(0.100)).kind == "pending"

    with pytest.raises(ValueError, match="monotonic"):
        gate.sample(frame(0.099))


def test_frame_requires_one_state_for_every_key_and_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="exactly one state for every key"):
        ContactSample(
            time=0.0,
            contacts=(
                KeyContact(
                    key="W",
                    travel=0.0,
                    normal_force=0.0,
                    touching_tarsi=(),
                ),
            ),
        )

    with pytest.raises(ValueError, match="exactly one state for every key"):
        ContactSample(
            time=0.0,
            contacts=tuple(
                KeyContact(
                    key="W",
                    travel=0.0,
                    normal_force=0.0,
                    touching_tarsi=(),
                )
                for _ in KEY_NAMES
            ),
        )


@pytest.mark.parametrize(
    ("travel", "force"),
    ((math.nan, 0.0), (0.0, math.inf), (-0.0001, 0.0), (0.0, -0.0001)),
)
def test_contact_values_must_be_finite_nonnegative_si_values(
    travel: float, force: float
) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        KeyContact(
            key="W",
            travel=travel,
            normal_force=force,
            touching_tarsi=("front_left",),
        )
