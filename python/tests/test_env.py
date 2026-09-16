from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fly_crossy.env import (
    FlyCrossyEnv,
    GameState,
    GridPosition,
    Hazard,
    Lane,
    WORLD_VERSION,
    create_game,
    difficulty_for_row,
    find_bounded_group_witness,
    generate_rows,
    has_bounded_group_path,
    hash_seed,
    hazard_position_at,
    observe,
    step_game,
)
from fly_crossy.schema import (
    OBSERVATION_INPUT_SIZE,
    Action,
    flatten_observation,
)


FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "episode-v1.json"
PARITY_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "environment-parity-v1.json"
)
PARITY = json.loads(PARITY_FIXTURE.read_text(encoding="utf-8"))
WORLD_PARITY_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "world-generation-v6.json"
)


def _lane_from(value: dict[str, Any]) -> Lane:
    return Lane(
        row=value["row"],
        kind=value["kind"],
        hazards=[Hazard(**hazard) for hazard in value["hazards"]],
        direction=value.get("direction"),
        speed=value.get("speed"),
        phase=value.get("phase"),
    )


def _state_from(specification: dict[str, Any]) -> GameState:
    if specification["base"] == "createGame":
        state = create_game(specification["seed"])
        overrides = specification["overrides"]
        return replace(
            state,
            step=overrides.get("step", state.step),
            time=overrides.get("time", state.time),
            fly=GridPosition(**overrides.get("fly", asdict(state.fly))),
            score=overrides.get("score", state.score),
        )

    value = specification["value"]
    return GameState(
        version=value["version"],
        seed=value["seed"],
        step=value["step"],
        time=value["time"],
        fly=GridPosition(**value["fly"]),
        score=value["score"],
        lanes=[_lane_from(lane) for lane in value["lanes"]],
        terminal=value["terminal"],
        previous_action=Action(value["previousAction"]),
    )


def _observation_sha256(observation: np.ndarray) -> str:
    little_endian = observation.astype("<f4", copy=False)
    return hashlib.sha256(little_endian.tobytes()).hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def test_hash_and_row_generation_match_browser_contract() -> None:
    assert hash_seed("") == 2_166_136_261
    assert hash_seed("episode-v1-safe-opening") == 4_109_462_559
    assert hash_seed("parity-🪰") == 3_375_820_797
    assert WORLD_VERSION == 6

    rows = generate_rows("parity-seed", 3, 2)
    assert [lane.row for lane in rows] == [3, 4]
    assert rows == generate_rows("parity-seed", 3, 2)


def test_same_seed_and_range_produce_identical_lanes() -> None:
    assert generate_rows("lab-7", -5, 40) == generate_rows("lab-7", -5, 40)


def test_chunk_boundaries_do_not_change_generation() -> None:
    assert generate_rows("lab-7", 0, 42) == [
        *generate_rows("lab-7", 0, 17),
        *generate_rows("lab-7", 17, 25),
    ]


def test_section_templates_enforce_grammar_lengths_frequencies_and_recovery() -> None:
    approved_templates = {
        ("road", "road", "road", "grass", "road", "road"),
        ("road", "road", "grass", "grass", "road", "road"),
        ("rail", "grass", "road", "road", "road", "road"),
        ("rail", "grass", "grass", "road", "road", "road"),
        ("river", "river", "grass", "road", "road", "grass"),
    }
    row_counts = {"grass": 0, "road": 0, "rail": 0, "river": 0}
    section_counts = {"road": 0, "rail": 0, "river": 0}
    section_lengths: dict[str, list[int]] = {"road": [], "rail": [], "river": []}
    direct_unlike_transitions = 0
    fallback_groups = 0
    group_count = 0

    for seed_index in range(30):
        rows = generate_rows(f"section-audit-{seed_index}", 3, 210)
        section_kind: str | None = None
        section_length = 0
        for lane in rows:
            row_counts[lane.kind] += 1
            if lane.kind == "grass":
                if section_kind is not None:
                    section_lengths[section_kind].append(section_length)
                section_kind = None
                section_length = 0
            elif lane.kind == section_kind:
                section_length += 1
            else:
                if section_kind is not None:
                    direct_unlike_transitions += 1
                    section_lengths[section_kind].append(section_length)
                section_kind = lane.kind
                section_length = 1
                section_counts[lane.kind] += 1
        if section_kind is not None:
            section_lengths[section_kind].append(section_length)

        for offset in range(0, len(rows), 7):
            group = rows[offset : offset + 7]
            group_count += 1
            assert len(group) == 7
            assert group[6].kind == "grass"
            content_kinds = tuple(lane.kind for lane in group[:6])
            if all(kind == "grass" for kind in content_kinds):
                fallback_groups += 1
            else:
                assert content_kinds in approved_templates

    total_sections = sum(section_counts.values())
    total_rows = sum(row_counts.values())
    section_shares = {
        kind: count / total_sections for kind, count in section_counts.items()
    }
    row_shares = {kind: count / total_rows for kind, count in row_counts.items()}
    assert direct_unlike_transitions == 0
    assert all(2 <= length <= 4 for length in section_lengths["road"])
    assert all(length == 1 for length in section_lengths["rail"])
    assert all(length == 2 for length in section_lengths["river"])
    assert 0.65 <= section_shares["road"] <= 0.75, section_shares
    assert 0.19 <= section_shares["rail"] <= 0.29, section_shares
    assert 0.03 <= section_shares["river"] <= 0.09, section_shares
    assert section_shares["river"] < section_shares["rail"]
    assert 0.48 <= row_shares["road"] <= 0.58, row_shares
    assert 0.31 <= row_shares["grass"] <= 0.41, row_shares
    assert 0.04 <= row_shares["rail"] <= 0.09, row_shares
    assert 0.025 <= row_shares["river"] <= 0.065, row_shares
    assert row_shares["river"] < row_shares["rail"]
    assert fallback_groups / group_count <= 0.03


def test_row_generation_rejects_booleans_as_browser_non_numbers() -> None:
    with pytest.raises(ValueError, match="integer starting row"):
        generate_rows("parity-seed", True, 1)
    with pytest.raises(ValueError, match="nonnegative row count"):
        generate_rows("parity-seed", 0, False)


def test_known_impassable_first_group_is_replaced_by_a_bounded_reachable_group() -> None:
    group = generate_rows("solvability-probe-87", 2, 8)

    assert has_bounded_group_path(group, "solvability-probe-87") is True
    assert [group[0].kind, group[-1].kind] == ["grass", "grass"]


def _replay_group_witness(seed: str, rows: list[Lane], actions: list[Action]) -> None:
    start_row = rows[0].row
    time = 0.0
    for _ in range(max(0, start_row)):
        time += 0.2
    state = GameState(
        version=WORLD_VERSION,
        seed=seed,
        step=max(0, start_row),
        time=time,
        fly=GridPosition(row=start_row, column=0),
        score=max(0, start_row),
        lanes=rows,
        terminal=None,
        previous_action=Action.WAIT,
    )
    for action in actions:
        state = step_game(state, action).state
        assert state.terminal is None, (seed, action, state.step)
    assert state.fly.row == rows[-1].row


def test_audit_replay_21_solver_witness_survives_authoritative_float_replay() -> None:
    group = generate_rows("audit-replay-21", 2, 8)
    witness = find_bounded_group_witness(group, "audit-replay-21")

    assert witness is not None
    _replay_group_witness("audit-replay-21", group, witness)


def test_solver_witnesses_replay_through_authoritative_transition_sample() -> None:
    for seed_index in range(24):
        for group_index in (0, 1, 8, 24):
            seed = f"audit-replay-{seed_index}"
            start = 2 + group_index * 7
            group = generate_rows(seed, start, 8)
            witness = find_bounded_group_witness(group, seed)
            assert witness is not None, (seed, group_index)
            _replay_group_witness(seed, group, witness)


def test_generated_groups_have_a_bounded_route_over_a_deterministic_seed_sample() -> None:
    for seed_index in range(24):
        for group_index in (0, 1, 8, 24):
            start = 2 + group_index * 7
            assert has_bounded_group_path(
                generate_rows(f"solvability-property-{seed_index}", start, 8)
            ), (seed_index, group_index)


def test_documented_difficulty_parameters_increase_with_forward_distance() -> None:
    opening = difficulty_for_row(3)
    middle = difficulty_for_row(53)
    far = difficulty_for_row(153)

    assert middle.minimum_speed >= opening.minimum_speed
    assert middle.maximum_speed > opening.maximum_speed
    assert middle.minimum_hazards >= opening.minimum_hazards
    assert far.maximum_speed > middle.maximum_speed
    assert far.maximum_hazards > middle.maximum_hazards
    assert difficulty_for_row(52).level == 0
    assert difficulty_for_row(53).level == 1
    assert difficulty_for_row(102).level == 1
    assert difficulty_for_row(103).level == 2
    assert difficulty_for_row(152).level == 2
    assert difficulty_for_row(153).level == 3


def test_road_traffic_uses_canonical_sizes_without_circular_overlap() -> None:
    road_lanes = [
        lane
        for lane in generate_rows("experiment-001:auto:e323bb69", -5, 25)
        if lane.kind == "road"
    ]

    assert road_lanes
    for lane in road_lanes:
        for hazard in lane.hazards:
            assert hazard.size == (2 if hazard.kind == "car" else 3)
        for left_index, left in enumerate(lane.hazards):
            for right in lane.hazards[left_index + 1 :]:
                center_distance = abs(left.position - right.position)
                wrapped_distance = min(center_distance, 25 - center_distance)
                gap = wrapped_distance - (left.size + right.size) / 2
                assert gap >= 0.5, (lane.row, left, right, gap)


def test_river_sections_use_the_approved_candidate_b_log_profile() -> None:
    river_lanes = [
        lane
        for seed_index in range(30)
        for lane in generate_rows(f"section-audit-{seed_index}", 3, 210)
        if lane.kind == "river"
    ]

    assert river_lanes
    for lane in river_lanes:
        assert len(lane.hazards) == 5
        assert all(
            hazard.kind == "log"
            and type(hazard.size) is int
            and 2 <= hazard.size <= 4
            for hazard in lane.hazards
        )


def test_world_generation_matches_shared_cross_language_fixture() -> None:
    fixture = json.loads(WORLD_PARITY_FIXTURE.read_text(encoding="utf-8"))

    assert fixture["environmentVersion"] == WORLD_VERSION
    for parity_case in fixture["cases"]:
        lanes = [
            {key: value for key, value in asdict(lane).items() if value is not None}
            for lane in generate_rows(
                parity_case["seed"], parity_case["from"], parity_case["count"]
            )
        ]
        assert lanes == parity_case["rows"]


def test_wrapped_hazard_positions_match_authoritative_parity_fixture() -> None:
    assert PARITY["version"] == 1
    assert PARITY["observationEncoding"] == "float32-le-sha256"
    for check in PARITY["positionChecks"]:
        lane = generate_rows(check["seed"], check["row"], 1)[0]
        actual = hazard_position_at(
            lane,
            lane.hazards[check["hazardIndex"]],
            check["time"],
        )
        assert actual == check["expected"]


@pytest.mark.parametrize(
    "parity_case",
    PARITY["cases"],
    ids=[parity_case["name"] for parity_case in PARITY["cases"]],
)
def test_environment_transition_matches_authoritative_parity_fixture(
    parity_case: dict[str, Any],
) -> None:
    initial = _state_from(parity_case["state"])
    action = Action(parity_case["action"])
    expected = parity_case["expected"]
    direct_result = step_game(initial, action)
    env = FlyCrossyEnv()
    env.state = initial

    observation, reward, terminated, truncated, info = env.step(action)

    assert _observation_sha256(flatten_observation(observe(initial))) == expected[
        "beforeObservationSha256"
    ]
    assert _observation_sha256(observation) == expected["afterObservationSha256"]
    assert reward == pytest.approx(expected["reward"])
    assert asdict(env.state.fly) == expected["fly"]
    assert env.state.step == expected["step"]
    assert env.state.time == expected["time"]
    assert env.state.score == expected["score"]
    assert env.state.terminal == expected["terminalReason"]
    assert env.state.previous_action.value == expected["previousAction"]
    assert terminated is (expected["terminalReason"] is not None)
    assert truncated is False
    assert info == {
        "score": expected["score"],
        "terminalReason": expected["terminalReason"],
    }
    assert _json_value(direct_result.events) == expected["events"]
    assert [lane.row for lane in env.state.lanes] == expected["laneRows"]


def test_fixture_episode_matches_browser_contract() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    env = FlyCrossyEnv()

    observation, info = env.reset(seed=fixture["seed"])

    assert fixture["version"] == 1
    assert observation.shape == (OBSERVATION_INPUT_SIZE,)
    assert observation.dtype == np.float32
    assert info == {"score": 0, "seed": fixture["seed"]}

    rewards: list[float] = []
    for expected, action in zip(fixture["steps"], fixture["actions"], strict=True):
        observation, reward, terminated, truncated, info = env.step(Action(action))
        rewards.append(reward)
        assert observation.shape == (OBSERVATION_INPUT_SIZE,)
        assert reward == pytest.approx(expected["reward"])
        assert info["score"] == expected["score"]
        assert info["terminalReason"] == expected["terminalReason"]
        assert terminated is (expected["terminalReason"] is not None)
        assert truncated is False

    assert rewards == pytest.approx(fixture["rewards"])
    assert info["score"] == fixture["expectedScore"]
    assert info["terminalReason"] == fixture["terminalReason"]


def test_reset_is_deterministic_and_terminal_states_do_not_advance() -> None:
    env = FlyCrossyEnv()
    first, _ = env.reset(seed="episode-v1-safe-opening")
    env.step(Action.LEFT)
    second, _ = env.reset(seed="episode-v1-safe-opening")

    np.testing.assert_array_equal(second, first)

    env.state = GameState(
        version=WORLD_VERSION,
        seed="terminal-repeat",
        step=0,
        time=0,
        fly=GridPosition(row=3, column=0),
        score=3,
        lanes=[
            Lane(
                row=3,
                kind="road",
                hazards=[Hazard(kind="car", position=-1.5, size=1)],
                direction=1,
                speed=10,
                phase=0,
            )
        ],
        terminal=None,
        previous_action=Action.WAIT,
    )
    _, reward, terminated, truncated, info = env.step(Action.FORWARD)
    terminal_step = env.state.step
    _, repeated_reward, repeated_terminated, repeated_truncated, repeated_info = env.step(Action.WAIT)

    assert reward == pytest.approx(-10.06)
    assert terminated is True
    assert truncated is False
    assert info == {"score": 3, "terminalReason": "vehicle"}
    assert env.state.step == terminal_step
    assert repeated_reward == 0.0
    assert repeated_terminated is True
    assert repeated_truncated is False
    assert repeated_info == info
