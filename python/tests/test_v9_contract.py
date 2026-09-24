from __future__ import annotations

import json
from pathlib import Path

import pytest

from fly_crossy.env import (
    GameState,
    GridPosition,
    Hazard,
    Lane,
    WORLD_LAYOUT_VERSION,
    WORLD_VERSION,
    create_game,
    generate_rows,
    observe,
    step_game,
)
from fly_crossy.schema import (
    OBSERVATION_INPUT_SIZE,
    OBSERVATION_VERSION,
    Action,
    flatten_observation,
)


ROOT = Path(__file__).resolve().parents[2]


def test_v9_preserves_v6_world_layout_fixture() -> None:
    assert WORLD_VERSION == 9
    assert WORLD_LAYOUT_VERSION == 6

    fixture = json.loads(
        (ROOT / "tests/fixtures/world-generation-v6.json").read_text(
            encoding="utf-8"
        )
    )
    for case in fixture["cases"]:
        actual = generate_rows(case["seed"], case["from"], case["count"])
        assert [
            {
                key: value
                for key, value in {
                    "row": lane.row,
                    "kind": lane.kind,
                    "direction": lane.direction,
                    "speed": lane.speed,
                    "phase": lane.phase,
                    "hazards": [
                        {"kind": h.kind, "position": h.position, "size": h.size}
                        for h in lane.hazards
                    ],
                }.items()
                if value is not None
            }
            for lane in actual
        ] == case["rows"]


def test_observation_v3_has_signed_position_and_492_values() -> None:
    state = create_game("v9-position-contract")
    state = GameState(
        version=WORLD_VERSION,
        seed=state.seed,
        step=0,
        time=0,
        fly=GridPosition(row=0, column=3),
        score=0,
        lanes=state.lanes,
        terminal=None,
        previous_action=Action.WAIT,
    )
    observation = observe(state)
    encoded = flatten_observation(observation)

    assert OBSERVATION_VERSION == 3
    assert observation.version == 3
    assert OBSERVATION_INPUT_SIZE == 492
    assert observation.signed_column == pytest.approx(0.6)
    assert observation.edge_distance == pytest.approx(0.4)
    assert encoded.shape == (492,)
    assert encoded[-1] == pytest.approx(0.6)


def test_observation_v3_exposes_continuous_hazard_offset() -> None:
    state = GameState(
        version=WORLD_VERSION,
        seed="v9-hazard-offset",
        step=0,
        time=0,
        fly=GridPosition(row=2, column=0),
        score=2,
        lanes=[
            Lane(row=2, kind="grass", hazards=[]),
            Lane(
                row=3,
                kind="road",
                hazards=[Hazard(kind="car", position=-0.75, size=2)],
                direction=1,
                speed=1,
                phase=0,
            ),
        ],
        terminal=None,
        previous_action=Action.WAIT,
    )
    observation = observe(state)

    assert observation.cells[6][4] == 5
    assert observation.cells[6][5] == 5
    assert observation.hazard_offset[6][4] == pytest.approx(0.25)
    assert observation.hazard_offset[6][5] == pytest.approx(-0.75)
    assert observation.motion[6][5] == pytest.approx([1.0, 0.2])


def test_reward_v4_remains_unchanged_in_environment_v9() -> None:
    waiting = step_game(create_game("v9-wait-cost"), Action.WAIT)
    assert waiting.reward == pytest.approx(-0.11)

    carried_state = GameState(
        version=WORLD_VERSION,
        seed="v9-safe-carry",
        step=0,
        time=0,
        fly=GridPosition(row=3, column=0),
        score=3,
        lanes=[
            Lane(
                row=3,
                kind="river",
                hazards=[Hazard(kind="log", position=-0.5, size=2)],
                direction=1,
                speed=1,
                phase=0,
            )
        ],
        terminal=None,
        previous_action=Action.WAIT,
    )
    carried = step_game(carried_state, Action.WAIT)
    assert carried.state.terminal is None
    assert any(event.get("type") == "carried" for event in carried.events)
    assert carried.reward == pytest.approx(-0.01)
