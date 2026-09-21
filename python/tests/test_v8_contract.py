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
    _scenery_obstacle_columns,
    create_game,
    generate_rows,
    observe,
    step_game,
)
from fly_crossy.schema import OBSERVATION_VERSION, Action, flatten_observation


ROOT = Path(__file__).resolve().parents[2]


def test_v8_preserves_v6_world_layout_fixture() -> None:
    assert WORLD_VERSION == 8
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


def test_observation_v2_exposes_blocker_and_stays_370_values() -> None:
    seed = "v8-blocker-contract"
    row = 3
    while not _scenery_obstacle_columns(seed, row, "grass"):
        row += 1
    blocker = _scenery_obstacle_columns(seed, row, "grass")[0]

    state = create_game(seed)
    lane = Lane(row=row, kind="grass", hazards=[])
    state = GameState(
        version=WORLD_VERSION,
        seed=seed,
        step=row,
        time=row * 0.2,
        fly=GridPosition(row=row, column=0),
        score=row,
        lanes=[lane],
        terminal=None,
        previous_action=Action.WAIT,
    )
    observation = observe(state)
    column_index = blocker + 5

    assert OBSERVATION_VERSION == 2
    assert observation.version == 2
    assert observation.cells[5][column_index] == 8
    encoded = flatten_observation(observation)
    assert encoded.shape == (370,)
    assert encoded[5 * 11 + column_index] == pytest.approx(1.0)


def test_reward_v4_exempts_safe_carry_but_keeps_other_costs() -> None:
    waiting = step_game(create_game("v8-wait-cost"), Action.WAIT)
    assert waiting.reward == pytest.approx(-0.11)

    state = create_game("v8-block-cost")
    edge = GameState(
        version=WORLD_VERSION,
        seed=state.seed,
        step=state.step,
        time=state.time,
        fly=GridPosition(row=0, column=5),
        score=0,
        lanes=state.lanes,
        terminal=None,
        previous_action=Action.WAIT,
    )
    blocked = step_game(edge, Action.RIGHT)
    assert blocked.state.terminal is None
    assert any(event.get("type") == "blocked" for event in blocked.events)
    assert blocked.reward == pytest.approx(-0.21)


    carried_state = GameState(
        version=WORLD_VERSION,
        seed="v8-safe-carry",
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

    unsafe_carry_state = GameState(
        version=WORLD_VERSION,
        seed="v8-unsafe-carry",
        step=0,
        time=0,
        fly=GridPosition(row=3, column=4.9),
        score=3,
        lanes=[
            Lane(
                row=3,
                kind="river",
                hazards=[Hazard(kind="log", position=4.4, size=2)],
                direction=1,
                speed=1,
                phase=0,
            )
        ],
        terminal=None,
        previous_action=Action.WAIT,
    )
    unsafe_carry = step_game(unsafe_carry_state, Action.WAIT)
    assert any(event.get("type") == "carried" for event in unsafe_carry.events)
    assert unsafe_carry.state.terminal == "bounds"
    assert unsafe_carry.reward == pytest.approx(-10.11)
