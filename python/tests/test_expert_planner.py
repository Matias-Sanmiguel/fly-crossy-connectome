from __future__ import annotations

from fly_crossy.env import (
    GameState,
    GridPosition,
    Hazard,
    Lane,
    WORLD_VERSION,
    step_game,
)
from fly_crossy.expert_planner import plan_action
from fly_crossy.schema import Action


def make_state(*, row: int, column: float, lanes: list[Lane]) -> GameState:
    return GameState(
        version=WORLD_VERSION,
        seed="expert-planner-test",
        step=0,
        time=0.0,
        fly=GridPosition(row=row, column=column),
        score=float(row),
        lanes=lanes,
        terminal=None,
        previous_action=Action.WAIT,
    )


def test_moving_off_road_escapes_car_that_reaches_old_cell_later() -> None:
    state = make_state(
        row=10,
        column=0,
        lanes=[
            Lane(
                row=10,
                kind="road",
                direction=1,
                speed=10,
                phase=0,
                hazards=[Hazard(kind="car", position=-3, size=2)],
            ),
            Lane(row=11, kind="road", direction=1, speed=1, phase=0, hazards=[]),
        ],
    )
    result = step_game(state, Action.FORWARD)
    assert result.state.fly.row == 11
    assert result.state.terminal is None


def test_waiting_on_same_road_is_still_hit_by_swept_car() -> None:
    state = make_state(
        row=10,
        column=0,
        lanes=[
            Lane(
                row=10,
                kind="road",
                direction=1,
                speed=10,
                phase=0,
                hazards=[Hazard(kind="car", position=-3, size=2)],
            )
        ],
    )
    assert step_game(state, Action.WAIT).state.terminal == "vehicle"


def test_entering_road_is_unsafe_if_car_sweeps_destination_this_tick() -> None:
    state = make_state(
        row=10,
        column=0,
        lanes=[
            Lane(row=10, kind="grass", hazards=[]),
            Lane(
                row=11,
                kind="road",
                direction=1,
                speed=10,
                phase=0,
                hazards=[Hazard(kind="car", position=-3, size=2)],
            ),
        ],
    )
    result = step_game(state, Action.FORWARD)
    assert result.state.fly.row == 11
    assert result.state.terminal == "vehicle"


def test_planner_uses_lateral_setup_when_forward_is_deadly() -> None:
    state = make_state(
        row=10,
        column=0,
        lanes=[
            Lane(row=10, kind="grass", hazards=[]),
            Lane(
                row=11,
                kind="road",
                direction=-1,
                speed=1,
                phase=0,
                hazards=[Hazard(kind="car", position=1, size=2)],
            ),
            Lane(row=12, kind="grass", hazards=[]),
        ],
    )
    assert step_game(state, Action.FORWARD).state.terminal == "vehicle"
    decision = plan_action(state, depth=3)
    assert decision.action == Action.LEFT


def test_planner_avoids_immediate_train_death_when_safe_move_exists() -> None:
    state = make_state(
        row=20,
        column=0,
        lanes=[
            Lane(row=20, kind="grass", hazards=[]),
            Lane(
                row=21,
                kind="rail",
                direction=-1,
                speed=12,
                phase=0,
                hazards=[Hazard(kind="train", position=11, size=18)],
            ),
        ],
    )
    assert step_game(state, Action.FORWARD).state.terminal == "train"
    decision = plan_action(state, depth=4)
    assert step_game(state, decision.action).state.terminal is None
