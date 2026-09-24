from __future__ import annotations

import pytest

from fly_crossy.env import GameState, GridPosition, Hazard, Lane, WORLD_VERSION, traffic_radar
from fly_crossy.schema import Action
from fly_crossy.traffic_teacher import action_teacher_targets


def make_state(fly_row: int, fly_column: float, lanes: list[Lane]) -> GameState:
    return GameState(
        version=WORLD_VERSION,
        seed="v10-traffic-fixture",
        step=0,
        time=0.0,
        fly=GridPosition(row=fly_row, column=fly_column),
        score=float(fly_row),
        lanes=lanes,
        terminal=None,
        previous_action=Action.WAIT,
    )


def test_radar_sees_car_outside_playable_width_before_it_arrives() -> None:
    state = make_state(10, 0, [
        Lane(row=10, kind="road", direction=-1, speed=5, phase=0,
             hazards=[Hazard(kind="car", position=8, size=2)]),
        *[Lane(row=row, kind="grass", hazards=[]) for row in range(11, 15)],
    ])
    radar = traffic_radar(state)
    assert radar[0][1] == pytest.approx(-1.0)
    assert 0.0 < radar[0][3] < 1.0


def test_teacher_rejects_forward_gap_that_closes_inside_horizon() -> None:
    state = make_state(10, 0, [
        Lane(row=10, kind="grass", hazards=[]),
        Lane(row=11, kind="road", direction=-1, speed=5, phase=0,
             hazards=[Hazard(kind="car", position=4, size=2)]),
        *[Lane(row=row, kind="grass", hazards=[]) for row in range(12, 15)],
    ])
    targets = {x.action: x for x in action_teacher_targets(state)}
    assert targets[Action.FORWARD].terminal is False
    assert targets[Action.FORWARD].short_horizon_risk is True


def test_teacher_marks_wait_dangerous_on_exposed_road_and_finds_lateral_escape() -> None:
    state = make_state(10, 0, [
        Lane(row=10, kind="road", direction=1, speed=2, phase=0,
             hazards=[Hazard(kind="car", position=-2.5, size=2)]),
        Lane(row=11, kind="road", direction=-1, speed=2, phase=0,
             hazards=[Hazard(kind="car", position=0, size=2)]),
        *[Lane(row=row, kind="grass", hazards=[]) for row in range(12, 15)],
    ])
    targets = {x.action: x for x in action_teacher_targets(state)}
    assert targets[Action.WAIT].short_horizon_risk is True
    assert any(not targets[a].short_horizon_risk for a in (Action.LEFT, Action.RIGHT))


def test_teacher_anticipates_train_before_immediate_collision() -> None:
    state = make_state(20, 0, [
        Lane(row=20, kind="grass", hazards=[]),
        Lane(row=21, kind="rail", direction=-1, speed=10, phase=0,
             hazards=[Hazard(kind="train", position=15, size=18)]),
        *[Lane(row=row, kind="grass", hazards=[]) for row in range(22, 25)],
    ])
    targets = {x.action: x for x in action_teacher_targets(state)}
    assert targets[Action.FORWARD].short_horizon_risk is True
