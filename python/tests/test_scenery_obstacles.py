from __future__ import annotations

from fly_crossy.env import (
    GameState,
    GridPosition,
    WORLD_VERSION,
    _scenery_blocked,
    _scenery_obstacle_columns,
    generate_rows,
    observe,
    step_game,
)
from fly_crossy.schema import Action


def _state_at(seed: str, row: int, column: float) -> GameState:
    return GameState(
        version=WORLD_VERSION,
        seed=seed,
        step=row,
        time=row * 0.2,
        fly=GridPosition(row=row, column=column),
        score=row,
        lanes=generate_rows(seed, row - 8, 24),
        terminal=None,
        previous_action=Action.WAIT,
    )


def _blocking_scenario() -> tuple[str, int, int]:
    for seed_index in range(30):
        seed = f"collision-audit-{seed_index}"
        for row in range(3, 81):
            lane = generate_rows(seed, row, 1)[0]
            if lane.kind != "grass":
                continue
            columns = _scenery_obstacle_columns(seed, row, "grass")
            if columns:
                return seed, row, columns[0]
    raise AssertionError("expected a deterministic blocking scenery scenario")


def test_obstacle_layout_is_deterministic_sparse_and_can_block_center() -> None:
    saw_center = False
    for seed_index in range(12):
        seed = f"obstacle-audit-{seed_index}"
        for row in range(-30, 81):
            first = _scenery_obstacle_columns(seed, row, "grass")
            second = _scenery_obstacle_columns(seed, row, "grass")
            assert first == second

            recovery = (row - 3) % 7 == 6
            opening = 0 <= row < 3
            if recovery or opening:
                assert first == []
                continue

            assert 1 <= len(first) <= 3
            assert len(first) == len(set(first))
            assert all(abs(column) <= 4 for column in first)
            assert all(
                _scenery_blocked(seed, row, "grass", column)
                for column in first
            )
            saw_center = saw_center or 0 in first

    assert saw_center


def test_scenery_collision_blocks_without_terminal() -> None:
    seed, row, target = _blocking_scenario()
    start = target - 1
    state = _state_at(seed, row, start)
    assert next(lane for lane in state.lanes if lane.row == row).kind == "grass"

    result = step_game(state, Action.RIGHT)
    assert result.state.fly == GridPosition(row=row, column=start)
    assert result.state.terminal is None
    assert any(
        event.get("type") == "blocked" and event.get("reason") == "scenery"
        for event in result.events
    )


def test_manual_world_edge_blocks_without_bounds_terminal() -> None:
    result = step_game(_state_at("boundary-test", 0, 5), Action.RIGHT)
    assert result.state.fly == GridPosition(row=0, column=5)
    assert result.state.terminal is None
    assert any(
        event.get("type") == "blocked" and event.get("reason") == "bounds"
        for event in result.events
    )


def test_observation_marks_static_obstacle_as_unavailable_cell() -> None:
    scenario: tuple[str, int, int] | None = None
    for seed_index in range(30):
        seed = f"observation-audit-{seed_index}"
        for row in range(3, 81):
            lane = generate_rows(seed, row, 1)[0]
            if lane.kind != "grass":
                continue
            for column in _scenery_obstacle_columns(seed, row, "grass"):
                if column != 0:
                    scenario = (seed, row, column)
                    break
            if scenario is not None:
                break
        if scenario is not None:
            break

    assert scenario is not None
    seed, row, blocker = scenario
    observation = observe(_state_at(seed, row, 0))
    assert observation.cells[5][blocker + 5] == 0
    assert observation.motion[5][blocker + 5] == [0, 0]
