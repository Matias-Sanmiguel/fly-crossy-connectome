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


def test_obstacle_layout_is_deterministic_sparse_and_center_safe() -> None:
    assert _scenery_obstacle_columns('obstacle-test', 12, 'grass') == [-1]
    for row in range(-30, 81):
        columns = _scenery_obstacle_columns('obstacle-audit', row, 'grass')
        recovery = (row - 3) % 7 == 6
        opening = 0 <= row < 3
        if recovery or opening:
            assert columns == []
            continue
        assert 1 <= len(columns) <= 3
        assert 0 not in columns
        assert all(abs(column) <= 4 for column in columns)
        assert not _scenery_blocked('obstacle-audit', row, 'grass', 0)


def test_scenery_collision_blocks_without_terminal() -> None:
    state = _state_at('obstacle-test', 12, 0)
    assert next(lane for lane in state.lanes if lane.row == 12).kind == 'grass'
    result = step_game(state, Action.LEFT)
    assert result.state.fly == GridPosition(row=12, column=0)
    assert result.state.terminal is None
    assert any(event.get('type') == 'blocked' and event.get('reason') == 'scenery' for event in result.events)


def test_manual_world_edge_blocks_without_bounds_terminal() -> None:
    result = step_game(_state_at('boundary-test', 0, 5), Action.RIGHT)
    assert result.state.fly == GridPosition(row=0, column=5)
    assert result.state.terminal is None
    assert any(event.get('type') == 'blocked' and event.get('reason') == 'bounds' for event in result.events)


def test_observation_marks_static_obstacle_as_unavailable_cell() -> None:
    state = _state_at('obstacle-test', 12, 0)
    observation = observe(state)
    # center is index 5; obstacle at relative column -1 is index 4.
    assert observation.cells[5][4] == 0
    assert observation.motion[5][4] == [0, 0]
