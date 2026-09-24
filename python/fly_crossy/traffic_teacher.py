from __future__ import annotations

from dataclasses import dataclass

from .env import GameState, WORLD_HALF_WIDTH, _hazard_sweeps_column, _lane_for, step_game
from .schema import ACTION_ORDER, Action

TEACHER_RISK_HORIZON_SECONDS = 0.8

@dataclass(frozen=True, slots=True)
class ActionTeacherTarget:
    action: Action
    terminal: bool
    blocked: bool
    short_horizon_risk: bool
    opens_safe_forward: bool


def _blocked(result) -> bool:
    return any(event.get("type") == "blocked" for event in result.events)


def _position_exposed(state: GameState, horizon_seconds: float) -> bool:
    lane = _lane_for(state, state.fly.row)
    if lane.kind not in ("road", "rail"):
        return False
    return any(
        _hazard_sweeps_column(lane, hazard, state.fly.column, state.time, state.time + horizon_seconds)
        for hazard in lane.hazards
        if hazard.kind in ("car", "truck", "train")
    )


def _safe_forward_available(state: GameState, horizon_seconds: float) -> bool:
    if state.terminal is not None:
        return False
    result = step_game(state, Action.FORWARD)
    if result.state.terminal is not None or _blocked(result):
        return False
    return not _position_exposed(result.state, horizon_seconds)


def action_teacher_targets(state: GameState, *, horizon_seconds: float = TEACHER_RISK_HORIZON_SECONDS) -> tuple[ActionTeacherTarget, ...]:
    """Training/validation labels only; never available to inference."""
    targets: list[ActionTeacherTarget] = []
    for action in ACTION_ORDER:
        result = step_game(state, action)
        terminal = result.state.terminal is not None
        blocked = _blocked(result)
        risk = terminal or blocked or abs(result.state.fly.column) > WORLD_HALF_WIDTH or _position_exposed(result.state, horizon_seconds)
        targets.append(ActionTeacherTarget(
            action=action,
            terminal=terminal,
            blocked=blocked,
            short_horizon_risk=risk,
            opens_safe_forward=False if terminal or blocked else _safe_forward_available(result.state, horizon_seconds),
        ))
    return tuple(targets)
