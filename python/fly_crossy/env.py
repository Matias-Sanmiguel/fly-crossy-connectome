from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Literal, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

from .schema import Action, OBSERVATION_RADIUS, ObservationV1, flatten_observation


WORLD_VERSION = 3
DECISION_SECONDS = 0.2
WORLD_HALF_WIDTH = 5
HAZARD_CIRCUIT = 25
TRAIN_CIRCUIT = 80

OPENING_ROWS = 3
CONTENT_ROWS_PER_GROUP = 6
GROUP_ROWS = CONTENT_ROWS_PER_GROUP + 1
SOLVABILITY_STEP_LIMIT = 80
SOLVABILITY_TRANSITION_LIMIT = 1_024
GENERATION_ATTEMPTS = 8
ROWS_AHEAD = 15
ROWS_BEHIND = 8
TRAIN_WARNING_SECONDS = 1.2

PROGRESS_REWARD = 1
TERMINAL_PENALTY = -10
STEP_COST = -0.01
STAGNATION_COST = -0.05

LaneKind = Literal["grass", "road", "rail", "river"]
SectionFamily = Literal["road", "rail", "river"]
HazardKind = Literal["car", "truck", "train", "log"]
TerminalReason = Literal["vehicle", "train", "water", "bounds"]


@dataclass(frozen=True, slots=True)
class Hazard:
    kind: HazardKind
    position: int
    size: int


@dataclass(frozen=True, slots=True)
class Lane:
    row: int
    kind: LaneKind
    hazards: list[Hazard]
    direction: int | None = None
    speed: int | None = None
    phase: float | None = None


@dataclass(frozen=True, slots=True)
class GridPosition:
    row: int
    column: float


@dataclass(frozen=True, slots=True)
class GameState:
    version: int
    seed: str
    step: int
    time: float
    fly: GridPosition
    score: float
    lanes: list[Lane]
    terminal: TerminalReason | None
    previous_action: Action


@dataclass(frozen=True, slots=True)
class StepResult:
    state: GameState
    reward: float
    events: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class _LaneTransition:
    fly: GridPosition
    time: float
    terminal: TerminalReason | None
    events: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class DifficultyProfile:
    level: int
    minimum_speed: int
    maximum_speed: int
    minimum_hazards: int
    maximum_hazards: int


_T = TypeVar("_T")
_UINT32_MASK = 0xFFFF_FFFF
_HAZARD_KINDS: dict[LaneKind, Sequence[HazardKind]] = {
    "road": ("car", "truck"),
    "rail": ("train",),
    "river": ("log",),
    "grass": (),
}
_SECTION_FAMILY_POOL: tuple[SectionFamily, ...] = (
    "road", "road", "road", "road", "road",
    "road", "road", "road", "road", "road",
    "rail", "rail", "rail", "rail", "rail", "rail",
    "rail", "rail", "rail", "rail", "rail", "rail",
    "river", "river", "river",
)
_SECTION_TEMPLATES: dict[
    SectionFamily, tuple[tuple[LaneKind, ...], tuple[LaneKind, ...]]
] = {
    "road": (
        ("road", "road", "road", "grass", "road", "road"),
        ("road", "road", "grass", "grass", "road", "road"),
    ),
    "rail": (
        ("rail", "grass", "road", "road", "road", "road"),
        ("rail", "grass", "grass", "road", "road", "road"),
    ),
    "river": (
        ("river", "river", "grass", "road", "road", "grass"),
        ("river", "river", "grass", "road", "road", "grass"),
    ),
}


def _uint32(value: int) -> int:
    return value & _UINT32_MASK


def _imul(left: int, right: int) -> int:
    return _uint32(left * right)


def _utf16_code_units(value: str) -> list[int]:
    encoded = value.encode("utf-16-le", errors="surrogatepass")
    return [
        int.from_bytes(encoded[index : index + 2], "little")
        for index in range(0, len(encoded), 2)
    ]


def hash_seed(seed: str) -> int:
    """Hash a JavaScript string exactly like the browser's 32-bit FNV-1a."""
    hashed = 0x811C_9DC5
    for code_unit in _utf16_code_units(seed):
        hashed ^= code_unit
        hashed = _imul(hashed, 0x0100_0193)
    return hashed


class _Rng:
    def __init__(self, seed: str) -> None:
        self._state = hash_seed(seed)

    def next(self) -> float:
        self._state = _uint32(self._state + 0x6D2B_79F5)
        value = self._state
        value = _imul(value ^ (value >> 15), value | 1)
        value = _uint32(
            value ^ _uint32(value + _imul(value ^ (value >> 7), value | 61))
        )
        return _uint32(value ^ (value >> 14)) / 0x1_0000_0000

    def integer(self, minimum: int, maximum: int) -> int:
        if type(minimum) is not int or type(maximum) is not int or maximum < minimum:
            raise ValueError("Expected an inclusive integer range.")
        return minimum + math.floor(self.next() * (maximum - minimum + 1))

    def pick(self, values: Sequence[_T]) -> _T:
        if not values:
            raise ValueError("Cannot pick from an empty list.")
        return values[math.floor(self.next() * len(values))]


def _group_index_for(row: int) -> int:
    return math.floor((row - OPENING_ROWS) / GROUP_ROWS)


def _seed_for_group(seed: str, group_index: int) -> str:
    return f"{WORLD_VERSION}:{seed}:{group_index}"


def _lane_template_for_group(seed: str, group_index: int) -> tuple[LaneKind, ...]:
    rng = _Rng(f"{_seed_for_group(seed, group_index)}:template")
    family = rng.pick(_SECTION_FAMILY_POOL)
    return _SECTION_TEMPLATES[family][rng.integer(0, 1)]


def difficulty_for_row(row: int) -> DifficultyProfile:
    """Return the versioned progression in fixed 50-row distance bands."""
    forward_distance = max(0, row - OPENING_ROWS)
    level = min(3, forward_distance // 50)
    return DifficultyProfile(
        level=level,
        minimum_speed=min(3, 1 + level // 2),
        maximum_speed=2 + level,
        minimum_hazards=2 + level // 2,
        maximum_hazards=3 + level,
    )


def _hazard_count(kind: LaneKind, rng: _Rng, difficulty: DifficultyProfile) -> int:
    if kind == "rail":
        return 1
    if kind == "river":
        return 5
    return rng.integer(difficulty.minimum_hazards, difficulty.maximum_hazards)


def _create_hazards(
    kind: LaneKind, rng: _Rng, difficulty: DifficultyProfile
) -> list[Hazard]:
    count = _hazard_count(kind, rng, difficulty)
    circuit = TRAIN_CIRCUIT if kind == "rail" else HAZARD_CIRCUIT
    half_circuit = circuit // 2
    first_position = rng.integer(-half_circuit, half_circuit)
    spacing = circuit // count
    hazards: list[Hazard] = []
    for index in range(count):
        hazard_kind = rng.pick(_HAZARD_KINDS[kind])
        unwrapped_position = first_position + index * spacing
        position = ((unwrapped_position + half_circuit) % circuit) - half_circuit
        if hazard_kind == "train":
            size = 18
        elif hazard_kind == "car":
            size = 2
        elif hazard_kind == "truck":
            size = 3
        else:
            size = rng.integer(2, 4)
        hazards.append(Hazard(kind=hazard_kind, position=position, size=size))
    return hazards


def _grass(row: int) -> Lane:
    return Lane(row=row, kind="grass", hazards=[])


def _hazard_lane(
    seed: str,
    row: int,
    group_index: int,
    attempt: int,
    kind: LaneKind,
) -> Lane:
    row_offset = row - (OPENING_ROWS + group_index * GROUP_ROWS)
    base_seed = _seed_for_group(seed, group_index)
    group_seed = base_seed if attempt == 0 else f"{base_seed}:retry:{attempt}"
    lane_rng = _Rng(f"{group_seed}:{row_offset}")
    difficulty = difficulty_for_row(row)
    return Lane(
        row=row,
        kind=kind,
        direction=lane_rng.pick((-1, 1)),
        speed=(
            lane_rng.integer(10, 12)
            if kind == "rail"
            else lane_rng.integer(difficulty.minimum_speed, difficulty.maximum_speed)
        ),
        phase=lane_rng.integer(0, 999) / 1000,
        hazards=_create_hazards(kind, lane_rng, difficulty),
    )


@lru_cache(maxsize=512)
def _generate_group_cached(seed: str, group_index: int) -> tuple[Lane, ...]:
    first = OPENING_ROWS + group_index * GROUP_ROWS
    lane_template = _lane_template_for_group(seed, group_index)
    for attempt in range(GENERATION_ATTEMPTS):
        content = [
            (
                _grass(first + offset)
                if lane_template[offset] == "grass"
                else _hazard_lane(
                    seed,
                    first + offset,
                    group_index,
                    attempt,
                    lane_template[offset],
                )
            )
            for offset in range(CONTENT_ROWS_PER_GROUP)
        ]
        if has_bounded_group_path(
            [_grass(first - 1), *content, _grass(first + CONTENT_ROWS_PER_GROUP)]
        ):
            return tuple(content)
    return tuple(_grass(first + offset) for offset in range(CONTENT_ROWS_PER_GROUP))


def _generate_group(seed: str, group_index: int) -> list[Lane]:
    return [
        Lane(
            row=lane.row,
            kind=lane.kind,
            hazards=list(lane.hazards),
            direction=lane.direction,
            speed=lane.speed,
            phase=lane.phase,
        )
        for lane in _generate_group_cached(seed, group_index)
    ]


def generate_rows(seed: str, start: int, count: int) -> list[Lane]:
    if type(start) is not int or type(count) is not int or count < 0:
        raise ValueError("Expected an integer starting row and nonnegative row count.")

    rows: list[Lane] = []
    groups: dict[int, list[Lane]] = {}
    for offset in range(count):
        row = start + offset
        if 0 <= row < OPENING_ROWS:
            rows.append(_grass(row))
            continue
        group_offset = row - (OPENING_ROWS + _group_index_for(row) * GROUP_ROWS)
        if group_offset == CONTENT_ROWS_PER_GROUP:
            rows.append(_grass(row))
            continue
        group_index = _group_index_for(row)
        if group_index not in groups:
            groups[group_index] = _generate_group(seed, group_index)
        rows.append(groups[group_index][group_offset])
    return rows


def _lane_for(state: GameState, row: int) -> Lane:
    lane = next((candidate for candidate in state.lanes if candidate.row == row), None)
    return lane if lane is not None else generate_rows(state.seed, row, 1)[0]


def _unwrapped_hazard_position_at(lane: Lane, hazard: Hazard, time: float) -> float:
    return hazard.position + (lane.direction or 0) * (lane.speed or 0) * (time + (lane.phase or 0))


def hazard_position_at(lane: Lane, hazard: Hazard, time: float) -> float:
    unwrapped = _unwrapped_hazard_position_at(lane, hazard, time)
    circuit = TRAIN_CIRCUIT if hazard.kind == "train" else HAZARD_CIRCUIT
    half_circuit = circuit / 2
    first_remainder = math.fmod(unwrapped + half_circuit, circuit)
    return math.fmod(first_remainder + circuit, circuit) - half_circuit


def hazard_contains(lane: Lane, hazard: Hazard, column: float, time: float) -> bool:
    return abs(hazard_position_at(lane, hazard, time) - column) <= hazard.size / 2


def _hazard_sweeps_column(
    lane: Lane,
    hazard: Hazard,
    column: float,
    from_time: float,
    to_time: float,
) -> bool:
    start = _unwrapped_hazard_position_at(lane, hazard, from_time)
    end = _unwrapped_hazard_position_at(lane, hazard, to_time)
    half_size = hazard.size / 2
    lower = min(start, end) - half_size
    upper = max(start, end) + half_size
    circuit = TRAIN_CIRCUIT if hazard.kind == "train" else HAZARD_CIRCUIT
    first_image = math.ceil((lower - column) / circuit)
    last_image = math.floor((upper - column) / circuit)
    return first_image <= last_image


def _moved_position(fly: GridPosition, action: Action) -> GridPosition:
    if action == Action.FORWARD:
        return GridPosition(row=fly.row + 1, column=fly.column)
    if action == Action.BACKWARD:
        return GridPosition(row=fly.row - 1, column=fly.column)
    if action == Action.LEFT:
        return GridPosition(row=fly.row, column=fly.column - 1)
    if action == Action.RIGHT:
        return GridPosition(row=fly.row, column=fly.column + 1)
    if action == Action.WAIT:
        return GridPosition(row=fly.row, column=fly.column)
    raise ValueError(f"Unknown game action: {action!s}")


def _collision_reason(lane: Lane, hazard: Hazard) -> TerminalReason:
    return "train" if lane.kind == "rail" or hazard.kind == "train" else "vehicle"


def _advance_lane_transition(
    position: GridPosition,
    time: float,
    action: Action,
    lane_for: Callable[[int], Lane],
) -> _LaneTransition:
    """Advance one action with the authoritative collision, carry, and bounds rules."""
    start_position = position
    to_time = time + DECISION_SECONDS
    fly = position
    events: list[dict[str, object]] = []
    starting_lane = lane_for(position.row)
    terminal: TerminalReason | None = None

    if starting_lane.kind in ("road", "rail"):
        collision = next(
            (
                hazard
                for hazard in starting_lane.hazards
                if _hazard_sweeps_column(
                    starting_lane, hazard, position.column, time, to_time
                )
            ),
            None,
        )
        if collision is not None:
            terminal = _collision_reason(starting_lane, collision)

    if terminal is None:
        fly = _moved_position(position, action)
        if action == Action.WAIT:
            events.append({"type": "waited", "position": fly})
        else:
            events.append(
                {"type": "moved", "action": action, "from": start_position, "to": fly}
            )

        destination_lane = lane_for(fly.row)
        if destination_lane.kind == "river" and action == Action.WAIT:
            support = next(
                (
                    hazard
                    for hazard in destination_lane.hazards
                    if hazard.kind == "log"
                    and hazard_contains(
                        destination_lane, hazard, position.column, time
                    )
                ),
                None,
            )
            if support is not None:
                displacement = (
                    (destination_lane.direction or 0)
                    * (destination_lane.speed or 0)
                    * DECISION_SECONDS
                )
                fly = GridPosition(row=fly.row, column=fly.column + displacement)
                events.append(
                    {"type": "carried", "row": fly.row, "displacement": displacement}
                )

        if abs(fly.column) > WORLD_HALF_WIDTH:
            terminal = "bounds"
        elif destination_lane.kind == "river":
            supported = any(
                hazard.kind == "log"
                and hazard_contains(destination_lane, hazard, fly.column, to_time)
                for hazard in destination_lane.hazards
            )
            if not supported:
                terminal = "water"
        elif destination_lane.kind in ("road", "rail"):
            collision = next(
                (
                    hazard
                    for hazard in destination_lane.hazards
                    if hazard_contains(destination_lane, hazard, fly.column, to_time)
                ),
                None,
            )
            if collision is not None:
                terminal = _collision_reason(destination_lane, collision)

        if terminal is None and destination_lane.kind == "rail":
            warning_end = to_time + TRAIN_WARNING_SECONDS
            approaching = any(
                hazard.kind == "train"
                and _hazard_sweeps_column(
                    destination_lane, hazard, fly.column, to_time, warning_end
                )
                for hazard in destination_lane.hazards
            )
            if approaching:
                events.append({"type": "train-warning", "row": destination_lane.row})

    if terminal is not None:
        events.append({"type": "terminal", "reason": terminal, "position": fly})
    return _LaneTransition(fly=fly, time=to_time, terminal=terminal, events=events)


def _decision_time_after_steps(steps: int) -> float:
    time = 0.0
    for _ in range(steps):
        time += DECISION_SECONDS
    return time


def find_bounded_group_witness(rows: Sequence[Lane]) -> list[Action] | None:
    """Find an action witness using the same float transition as ``step_game``."""
    if len(rows) != CONTENT_ROWS_PER_GROUP + 2:
        return None
    ordered = sorted(rows, key=lambda lane: lane.row)
    if (
        any(
            lane.row != ordered[index - 1].row + 1
            for index, lane in enumerate(ordered)
            if index > 0
        )
        or ordered[0].kind != "grass"
        or ordered[-1].kind != "grass"
    ):
        return None
    lanes = {lane.row: lane for lane in ordered}
    start_row = ordered[0].row
    goal_row = ordered[-1].row
    frontier: list[tuple[GridPosition, float, list[Action], int]] = [
        (
            GridPosition(row=start_row, column=0),
            _decision_time_after_steps(max(0, start_row)),
            [],
            0,
        )
    ]
    visited: set[tuple[int, int, float]] = set()
    actions = (
        Action.FORWARD,
        Action.LEFT,
        Action.RIGHT,
        Action.WAIT,
        Action.BACKWARD,
    )
    transitions_checked = 0
    next_order = 1

    while frontier:
        frontier.sort(key=lambda state: (-state[0].row, len(state[2]), state[3]))
        position, time, witness, _ = frontier.pop(0)
        depth = len(witness)
        if depth >= SOLVABILITY_STEP_LIMIT:
            continue
        for action in actions:
            if transitions_checked >= SOLVABILITY_TRANSITION_LIMIT:
                return None
            transitions_checked += 1
            attempted_row = position.row + (
                1
                if action == Action.FORWARD
                else -1
                if action == Action.BACKWARD
                else 0
            )
            if attempted_row < start_row or attempted_row > goal_row:
                continue
            transition = _advance_lane_transition(
                position, time, action, lambda row: lanes[row]
            )
            if transition.terminal is not None:
                continue
            candidate_witness = [*witness, action]
            if transition.fly.row == goal_row:
                return candidate_witness
            candidate = (
                transition.fly,
                transition.time,
                candidate_witness,
                next_order,
            )
            next_order += 1
            key = (depth + 1, transition.fly.row, transition.fly.column)
            if key not in visited:
                visited.add(key)
                frontier.append(candidate)
    return None


def has_bounded_group_path(rows: Sequence[Lane]) -> bool:
    """Return whether the authoritative transition yields a bounded witness."""
    return find_bounded_group_witness(rows) is not None


def _extend_world(state: GameState, fly_row: int) -> list[Lane]:
    minimum = fly_row - ROWS_BEHIND
    maximum = fly_row + ROWS_AHEAD
    retained = {lane.row: lane for lane in state.lanes if lane.row >= minimum}
    for lane in generate_rows(state.seed, minimum, maximum - minimum + 1):
        if lane.row not in retained:
            retained[lane.row] = lane
    return [retained[row] for row in sorted(retained)]


def create_game(seed: str) -> GameState:
    return GameState(
        version=WORLD_VERSION,
        seed=seed,
        step=0,
        time=0,
        fly=GridPosition(row=0, column=0),
        score=0,
        lanes=generate_rows(seed, -ROWS_BEHIND, ROWS_BEHIND + ROWS_AHEAD + 1),
        terminal=None,
        previous_action=Action.WAIT,
    )


def _reward(
    previous: GameState,
    next_state: GameState,
) -> float:
    progress = (
        max(
            0,
            next_state.score - previous.score,
        )
        * PROGRESS_REWARD
    )

    terminal = (
        TERMINAL_PENALTY
        if (
            previous.terminal is None
            and next_state.terminal is not None
        )
        else 0
    )

    stagnation = (
        STAGNATION_COST
        if next_state.score <= previous.score
        else 0
    )

    return (
        progress
        + terminal
        + STEP_COST
        + stagnation
    )


def step_game(state: GameState, action: Action) -> StepResult:
    if state.terminal is not None:
        return StepResult(state=state, reward=0, events=[])
    transition = _advance_lane_transition(
        state.fly, state.time, action, lambda row: _lane_for(state, row)
    )

    next_state = GameState(
        version=WORLD_VERSION,
        seed=state.seed,
        step=state.step + 1,
        time=transition.time,
        fly=transition.fly,
        score=max(state.score, transition.fly.row),
        lanes=_extend_world(state, transition.fly.row),
        terminal=transition.terminal,
        previous_action=action,
    )
    return StepResult(
        state=next_state,
        reward=_reward(state, next_state),
        events=transition.events,
    )


def _occupying_hazard(lane: Lane, column: float, time: float) -> Hazard | None:
    return next(
        (
            hazard
            for hazard in lane.hazards
            if hazard_contains(lane, hazard, column, time)
        ),
        None,
    )


def _occupied_encoding(hazard: Hazard) -> int:
    if hazard.kind == "train":
        return 6
    if hazard.kind == "log":
        return 7
    return 5


def _lane_motion(lane: Lane, hazard: Hazard | None) -> list[float]:
    if hazard is None:
        return [0, 0]
    return [lane.direction or 0, min(1, (lane.speed or 0) / 3)]


def _is_river_supported(state: GameState) -> bool:
    lane = next((candidate for candidate in state.lanes if candidate.row == state.fly.row), None)
    return lane is not None and lane.kind == "river" and any(
        hazard.kind == "log" and hazard_contains(lane, hazard, state.fly.column, state.time)
        for hazard in lane.hazards
    )


def _js_round(value: float) -> int:
    return math.floor(value + 0.5)


def observe(state: GameState) -> ObservationV1:
    size = OBSERVATION_RADIUS * 2 + 1
    anchor_column = _js_round(state.fly.column)
    lanes = {lane.row: lane for lane in state.lanes}
    cells = [[0 for _ in range(size)] for _ in range(size)]
    motion = [[[0, 0] for _ in range(size)] for _ in range(size)]
    lane_encoding: dict[LaneKind, int] = {"grass": 1, "road": 2, "rail": 3, "river": 4}

    for row_offset in range(-OBSERVATION_RADIUS, OBSERVATION_RADIUS + 1):
        lane = lanes.get(state.fly.row + row_offset)
        if lane is None:
            continue
        observation_row = row_offset + OBSERVATION_RADIUS
        for column_offset in range(-OBSERVATION_RADIUS, OBSERVATION_RADIUS + 1):
            column = anchor_column + column_offset
            if abs(column) > WORLD_HALF_WIDTH:
                continue
            observation_column = column_offset + OBSERVATION_RADIUS
            hazard = _occupying_hazard(lane, column, state.time)
            cells[observation_row][observation_column] = (
                _occupied_encoding(hazard) if hazard is not None else lane_encoding[lane.kind]
            )
            motion[observation_row][observation_column] = _lane_motion(lane, hazard)

    edge_distance = max(
        0,
        min(1, (WORLD_HALF_WIDTH - abs(state.fly.column)) / WORLD_HALF_WIDTH),
    )
    return ObservationV1(
        cells=cells,
        motion=motion,
        support=1 if _is_river_supported(state) else 0,
        previous_action=state.previous_action,
        edge_distance=edge_distance,
    )


class FlyCrossyEnv:
    state: GameState

    def reset(self, seed: str) -> tuple[NDArray[np.float32], dict[str, object]]:
        self.state = create_game(seed)
        return flatten_observation(observe(self.state)), {"score": 0, "seed": seed}

    def step(
        self,
        action: Action,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        if not hasattr(self, "state"):
            raise RuntimeError("Call reset(seed) before step(action).")
        result = step_game(self.state, action)
        self.state = result.state
        return (
            flatten_observation(observe(self.state)),
            result.reward,
            self.state.terminal is not None,
            False,
            {"score": self.state.score, "terminalReason": self.state.terminal},
        )
