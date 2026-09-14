import { reward as calculateReward } from './reward.ts';
import type { Action, Hazard, Lane } from './types.ts';
import { generateRows } from './world.ts';

export type TerminalReason = 'vehicle' | 'train' | 'water' | 'bounds';

export type GridPosition = {
  row: number;
  column: number;
};

export type GameEvent =
  | { type: 'moved'; action: Exclude<Action, 'wait'>; from: GridPosition; to: GridPosition }
  | { type: 'waited'; position: GridPosition }
  | { type: 'carried'; row: number; displacement: number }
  | { type: 'train-warning'; row: number }
  | { type: 'terminal'; reason: TerminalReason; position: GridPosition };

export type GameState = {
  version: 1;
  seed: string;
  step: number;
  time: number;
  fly: GridPosition;
  score: number;
  lanes: Lane[];
  terminal: TerminalReason | null;
  previousAction: Action;
};

export type StepResult = {
  state: GameState;
  reward: number;
  events: GameEvent[];
};

export const DECISION_SECONDS = 0.2;
export const WORLD_HALF_WIDTH = 5;
export const HAZARD_CIRCUIT = 25;

const ROWS_AHEAD = 15;
const ROWS_BEHIND = 8;
const TRAIN_WARNING_SECONDS = 2;

function laneFor(state: GameState, row: number): Lane {
  return state.lanes.find((lane) => lane.row === row) ?? generateRows(state.seed, row, 1)[0]!;
}

function unwrappedHazardPositionAt(lane: Lane, hazard: Hazard, time: number): number {
  const direction = lane.direction ?? 0;
  const speed = lane.speed ?? 0;
  const phase = lane.phase ?? 0;
  return hazard.position + direction * speed * (time + phase);
}

/** Horizontal center on the repeating hazard circuit at an authoritative time. */
export function hazardPositionAt(lane: Lane, hazard: Hazard, time: number): number {
  const unwrapped = unwrappedHazardPositionAt(lane, hazard, time);
  const halfCircuit = HAZARD_CIRCUIT / 2;
  return ((unwrapped + halfCircuit) % HAZARD_CIRCUIT + HAZARD_CIRCUIT) % HAZARD_CIRCUIT
    - halfCircuit;
}

export function hazardContains(lane: Lane, hazard: Hazard, column: number, time: number): boolean {
  return Math.abs(hazardPositionAt(lane, hazard, time) - column) <= hazard.size / 2;
}

function hazardSweepsColumn(
  lane: Lane,
  hazard: Hazard,
  column: number,
  fromTime: number,
  toTime: number,
): boolean {
  const from = unwrappedHazardPositionAt(lane, hazard, fromTime);
  const to = unwrappedHazardPositionAt(lane, hazard, toTime);
  const halfSize = hazard.size / 2;
  const lower = Math.min(from, to) - halfSize;
  const upper = Math.max(from, to) + halfSize;
  const firstImage = Math.ceil((lower - column) / HAZARD_CIRCUIT);
  const lastImage = Math.floor((upper - column) / HAZARD_CIRCUIT);
  return firstImage <= lastImage;
}

function movedPosition(fly: GridPosition, action: Action): GridPosition {
  switch (action) {
    case 'forward': return { row: fly.row + 1, column: fly.column };
    case 'backward': return { row: fly.row - 1, column: fly.column };
    case 'left': return { row: fly.row, column: fly.column - 1 };
    case 'right': return { row: fly.row, column: fly.column + 1 };
    case 'wait': return { ...fly };
    default: throw Error(`Unknown game action: ${String(action)}`);
  }
}

function extendWorld(state: GameState, flyRow: number): Lane[] {
  const minimum = flyRow - ROWS_BEHIND;
  const maximum = flyRow + ROWS_AHEAD;
  const retained = new Map(
    state.lanes
      .filter((lane) => lane.row >= minimum)
      .map((lane) => [lane.row, lane]),
  );

  for (let row = minimum; row <= maximum; row += 1) {
    if (!retained.has(row)) retained.set(row, generateRows(state.seed, row, 1)[0]!);
  }

  return [...retained.values()].sort((left, right) => left.row - right.row);
}

function terminalEvent(reason: TerminalReason, position: GridPosition): GameEvent {
  return { type: 'terminal', reason, position: { ...position } };
}

export function createGame(seed: string): GameState {
  return {
    version: 1,
    seed,
    step: 0,
    time: 0,
    fly: { row: 0, column: 0 },
    score: 0,
    lanes: generateRows(seed, -ROWS_BEHIND, ROWS_BEHIND + ROWS_AHEAD + 1),
    terminal: null,
    previousAction: 'wait',
  };
}

export function stepGame(state: GameState, action: Action): StepResult {
  if (state.terminal !== null) return { state, reward: 0, events: [] };

  const from = { ...state.fly };
  const toTime = state.time + DECISION_SECONDS;
  let fly = movedPosition(state.fly, action);
  const events: GameEvent[] = action === 'wait'
    ? [{ type: 'waited', position: { ...fly } }]
    : [{ type: 'moved', action, from, to: { ...fly } }];
  const lane = laneFor(state, fly.row);

  if (lane.kind === 'river' && action === 'wait') {
    const support = lane.hazards.find((hazard) => (
      hazard.kind === 'log' && hazardContains(lane, hazard, state.fly.column, state.time)
    ));
    if (support) {
      const displacement = (lane.direction ?? 0) * (lane.speed ?? 0) * DECISION_SECONDS;
      fly = { ...fly, column: fly.column + displacement };
      events.push({ type: 'carried', row: fly.row, displacement });
    }
  }

  let terminal: TerminalReason | null = null;
  if (Math.abs(fly.column) > WORLD_HALF_WIDTH) {
    terminal = 'bounds';
  } else if (lane.kind === 'river') {
    const supported = lane.hazards.some((hazard) => (
      hazard.kind === 'log' && hazardContains(lane, hazard, fly.column, toTime)
    ));
    if (!supported) terminal = 'water';
  } else if (lane.kind === 'road' || lane.kind === 'rail') {
    const collision = lane.hazards.find((hazard) => (
      hazardSweepsColumn(lane, hazard, fly.column, state.time, toTime)
    ));
    if (collision) terminal = lane.kind === 'rail' || collision.kind === 'train' ? 'train' : 'vehicle';
  }

  if (terminal === null && lane.kind === 'rail') {
    const warningEnd = toTime + TRAIN_WARNING_SECONDS;
    const approaching = lane.hazards.some((hazard) => (
      hazard.kind === 'train' && hazardSweepsColumn(lane, hazard, fly.column, toTime, warningEnd)
    ));
    if (approaching) events.push({ type: 'train-warning', row: lane.row });
  }
  if (terminal !== null) events.push(terminalEvent(terminal, fly));

  const next: GameState = {
    version: 1,
    seed: state.seed,
    step: state.step + 1,
    time: toTime,
    fly,
    score: Math.max(state.score, fly.row),
    lanes: extendWorld(state, fly.row),
    terminal,
    previousAction: action,
  };

  return { state: next, reward: calculateReward(state, next), events };
}
