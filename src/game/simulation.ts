import { reward as calculateReward } from './reward.ts';
import type { Action, Lane } from './types.ts';
import { generateRows, WORLD_VERSION } from './world.ts';
import { advanceLaneTransition } from './transition.ts';
import type { GridPosition, TerminalReason, TransitionEvent } from './transition.ts';

export {
  DECISION_SECONDS,
  HAZARD_CIRCUIT,
  WORLD_HALF_WIDTH,
  hazardContains,
  hazardPositionAt,
} from './transition.ts';
export type { GridPosition, TerminalReason } from './transition.ts';
export type GameEvent = TransitionEvent;

export type GameState = {
  version: typeof WORLD_VERSION;
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

const ROWS_AHEAD = 15;
const ROWS_BEHIND = 8;

function laneFor(state: GameState, row: number): Lane {
  return state.lanes.find((lane) => lane.row === row) ?? generateRows(state.seed, row, 1)[0]!;
}

function extendWorld(state: GameState, flyRow: number): Lane[] {
  const minimum = flyRow - ROWS_BEHIND;
  const maximum = flyRow + ROWS_AHEAD;
  const retained = new Map(
    state.lanes
      .filter((lane) => lane.row >= minimum)
      .map((lane) => [lane.row, lane]),
  );

  for (const lane of generateRows(state.seed, minimum, maximum - minimum + 1)) {
    if (!retained.has(lane.row)) retained.set(lane.row, lane);
  }

  return [...retained.values()].sort((left, right) => left.row - right.row);
}

export function createGame(seed: string): GameState {
  return {
    version: WORLD_VERSION,
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
  const transition = advanceLaneTransition(
    state.fly,
    state.time,
    action,
    (row) => laneFor(state, row),
  );

  const next: GameState = {
    version: WORLD_VERSION,
    seed: state.seed,
    step: state.step + 1,
    time: transition.time,
    fly: transition.fly,
    score: Math.max(state.score, transition.fly.row),
    lanes: extendWorld(state, transition.fly.row),
    terminal: transition.terminal,
    previousAction: action,
  };

  return { state: next, reward: calculateReward(state, next), events: transition.events };
}
