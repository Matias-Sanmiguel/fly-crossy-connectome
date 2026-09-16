import { createRng } from './random.ts';
import type { Direction, Hazard, HazardKind, Lane, LaneKind } from './types.ts';
import type { Action } from './types.ts';
import {
  advanceLaneTransition,
  decisionTimeAfterSteps,
  HAZARD_CIRCUIT,
} from './transition.ts';

export const WORLD_VERSION = 3;

const OPENING_ROWS = 3;
const HAZARD_ROWS_PER_GROUP = 4;
const GROUP_ROWS = HAZARD_ROWS_PER_GROUP + 1;
const SOLVABILITY_STEP_LIMIT = 80;
const SOLVABILITY_TRANSITION_LIMIT = 1_024;
const GENERATION_ATTEMPTS = 8;
const GROUP_CACHE_LIMIT = 512;
const groupCache = new Map<string, Lane[]>();

export type DifficultyProfile = {
  level: number;
  minimumSpeed: number;
  maximumSpeed: number;
  minimumHazards: number;
  maximumHazards: number;
};

const hazardKindByLane: Record<Exclude<LaneKind, 'grass'>, readonly HazardKind[]> = {
  road: ['car', 'truck'],
  rail: ['train'],
  river: ['log'],
};

function groupIndexFor(row: number): number {
  return Math.floor((row - OPENING_ROWS) / GROUP_ROWS);
}

function seedForGroup(seed: string, groupIndex: number): string {
  return `${WORLD_VERSION}:${seed}:${groupIndex}`;
}

/**
 * Versioned distance progression. Every ten groups raises speed first; after
 * thirty groups it also raises obstacle density. Solvability is checked after
 * applying these parameters, so increasing difficulty never removes the
 * bounded safe-route guarantee.
 */
export function difficultyForRow(row: number): DifficultyProfile {
  const forwardGroup = Math.max(0, groupIndexFor(row));
  const level = Math.min(3, Math.floor(forwardGroup / 10));
  return {
    level,
    minimumSpeed: Math.min(3, 1 + Math.floor(level / 2)),
    maximumSpeed: 2 + level,
    minimumHazards: 2 + Math.floor(level / 2),
    maximumHazards: 3 + level,
  };
}

function hazardCount(
  kind: Exclude<LaneKind, 'grass'>,
  rng: ReturnType<typeof createRng>,
  difficulty: DifficultyProfile,
): number {
  if (kind === 'rail') return 1;
  return rng.integer(difficulty.minimumHazards, difficulty.maximumHazards);
}

function createHazards(
  kind: Exclude<LaneKind, 'grass'>,
  rng: ReturnType<typeof createRng>,
  difficulty: DifficultyProfile,
): Hazard[] {
  const count = hazardCount(kind, rng, difficulty);
  const kinds = hazardKindByLane[kind];
  const halfCircuit = Math.floor(HAZARD_CIRCUIT / 2);
  const firstPosition = rng.integer(-halfCircuit, halfCircuit);
  const spacing = Math.floor(HAZARD_CIRCUIT / count);
  return Array.from({ length: count }, (_, index) => {
    const hazardKind = rng.pick(kinds);
    const unwrappedPosition = firstPosition + index * spacing;
    const position = (
      (unwrappedPosition + halfCircuit) % HAZARD_CIRCUIT + HAZARD_CIRCUIT
    ) % HAZARD_CIRCUIT - halfCircuit;
    const size = hazardKind === 'train'
      ? 4
      : hazardKind === 'car'
        ? 2
        : hazardKind === 'truck'
          ? 3
          : rng.integer(1, 3);
    return { kind: hazardKind, position, size };
  });
}

function grass(row: number): Lane {
  return { row, kind: 'grass', hazards: [] };
}

function hazardLane(seed: string, row: number, groupIndex: number, attempt: number): Lane {
  const rowOffset = row - (OPENING_ROWS + groupIndex * GROUP_ROWS);
  const baseSeed = seedForGroup(seed, groupIndex);
  const groupSeed = attempt === 0 ? baseSeed : `${baseSeed}:retry:${attempt}`;
  const groupRng = createRng(groupSeed);
  const kind = groupRng.pick<Exclude<LaneKind, 'grass'>>(['road', 'rail', 'river']);
  const laneRng = createRng(`${groupSeed}:${rowOffset}`);
  const direction: Direction = laneRng.pick<Direction>([-1, 1]);
  const difficulty = difficultyForRow(row);

  return {
    row,
    kind,
    direction,
    speed: laneRng.integer(difficulty.minimumSpeed, difficulty.maximumSpeed),
    phase: laneRng.integer(0, 999) / 1000,
    hazards: createHazards(kind, laneRng, difficulty),
  };
}

type ReachableState = { row: number; column: number; time: number; actions: Action[] };

/**
 * Check for a route from the safe row immediately before one generated group
 * to its recovery row. The bounded search uses the authoritative 0.2-second
 * movement/collision rules and repeated floating-point time accumulation used
 * by the game. A canonical earliest-arrival phase is used for each group; the safe
 * row also permits waiting and lateral setup within the bound.
 */
export function findBoundedGroupWitness(rows: readonly Lane[]): Action[] | null {
  if (rows.length !== HAZARD_ROWS_PER_GROUP + 2) return null;
  const ordered = [...rows].sort((left, right) => left.row - right.row);
  if (ordered.some((lane, index) => index > 0 && lane.row !== ordered[index - 1]!.row + 1)
    || ordered[0]!.kind !== 'grass' || ordered.at(-1)!.kind !== 'grass') return null;
  const lanes = new Map(ordered.map((lane) => [lane.row, lane]));
  const startRow = ordered[0]!.row;
  const goalRow = ordered.at(-1)!.row;
  let frontier: ReachableState[] = [{
    row: startRow,
    column: 0,
    time: decisionTimeAfterSteps(Math.max(0, startRow)),
    actions: [],
  }];
  const visited = new Set<string>();
  const actions: readonly Action[] = ['forward', 'backward', 'left', 'right', 'wait'];
  let transitionsChecked = 0;

  for (let depth = 0; depth < SOLVABILITY_STEP_LIMIT && frontier.length > 0; depth += 1) {
    const next: ReachableState[] = [];
    for (const state of frontier) {
      for (const action of actions) {
        if (transitionsChecked >= SOLVABILITY_TRANSITION_LIMIT) return null;
        transitionsChecked += 1;
        const attemptedRow = state.row
          + (action === 'forward' ? 1 : action === 'backward' ? -1 : 0);
        if (attemptedRow < startRow || attemptedRow > goalRow) continue;
        const transition = advanceLaneTransition(
          { row: state.row, column: state.column },
          state.time,
          action,
          (row) => lanes.get(row)!,
        );
        if (transition.terminal !== null) continue;
        const witness = [...state.actions, action];
        if (transition.fly.row === goalRow) return witness;
        const candidate = {
          row: transition.fly.row,
          column: transition.fly.column,
          time: transition.time,
          actions: witness,
        };
        const key = `${depth + 1}:${candidate.row}:${candidate.column}`;
        if (!visited.has(key)) {
          visited.add(key);
          next.push(candidate);
        }
      }
    }
    frontier = next;
  }
  return null;
}

export function hasBoundedGroupPath(rows: readonly Lane[]): boolean {
  return findBoundedGroupWitness(rows) !== null;
}

function generateGroup(seed: string, groupIndex: number): Lane[] {
  const cacheKey = `${seed}:${groupIndex}`;
  const cached = groupCache.get(cacheKey);
  if (cached) return cached.map((lane) => ({ ...lane, hazards: [...lane.hazards] }));
  const first = OPENING_ROWS + groupIndex * GROUP_ROWS;
  let generated: Lane[] | null = null;
  for (let attempt = 0; attempt < GENERATION_ATTEMPTS; attempt += 1) {
    const hazards = Array.from(
      { length: HAZARD_ROWS_PER_GROUP },
      (_, offset) => hazardLane(seed, first + offset, groupIndex, attempt),
    );
    if (hasBoundedGroupPath([grass(first - 1), ...hazards, grass(first + HAZARD_ROWS_PER_GROUP)])) {
      generated = hazards;
      break;
    }
  }
  // Deterministic safe fallback: the full group becomes a grass crossing. This
  // is intentionally conservative and guarantees recovery instead of emitting
  // a knowingly impassable layout after the bounded retry budget.
  generated ??= Array.from(
    { length: HAZARD_ROWS_PER_GROUP },
    (_, offset) => grass(first + offset),
  );
  groupCache.set(cacheKey, generated);
  if (groupCache.size > GROUP_CACHE_LIMIT) groupCache.delete(groupCache.keys().next().value!);
  return generated.map((lane) => ({ ...lane, hazards: [...lane.hazards] }));
}

/**
 * Generate lanes independently by coordinate. Every group has four hazardous
 * rows followed by a grass recovery row, so ranges can be requested in any
 * order without changing their contents.
 */
export function generateRows(seed: string, from: number, count: number): Lane[] {
  if (!Number.isInteger(from) || !Number.isInteger(count) || count < 0) {
    throw Error('Expected an integer starting row and nonnegative row count.');
  }

  const groups = new Map<number, Lane[]>();
  return Array.from({ length: count }, (_, offset) => {
    const row = from + offset;
    if (row >= 0 && row < OPENING_ROWS) return grass(row);
    const groupOffset = row - (OPENING_ROWS + groupIndexFor(row) * GROUP_ROWS);
    if (groupOffset === HAZARD_ROWS_PER_GROUP) return grass(row);
    const groupIndex = groupIndexFor(row);
    if (!groups.has(groupIndex)) groups.set(groupIndex, generateGroup(seed, groupIndex));
    return groups.get(groupIndex)![groupOffset]!;
  });
}
