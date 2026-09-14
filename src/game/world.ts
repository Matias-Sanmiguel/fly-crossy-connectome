import { createRng } from './random.ts';
import type { Direction, Hazard, HazardKind, Lane, LaneKind } from './types.ts';

export const WORLD_VERSION = 2;

const OPENING_ROWS = 3;
const HAZARD_ROWS_PER_GROUP = 4;
const GROUP_ROWS = HAZARD_ROWS_PER_GROUP + 1;
const WORLD_HALF_WIDTH = 5;
const HAZARD_CIRCUIT = 25;
const DECISION_SECONDS = 0.2;
const SOLVABILITY_STEP_LIMIT = 80;
const GENERATION_ATTEMPTS = 8;

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
  return Array.from({ length: count }, () => ({
    kind: rng.pick(kinds),
    position: rng.integer(-12, 12),
    size: kind === 'rail' ? 4 : rng.integer(1, 3),
  }));
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

function unwrappedHazardPositionAt(lane: Lane, hazard: Hazard, time: number): number {
  return hazard.position + (lane.direction ?? 0) * (lane.speed ?? 0) * (time + (lane.phase ?? 0));
}

function hazardPositionAt(lane: Lane, hazard: Hazard, time: number): number {
  const unwrapped = unwrappedHazardPositionAt(lane, hazard, time);
  const halfCircuit = HAZARD_CIRCUIT / 2;
  return ((unwrapped + halfCircuit) % HAZARD_CIRCUIT + HAZARD_CIRCUIT) % HAZARD_CIRCUIT
    - halfCircuit;
}

function hazardContains(lane: Lane, hazard: Hazard, column: number, time: number): boolean {
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
  const lower = Math.min(from, to) - hazard.size / 2;
  const upper = Math.max(from, to) + hazard.size / 2;
  return Math.ceil((lower - column) / HAZARD_CIRCUIT)
    <= Math.floor((upper - column) / HAZARD_CIRCUIT);
}

type ReachableState = { row: number; columnTicks: number; step: number };

/**
 * Check for a route from the safe row immediately before one generated group
 * to its recovery row. The bounded search uses the authoritative 0.2-second
 * movement/collision rules and integer fifth-cell ticks for exact JS/Python
 * parity. A canonical earliest-arrival phase is used for each group; the safe
 * row also permits waiting and lateral setup within the bound.
 */
export function hasBoundedGroupPath(rows: readonly Lane[]): boolean {
  if (rows.length !== HAZARD_ROWS_PER_GROUP + 2) return false;
  const ordered = [...rows].sort((left, right) => left.row - right.row);
  if (ordered.some((lane, index) => index > 0 && lane.row !== ordered[index - 1]!.row + 1)
    || ordered[0]!.kind !== 'grass' || ordered.at(-1)!.kind !== 'grass') return false;
  const lanes = new Map(ordered.map((lane) => [lane.row, lane]));
  const startRow = ordered[0]!.row;
  const goalRow = ordered.at(-1)!.row;
  let frontier: ReachableState[] = [{ row: startRow, columnTicks: 0, step: Math.max(0, startRow) }];
  const visited = new Set<string>();
  const actions = ['forward', 'backward', 'left', 'right', 'wait'] as const;

  for (let depth = 0; depth < SOLVABILITY_STEP_LIMIT && frontier.length > 0; depth += 1) {
    const next: ReachableState[] = [];
    for (const state of frontier) {
      const lane = lanes.get(state.row)!;
      const fromTime = state.step * DECISION_SECONDS;
      const toTime = fromTime + DECISION_SECONDS;
      const column = state.columnTicks / 5;
      if ((lane.kind === 'road' || lane.kind === 'rail')
        && lane.hazards.some((hazard) => hazardSweepsColumn(lane, hazard, column, fromTime, toTime))) {
        continue;
      }
      for (const action of actions) {
        let row = state.row + (action === 'forward' ? 1 : action === 'backward' ? -1 : 0);
        let columnTicks = state.columnTicks + (action === 'right' ? 5 : action === 'left' ? -5 : 0);
        if (row < startRow || row > goalRow || Math.abs(columnTicks) > WORLD_HALF_WIDTH * 5) continue;
        const destination = lanes.get(row)!;
        if (destination.kind === 'river' && action === 'wait') {
          const supportedBefore = destination.hazards.some((hazard) => (
            hazard.kind === 'log' && hazardContains(destination, hazard, column, fromTime)
          ));
          if (supportedBefore) columnTicks += (destination.direction ?? 0) * (destination.speed ?? 0);
        }
        const nextColumn = columnTicks / 5;
        if (Math.abs(columnTicks) > WORLD_HALF_WIDTH * 5) continue;
        if (destination.kind === 'river'
          && !destination.hazards.some((hazard) => (
            hazard.kind === 'log' && hazardContains(destination, hazard, nextColumn, toTime)
          ))) continue;
        if ((destination.kind === 'road' || destination.kind === 'rail')
          && destination.hazards.some((hazard) => hazardContains(destination, hazard, nextColumn, toTime))) {
          continue;
        }
        if (row === goalRow) return true;
        const candidate = { row, columnTicks, step: state.step + 1 };
        const key = `${candidate.step}:${candidate.row}:${candidate.columnTicks}`;
        if (!visited.has(key)) {
          visited.add(key);
          next.push(candidate);
        }
      }
    }
    frontier = next;
  }
  return false;
}

function generateGroup(seed: string, groupIndex: number): Lane[] {
  const first = OPENING_ROWS + groupIndex * GROUP_ROWS;
  for (let attempt = 0; attempt < GENERATION_ATTEMPTS; attempt += 1) {
    const hazards = Array.from(
      { length: HAZARD_ROWS_PER_GROUP },
      (_, offset) => hazardLane(seed, first + offset, groupIndex, attempt),
    );
    if (hasBoundedGroupPath([grass(first - 1), ...hazards, grass(first + HAZARD_ROWS_PER_GROUP)])) {
      return hazards;
    }
  }
  // Deterministic safe fallback: the full group becomes a grass crossing. This
  // is intentionally conservative and guarantees recovery instead of emitting
  // a knowingly impassable layout after the bounded retry budget.
  return Array.from({ length: HAZARD_ROWS_PER_GROUP }, (_, offset) => grass(first + offset));
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
