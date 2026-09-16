import { createRng } from './random.ts';
import type { Direction, Hazard, HazardKind, Lane, LaneKind } from './types.ts';
import type { Action } from './types.ts';
import {
  advanceLaneTransition,
  decisionTimeAfterSteps,
  HAZARD_CIRCUIT,
  TRAIN_CIRCUIT,
} from './transition.ts';

export const WORLD_VERSION = 3;

const OPENING_ROWS = 3;
const CONTENT_ROWS_PER_GROUP = 6;
const GROUP_ROWS = CONTENT_ROWS_PER_GROUP + 1;
const SOLVABILITY_STEP_LIMIT = 80;
const SOLVABILITY_TRANSITION_LIMIT = 1_024;
const GENERATION_ATTEMPTS = 8;
const GROUP_CACHE_LIMIT = 512;
const groupCache = new Map<string, Lane[]>();

type HazardLaneKind = Exclude<LaneKind, 'grass'>;
type SectionFamily = HazardLaneKind;

const SECTION_FAMILY_POOL: readonly SectionFamily[] = [
  'road', 'road', 'road', 'road', 'road',
  'road', 'road', 'road', 'road', 'road',
  'rail', 'rail', 'rail', 'rail', 'rail', 'rail',
  'rail', 'rail', 'rail', 'rail', 'rail', 'rail',
  'river', 'river', 'river',
];
const SECTION_TEMPLATES: Record<
  SectionFamily,
  readonly [readonly LaneKind[], readonly LaneKind[]]
> = {
  road: [
    ['road', 'road', 'road', 'grass', 'road', 'road'],
    ['road', 'road', 'grass', 'grass', 'road', 'road'],
  ],
  rail: [
    ['rail', 'grass', 'road', 'road', 'road', 'road'],
    ['rail', 'grass', 'grass', 'road', 'road', 'road'],
  ],
  river: [
    ['river', 'river', 'grass', 'road', 'road', 'grass'],
    ['river', 'river', 'grass', 'road', 'road', 'grass'],
  ],
};

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

function laneTemplateForGroup(seed: string, groupIndex: number): readonly LaneKind[] {
  const rng = createRng(`${seedForGroup(seed, groupIndex)}:template`);
  const family = rng.pick(SECTION_FAMILY_POOL);
  return SECTION_TEMPLATES[family][rng.integer(0, 1)]!;
}

/**
 * Versioned distance progression. Every 50 forward rows raises speed first;
 * after 150 rows it also raises obstacle density. Row distance preserves the
 * previous five-row-group thresholds after groups expand to seven rows.
 * Solvability is checked after applying these parameters.
 */
export function difficultyForRow(row: number): DifficultyProfile {
  const forwardDistance = Math.max(0, row - OPENING_ROWS);
  const level = Math.min(3, Math.floor(forwardDistance / 50));
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
  if (kind === 'river') return 5;
  return rng.integer(difficulty.minimumHazards, difficulty.maximumHazards);
}

function createHazards(
  kind: Exclude<LaneKind, 'grass'>,
  rng: ReturnType<typeof createRng>,
  difficulty: DifficultyProfile,
): Hazard[] {
  const count = hazardCount(kind, rng, difficulty);
  const kinds = hazardKindByLane[kind];
  const circuit = kind === 'rail' ? TRAIN_CIRCUIT : HAZARD_CIRCUIT;
  const halfCircuit = Math.floor(circuit / 2);
  const firstPosition = rng.integer(-halfCircuit, halfCircuit);
  const spacing = Math.floor(circuit / count);
  return Array.from({ length: count }, (_, index) => {
    const hazardKind = rng.pick(kinds);
    const unwrappedPosition = firstPosition + index * spacing;
    const position = (
      (unwrappedPosition + halfCircuit) % circuit + circuit
    ) % circuit - halfCircuit;
    const size = hazardKind === 'train'
      ? 18
      : hazardKind === 'car'
        ? 2
        : hazardKind === 'truck'
          ? 3
          : rng.integer(2, 4);
    return { kind: hazardKind, position, size };
  });
}

function grass(row: number): Lane {
  return { row, kind: 'grass', hazards: [] };
}

function hazardLane(
  seed: string,
  row: number,
  groupIndex: number,
  attempt: number,
  kind: HazardLaneKind,
): Lane {
  const rowOffset = row - (OPENING_ROWS + groupIndex * GROUP_ROWS);
  const baseSeed = seedForGroup(seed, groupIndex);
  const groupSeed = attempt === 0 ? baseSeed : `${baseSeed}:retry:${attempt}`;
  const laneRng = createRng(`${groupSeed}:${rowOffset}`);
  const direction: Direction = laneRng.pick<Direction>([-1, 1]);
  const difficulty = difficultyForRow(row);

  return {
    row,
    kind,
    direction,
    speed: kind === 'rail'
      ? laneRng.integer(10, 12)
      : laneRng.integer(difficulty.minimumSpeed, difficulty.maximumSpeed),
    phase: laneRng.integer(0, 999) / 1000,
    hazards: createHazards(kind, laneRng, difficulty),
  };
}

type ReachableState = {
  row: number;
  column: number;
  time: number;
  actions: Action[];
  order: number;
};

/**
 * Check for a route from the safe row immediately before one generated group
 * to its recovery row. The bounded search uses the authoritative 0.2-second
 * movement/collision rules and repeated floating-point time accumulation used
 * by the game. A canonical earliest-arrival phase is used for each group; the safe
 * row also permits waiting and lateral setup within the bound.
 */
export function findBoundedGroupWitness(rows: readonly Lane[]): Action[] | null {
  if (rows.length !== CONTENT_ROWS_PER_GROUP + 2) return null;
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
    order: 0,
  }];
  const visited = new Set<string>();
  const actions: readonly Action[] = ['forward', 'left', 'right', 'wait', 'backward'];
  let transitionsChecked = 0;
  let nextOrder = 1;

  while (frontier.length > 0) {
    frontier.sort((left, right) => (
      right.row - left.row
      || left.actions.length - right.actions.length
      || left.order - right.order
    ));
    const state = frontier.shift()!;
    const depth = state.actions.length;
    if (depth >= SOLVABILITY_STEP_LIMIT) continue;
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
        order: nextOrder,
      };
      nextOrder += 1;
      const key = `${depth + 1}:${candidate.row}:${candidate.column}`;
      if (!visited.has(key)) {
        visited.add(key);
        frontier.push(candidate);
      }
    }
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
  const laneTemplate = laneTemplateForGroup(seed, groupIndex);
  let generated: Lane[] | null = null;
  for (let attempt = 0; attempt < GENERATION_ATTEMPTS; attempt += 1) {
    const content = Array.from(
      { length: CONTENT_ROWS_PER_GROUP },
      (_, offset) => laneTemplate[offset] === 'grass'
        ? grass(first + offset)
        : hazardLane(
          seed,
          first + offset,
          groupIndex,
          attempt,
          laneTemplate[offset] as HazardLaneKind,
        ),
    );
    if (hasBoundedGroupPath([grass(first - 1), ...content, grass(first + CONTENT_ROWS_PER_GROUP)])) {
      generated = content;
      break;
    }
  }
  // Deterministic safe fallback: the full group becomes a grass crossing. This
  // is intentionally conservative and guarantees recovery instead of emitting
  // a knowingly impassable layout after the bounded retry budget.
  generated ??= Array.from(
    { length: CONTENT_ROWS_PER_GROUP },
    (_, offset) => grass(first + offset),
  );
  groupCache.set(cacheKey, generated);
  if (groupCache.size > GROUP_CACHE_LIMIT) groupCache.delete(groupCache.keys().next().value!);
  return generated.map((lane) => ({ ...lane, hazards: [...lane.hazards] }));
}

/**
 * Generate lanes independently by coordinate. Every group has six
 * section-template rows followed by recovery grass, so ranges can be requested
 * in any order without changing their contents.
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
    if (groupOffset === CONTENT_ROWS_PER_GROUP) return grass(row);
    const groupIndex = groupIndexFor(row);
    if (!groups.has(groupIndex)) groups.set(groupIndex, generateGroup(seed, groupIndex));
    return groups.get(groupIndex)![groupOffset]!;
  });
}
