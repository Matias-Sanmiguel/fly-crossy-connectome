import { createRng } from './random.ts';
import type { Direction, Hazard, HazardKind, Lane, LaneKind } from './types.ts';

export const WORLD_VERSION = 1;

const OPENING_ROWS = 3;
const HAZARD_ROWS_PER_GROUP = 4;
const GROUP_ROWS = HAZARD_ROWS_PER_GROUP + 1;

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

function hazardCount(kind: Exclude<LaneKind, 'grass'>, rng: ReturnType<typeof createRng>): number {
  if (kind === 'rail') return 1;
  return rng.integer(2, 4);
}

function createHazards(kind: Exclude<LaneKind, 'grass'>, rng: ReturnType<typeof createRng>): Hazard[] {
  const count = hazardCount(kind, rng);
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

function hazardLane(seed: string, row: number): Lane {
  const groupIndex = groupIndexFor(row);
  const rowOffset = row - (OPENING_ROWS + groupIndex * GROUP_ROWS);
  const groupSeed = seedForGroup(seed, groupIndex);
  const groupRng = createRng(groupSeed);
  const kind = groupRng.pick<Exclude<LaneKind, 'grass'>>(['road', 'rail', 'river']);
  const laneRng = createRng(`${groupSeed}:${rowOffset}`);
  const direction: Direction = laneRng.pick<Direction>([-1, 1]);

  return {
    row,
    kind,
    direction,
    speed: laneRng.integer(1, 3),
    phase: laneRng.integer(0, 999) / 1000,
    hazards: createHazards(kind, laneRng),
  };
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

  return Array.from({ length: count }, (_, offset) => {
    const row = from + offset;
    if (row >= 0 && row < OPENING_ROWS) return grass(row);
    const groupOffset = row - (OPENING_ROWS + groupIndexFor(row) * GROUP_ROWS);
    return groupOffset === HAZARD_ROWS_PER_GROUP ? grass(row) : hazardLane(seed, row);
  });
}
