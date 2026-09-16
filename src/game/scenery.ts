import { createRng } from './random.ts';
import type { GameAssetRole } from './kenneyAssets.ts';
import type { LaneKind } from './types.ts';

export type DecorationRole = Extract<GameAssetRole, `decoration.${string}`>;

export type SceneryDecoration = {
  role: DecorationRole;
  column: number;
  rowOffset: number;
  rotationY: number;
  scale: number;
  blocking: boolean;
};

const OPENING_ROWS = 3;
const GROUP_ROWS = 7;
const RECOVERY_OFFSET = 6;
const OBSTACLE_CLEARANCE = 0.45;
const INTERIOR_COLUMNS = [-4, -3, -2, -1, 1, 2, 3, 4] as const;
const BOUNDARY_COLUMNS = [-8, -7, -6, 6, 7, 8] as const;

const TREE_ROLES: readonly DecorationRole[] = [
  'decoration.tree',
  'decoration.tree.oak',
  'decoration.tree.pine-round',
  'decoration.tree.fat',
  'decoration.tree.simple-dark',
  'decoration.tree.oak-fall',
];

const INTERIOR_ROLES: readonly DecorationRole[] = [
  'decoration.tree',
  'decoration.tree.oak',
  'decoration.tree.pine-round',
  'decoration.tree.fat',
  'decoration.tree.simple-dark',
  'decoration.tree.oak-fall',
  'decoration.rocks',
  'decoration.rocks.small-a',
  'decoration.rocks.small-c',
];

const PLANT_ROLES: readonly DecorationRole[] = [
  'decoration.plant',
  'decoration.plant.small',
];

const PLANT_COLUMNS = [-4, -3, -2, -1, 0, 1, 2, 3, 4] as const;

function positiveModulo(value: number, divisor: number): number {
  return ((value % divisor) + divisor) % divisor;
}

function receivesInteriorObstacles(row: number): boolean {
  if (row >= 0 && row < OPENING_ROWS) return false;
  return positiveModulo(row - OPENING_ROWS, GROUP_ROWS) !== RECOVERY_OFFSET;
}

export function obstacleColumnsForRow(
  seed: string,
  row: number,
  laneKind: LaneKind,
): number[] {
  if (laneKind !== 'grass' || !receivesInteriorObstacles(row)) return [];

  const rng = createRng(`obstacles:v1:${seed}:${row}`);
  const candidates = [...INTERIOR_COLUMNS];
  const count = rng.integer(1, 3);
  const columns: number[] = [];

  for (let index = 0; index < count; index += 1) {
    const candidateIndex = rng.integer(0, candidates.length - 1);
    columns.push(candidates.splice(candidateIndex, 1)[0]!);
  }

  return columns.sort((left, right) => left - right);
}

export function isSceneryBlocked(
  seed: string,
  row: number,
  laneKind: LaneKind,
  column: number,
): boolean {
  return obstacleColumnsForRow(seed, row, laneKind).some((obstacleColumn) => (
    Math.abs(obstacleColumn - column) < OBSTACLE_CLEARANCE
  ));
}

function scaleForRole(role: DecorationRole, rng: ReturnType<typeof createRng>): number {
  if (role.startsWith('decoration.tree')) return 0.88 + rng.next() * 0.22;
  if (role.startsWith('decoration.rocks.small')) return 1.35 + rng.next() * 0.35;
  if (role.startsWith('decoration.rocks')) return 0.95 + rng.next() * 0.2;
  return 1.15 + rng.next() * 0.3;
}

export function decorationsForRow(
  seed: string,
  row: number,
  laneKind: LaneKind,
): SceneryDecoration[] {
  if (laneKind !== 'grass') return [];

  const rng = createRng(`scenery:v2:${seed}:${row}`);
  const interior = obstacleColumnsForRow(seed, row, laneKind).map((column) => {
    const role = rng.pick(INTERIOR_ROLES);
    return {
      role,
      column,
      rowOffset: 0,
      rotationY: rng.next() * Math.PI * 2,
      scale: scaleForRole(role, rng),
      blocking: true,
    };
  });

  const plantRng = createRng(`plants:v1:${seed}:${row}`);
  const blockedColumns = new Set(interior.map((item) => item.column));
  const plantCandidates = PLANT_COLUMNS.filter(
    (column) => !blockedColumns.has(column),
  );
  const plantCount = plantRng.integer(0, 2);
  const plants = Array.from({ length: plantCount }, () => {
    const candidateIndex = plantRng.integer(0, plantCandidates.length - 1);
    const column = plantCandidates.splice(candidateIndex, 1)[0]!;
    const role = plantRng.pick(PLANT_ROLES);
    return {
      role,
      column,
      rowOffset: (plantRng.next() - 0.5) * 0.16,
      rotationY: plantRng.next() * Math.PI * 2,
      scale: scaleForRole(role, plantRng),
      blocking: false,
    };
  });

  // Crossy-style visual boundary. These trees sit just outside the playable
  // +/-5 columns. No palm asset exists in TREE_ROLES by design.
  const boundary = BOUNDARY_COLUMNS.map((column) => {
    const role = rng.pick(TREE_ROLES);
    return {
      role,
      column: column + (column < 0 ? -1 : 1) * rng.next() * 0.18,
      rowOffset: (rng.next() - 0.5) * 0.24,
      rotationY: rng.next() * Math.PI * 2,
      scale: 1.02 + rng.next() * 0.28,
      blocking: false,
    };
  });

  return [...interior, ...plants, ...boundary];
}
