import { WORLD_HALF_WIDTH, hazardContains } from './simulation.ts';
import type { GameState } from './simulation.ts';
import type { Action, Hazard, Lane, LaneKind } from './types.ts';

export type ObservationV1 = {
  version: 1;
  radius: 5;
  cells: number[][];
  motion: number[][][];
  support: 0 | 1;
  previousAction: Action;
  edgeDistance: number;
};

export const OBSERVATION_RADIUS = 5;
export const CELL_ENCODING = {
  unknown: 0,
  grass: 1,
  road: 2,
  rail: 3,
  river: 4,
  vehicle: 5,
  train: 6,
  log: 7,
} as const;

const laneEncoding: Record<LaneKind, number> = {
  grass: CELL_ENCODING.grass,
  road: CELL_ENCODING.road,
  rail: CELL_ENCODING.rail,
  river: CELL_ENCODING.river,
};

function occupyingHazard(lane: Lane, column: number, time: number): Hazard | undefined {
  return lane.hazards.find((hazard) => hazardContains(lane, hazard, column, time));
}

function occupiedEncoding(hazard: Hazard): number {
  if (hazard.kind === 'train') return CELL_ENCODING.train;
  if (hazard.kind === 'log') return CELL_ENCODING.log;
  return CELL_ENCODING.vehicle;
}

function laneMotion(lane: Lane, hazard: Hazard | undefined): number[] {
  if (!hazard) return [0, 0];
  const direction = lane.direction ?? 0;
  const normalizedSpeed = Math.min(1, (lane.speed ?? 0) / 3);
  return [direction, normalizedSpeed];
}

function isRiverSupported(state: GameState): boolean {
  const lane = state.lanes.find((candidate) => candidate.row === state.fly.row);
  return lane?.kind === 'river' && lane.hazards.some((hazard) => (
    hazard.kind === 'log' && hazardContains(lane, hazard, state.fly.column, state.time)
  ));
}

/** Encode a bounded egocentric snapshot with no rows or columns outside radius five. */
export function observe(state: GameState): ObservationV1 {
  const size = OBSERVATION_RADIUS * 2 + 1;
  const anchorColumn = Math.round(state.fly.column);
  const lanes = new Map(state.lanes.map((lane) => [lane.row, lane]));
  const cells = Array.from({ length: size }, () => Array<number>(size).fill(CELL_ENCODING.unknown));
  const motion = Array.from(
    { length: size },
    () => Array.from({ length: size }, () => [0, 0]),
  );

  for (let rowOffset = -OBSERVATION_RADIUS; rowOffset <= OBSERVATION_RADIUS; rowOffset += 1) {
    const lane = lanes.get(state.fly.row + rowOffset);
    if (!lane) continue;
    const observationRow = rowOffset + OBSERVATION_RADIUS;

    for (let columnOffset = -OBSERVATION_RADIUS; columnOffset <= OBSERVATION_RADIUS; columnOffset += 1) {
      const column = anchorColumn + columnOffset;
      if (Math.abs(column) > WORLD_HALF_WIDTH) continue;
      const observationColumn = columnOffset + OBSERVATION_RADIUS;
      const hazard = occupyingHazard(lane, column, state.time);
      cells[observationRow]![observationColumn] = hazard
        ? occupiedEncoding(hazard)
        : laneEncoding[lane.kind];
      motion[observationRow]![observationColumn] = laneMotion(lane, hazard);
    }
  }

  const edgeDistance = Math.max(0, Math.min(1, (
    WORLD_HALF_WIDTH - Math.abs(state.fly.column)
  ) / WORLD_HALF_WIDTH));

  return {
    version: 1,
    radius: OBSERVATION_RADIUS,
    cells,
    motion,
    support: isRiverSupported(state) ? 1 : 0,
    previousAction: state.previousAction ?? 'wait',
    edgeDistance,
  };
}
