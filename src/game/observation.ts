import { HAZARD_CIRCUIT, WORLD_HALF_WIDTH, hazardContains, hazardPositionAt } from './simulation.ts';
import { TRAIN_CIRCUIT } from './transition.ts';
import type { GameState } from './simulation.ts';
import { isSceneryBlocked } from './scenery.ts';
import type { Action, Hazard, Lane, LaneKind } from './types.ts';

export const OBSERVATION_VERSION = 4 as const;

/** ObservationV4; legacy type name is retained to avoid protocol churn. */
export type ObservationV1 = {
  version: 4;
  radius: 5;
  cells: number[][];
  motion: number[][][];
  hazardOffset: number[][];
  support: 0 | 1;
  previousAction: Action;
  edgeDistance: number;
  signedColumn: number;
  traffic: number[][];
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
  blocker: 8,
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
  const speedScale = hazard.kind === 'train' ? 12 : 5;
  const normalizedSpeed = Math.min(1, (lane.speed ?? 0) / speedScale);
  return [direction, normalizedSpeed];
}

function occupyingHazardOffset(
  lane: Lane,
  hazard: Hazard | undefined,
  column: number,
  time: number,
): number {
  if (!hazard) return 0;
  const halfSize = Math.max(1, hazard.size / 2);
  const offset = (hazardPositionAt(lane, hazard, time) - column) / halfSize;
  return Math.max(-1, Math.min(1, offset));
}

function isRiverSupported(state: GameState): boolean {
  const lane = state.lanes.find((candidate) => candidate.row === state.fly.row);
  return lane?.kind === 'river' && lane.hazards.some((hazard) => (
    hazard.kind === 'log' && hazardContains(lane, hazard, state.fly.column, state.time)
  ));
}


export const TRAFFIC_RADAR_ROWS = 5;
export const TRAFFIC_RADAR_HORIZON_SECONDS = 2;

function positiveModulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function timeToHazardContact(
  lane: Lane,
  hazard: Hazard,
  column: number,
  time: number,
): number {
  const speed = lane.speed ?? 0;
  const direction = lane.direction ?? 0;
  if (speed <= 0 || direction === 0) return Number.POSITIVE_INFINITY;
  if (hazardContains(lane, hazard, column, time)) return 0;

  const center = hazardPositionAt(lane, hazard, time);
  const circuit = hazard.kind === 'train' ? TRAIN_CIRCUIT : HAZARD_CIRCUIT;
  const alongMotion = positiveModulo(direction * (column - center), circuit);
  return Math.max(0, alongMotion - hazard.size / 2) / speed;
}

export function trafficRadar(state: GameState): number[][] {
  const anchorColumn = Math.round(state.fly.column);
  const lanes = new Map(state.lanes.map((lane) => [lane.row, lane]));

  return Array.from({ length: TRAFFIC_RADAR_ROWS }, (_, rowOffset) => {
    const lane = lanes.get(state.fly.row + rowOffset);
    if (!lane) return [0, 0, 1, 1, 1];

    const scale = lane.kind === 'rail' ? 12 : 5;
    const signedSpeed = lane.kind === 'road' || lane.kind === 'rail'
      ? (lane.direction ?? 0) * Math.min(1, (lane.speed ?? 0) / scale)
      : 0;

    const ttc = [-1, 0, 1].map((columnOffset) => {
      const column = anchorColumn + columnOffset;
      if (Math.abs(column) > WORLD_HALF_WIDTH) return 0;
      if (lane.kind !== 'road' && lane.kind !== 'rail') return 1;

      let contact = Number.POSITIVE_INFINITY;
      for (const hazard of lane.hazards) {
        if (hazard.kind !== 'car' && hazard.kind !== 'truck' && hazard.kind !== 'train') continue;
        contact = Math.min(contact, timeToHazardContact(lane, hazard, column, state.time));
      }
      return Math.max(0, Math.min(1, contact / TRAFFIC_RADAR_HORIZON_SECONDS));
    });

    return [laneEncoding[lane.kind] / 8, signedSpeed, ...ttc];
  });
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
  const hazardOffset = Array.from(
    { length: size },
    () => Array<number>(size).fill(0),
  );

  for (let rowOffset = -OBSERVATION_RADIUS; rowOffset <= OBSERVATION_RADIUS; rowOffset += 1) {
    const lane = lanes.get(state.fly.row + rowOffset);
    if (!lane) continue;
    const observationRow = rowOffset + OBSERVATION_RADIUS;

    for (let columnOffset = -OBSERVATION_RADIUS; columnOffset <= OBSERVATION_RADIUS; columnOffset += 1) {
      const column = anchorColumn + columnOffset;
      if (Math.abs(column) > WORLD_HALF_WIDTH) continue;
      const observationColumn = columnOffset + OBSERVATION_RADIUS;
      if (isSceneryBlocked(state.seed, lane.row, lane.kind, column)) {
        cells[observationRow]![observationColumn] = CELL_ENCODING.blocker;
        motion[observationRow]![observationColumn] = [0, 0];
        hazardOffset[observationRow]![observationColumn] = 0;
        continue;
      }
      const hazard = occupyingHazard(lane, column, state.time);
      cells[observationRow]![observationColumn] = hazard
        ? occupiedEncoding(hazard)
        : laneEncoding[lane.kind];
      motion[observationRow]![observationColumn] = laneMotion(lane, hazard);
      hazardOffset[observationRow]![observationColumn] = occupyingHazardOffset(
        lane,
        hazard,
        column,
        state.time,
      );
    }
  }

  const edgeDistance = Math.max(0, Math.min(1, (
    WORLD_HALF_WIDTH - Math.abs(state.fly.column)
  ) / WORLD_HALF_WIDTH));
  const signedColumn = Math.max(
    -1,
    Math.min(1, state.fly.column / WORLD_HALF_WIDTH),
  );

  return {
    version: OBSERVATION_VERSION,
    radius: OBSERVATION_RADIUS,
    cells,
    motion,
    hazardOffset,
    support: isRiverSupported(state) ? 1 : 0,
    previousAction: state.previousAction ?? 'wait',
    edgeDistance,
    signedColumn,
    traffic: trafficRadar(state),
  };
}
