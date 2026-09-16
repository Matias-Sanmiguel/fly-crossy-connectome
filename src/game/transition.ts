import type { Action, Hazard, Lane } from './types.ts';

export type TerminalReason = 'vehicle' | 'train' | 'water' | 'bounds';

export type GridPosition = {
  row: number;
  column: number;
};

export type TransitionEvent =
  | { type: 'moved'; action: Exclude<Action, 'wait'>; from: GridPosition; to: GridPosition }
  | { type: 'waited'; position: GridPosition }
  | { type: 'carried'; row: number; displacement: number }
  | { type: 'train-warning'; row: number }
  | { type: 'terminal'; reason: TerminalReason; position: GridPosition };

export type LaneResolver = (row: number) => Lane;

export type LaneTransition = {
  fly: GridPosition;
  time: number;
  terminal: TerminalReason | null;
  events: TransitionEvent[];
};

export const DECISION_SECONDS = 0.2;
export const WORLD_HALF_WIDTH = 5;
export const HAZARD_CIRCUIT = 25;
export const TRAIN_CIRCUIT = 80;
export const TRAIN_WARNING_SECONDS = 1.2;

function hazardCircuitLength(hazard: Hazard): number {
  return hazard.kind === 'train' ? TRAIN_CIRCUIT : HAZARD_CIRCUIT;
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
  const circuit = hazardCircuitLength(hazard);
  const halfCircuit = circuit / 2;
  return ((unwrapped + halfCircuit) % circuit + circuit) % circuit
    - halfCircuit;
}

export function hazardContains(lane: Lane, hazard: Hazard, column: number, time: number): boolean {
  return Math.abs(hazardPositionAt(lane, hazard, time) - column) <= hazard.size / 2;
}

export function hazardSweepsColumn(
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
  const circuit = hazardCircuitLength(hazard);
  const firstImage = Math.ceil((lower - column) / circuit);
  const lastImage = Math.floor((upper - column) / circuit);
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

function collisionReason(lane: Lane, hazard: Hazard): TerminalReason {
  return lane.kind === 'rail' || hazard.kind === 'train' ? 'train' : 'vehicle';
}

/** Advance one action using the exact collision, carry, time, and bounds rules. */
export function advanceLaneTransition(
  position: GridPosition,
  time: number,
  action: Action,
  laneFor: LaneResolver,
): LaneTransition {
  const from = { ...position };
  const toTime = time + DECISION_SECONDS;
  let fly = { ...position };
  const events: TransitionEvent[] = [];
  const startingLane = laneFor(position.row);
  let terminal: TerminalReason | null = null;
  if (startingLane.kind === 'road' || startingLane.kind === 'rail') {
    const collision = startingLane.hazards.find((hazard) => (
      hazardSweepsColumn(startingLane, hazard, position.column, time, toTime)
    ));
    if (collision) terminal = collisionReason(startingLane, collision);
  }

  if (terminal === null) {
    fly = movedPosition(position, action);
    if (action === 'wait') events.push({ type: 'waited', position: { ...fly } });
    else events.push({ type: 'moved', action, from, to: { ...fly } });

    const destinationLane = laneFor(fly.row);
    if (destinationLane.kind === 'river' && action === 'wait') {
      const support = destinationLane.hazards.find((hazard) => (
        hazard.kind === 'log'
        && hazardContains(destinationLane, hazard, position.column, time)
      ));
      if (support) {
        const displacement = (destinationLane.direction ?? 0)
          * (destinationLane.speed ?? 0)
          * DECISION_SECONDS;
        fly = { ...fly, column: fly.column + displacement };
        events.push({ type: 'carried', row: fly.row, displacement });
      }
    }

    if (Math.abs(fly.column) > WORLD_HALF_WIDTH) {
      terminal = 'bounds';
    } else if (destinationLane.kind === 'river') {
      const supported = destinationLane.hazards.some((hazard) => (
        hazard.kind === 'log' && hazardContains(destinationLane, hazard, fly.column, toTime)
      ));
      if (!supported) terminal = 'water';
    } else if (destinationLane.kind === 'road' || destinationLane.kind === 'rail') {
      const collision = destinationLane.hazards.find((hazard) => (
        hazardContains(destinationLane, hazard, fly.column, toTime)
      ));
      if (collision) terminal = collisionReason(destinationLane, collision);
    }

    if (terminal === null && destinationLane.kind === 'rail') {
      const warningEnd = toTime + TRAIN_WARNING_SECONDS;
      const approaching = destinationLane.hazards.some((hazard) => (
        hazard.kind === 'train'
        && hazardSweepsColumn(destinationLane, hazard, fly.column, toTime, warningEnd)
      ));
      if (approaching) events.push({ type: 'train-warning', row: destinationLane.row });
    }
  }

  if (terminal !== null) {
    events.push({ type: 'terminal', reason: terminal, position: { ...fly } });
  }
  return { fly, time: toTime, terminal, events };
}

/** Match repeated authoritative fixed-step accumulation instead of multiplying. */
export function decisionTimeAfterSteps(steps: number): number {
  let time = 0;
  for (let step = 0; step < steps; step += 1) time += DECISION_SECONDS;
  return time;
}
