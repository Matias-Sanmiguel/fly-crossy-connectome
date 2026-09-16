import type { LaneKind } from './types.ts';

export const LANE_ROW_DEPTH = 1;

export type LaneSubstrateLayout = {
  position: readonly [number, number, number];
  size: readonly [number, number, number];
};

export function laneSubstrateLayout(circuitLength: number): LaneSubstrateLayout {
  if (!Number.isFinite(circuitLength) || circuitLength <= 0) {
    throw new Error('Expected a positive finite circuit length.');
  }
  return {
    position: [0, -0.12, 0],
    size: [circuitLength, 0.2, LANE_ROW_DEPTH],
  };
}

export type LaneDetail = {
  kind: 'rail' | 'sleeper';
  position: readonly [number, number, number];
  size: readonly [number, number, number];
};

const SLEEPER_SPACING = 0.76;

export function laneDetails(
  kind: LaneKind,
  circuitLength: number,
): readonly LaneDetail[] {
  if (kind !== 'rail' || !Number.isFinite(circuitLength) || circuitLength <= 0) {
    return [];
  }

  const halfLength = circuitLength / 2;
  const details: LaneDetail[] = [
    {
      kind: 'rail',
      position: [0, 0.015, -0.25],
      size: [circuitLength, 0.07, 0.055],
    },
    {
      kind: 'rail',
      position: [0, 0.015, 0.25],
      size: [circuitLength, 0.07, 0.055],
    },
  ];

  for (let x = -halfLength; x <= halfLength; x += SLEEPER_SPACING) {
    details.push({
      kind: 'sleeper',
      position: [x, -0.025, 0],
      size: [0.22, 0.055, 0.72],
    });
  }
  return details;
}
