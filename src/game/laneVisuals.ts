import type { LaneKind } from './types.ts';

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
