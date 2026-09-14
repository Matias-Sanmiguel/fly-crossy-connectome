import type { GameState } from './simulation.ts';
import type { Hazard, HazardKind, Lane } from './types.ts';

export const RENDER_LANE_CAPACITY = 32;
export const RENDER_HAZARD_CAPACITY = 128;

export type RenderableHazard = {
  lane: Lane;
  hazard: Hazard;
};

export type RenderableInstances = {
  lanes: Lane[];
  hazards: RenderableHazard[];
};

/** Select only the camera-nearest state needed to populate fixed GPU instance buffers. */
export function selectRenderableInstances(
  state: Pick<GameState, 'fly' | 'lanes'>,
): RenderableInstances {
  const lanes = [...state.lanes]
    .sort((left, right) => (
      Math.abs(left.row - state.fly.row) - Math.abs(right.row - state.fly.row)
      || left.row - right.row
    ))
    .slice(0, RENDER_LANE_CAPACITY)
    .sort((left, right) => left.row - right.row);
  const hazardCounts: Record<HazardKind, number> = { car: 0, truck: 0, train: 0, log: 0 };
  const hazards: RenderableHazard[] = [];

  for (const lane of lanes) {
    for (const hazard of lane.hazards) {
      if (hazardCounts[hazard.kind] >= RENDER_HAZARD_CAPACITY) continue;
      hazardCounts[hazard.kind] += 1;
      hazards.push({ lane, hazard });
    }
  }

  return { lanes, hazards };
}
