import { createRng } from './random.ts';
import type { GameAssetRole } from './kenneyAssets.ts';
import type { LaneKind } from './types.ts';

export type DecorationRole = Extract<GameAssetRole, `decoration.${string}`>;

export type SceneryDecoration = {
  role: DecorationRole;
  column: number;
  rotationY: number;
  scale: number;
};

const rolesByLane: Record<LaneKind, readonly DecorationRole[]> = {
  grass: ['decoration.tree', 'decoration.rocks', 'decoration.plant'],
  road: ['decoration.traffic-light'],
  rail: ['decoration.traffic-light'],
  river: ['decoration.rocks', 'decoration.plant'],
};

export function decorationsForRow(
  seed: string,
  row: number,
  laneKind: LaneKind,
): SceneryDecoration[] {
  const rng = createRng(`scenery:v1:${seed}:${row}:${laneKind}`);
  const maximum = laneKind === 'grass' ? 2 : 1;
  const count = rng.integer(0, maximum);
  const roles = rolesByLane[laneKind];

  return Array.from({ length: count }, (_, index) => {
    const side = index === 0
      ? rng.pick<1 | -1>([-1, 1])
      : -1 as const;
    const resolvedSide = index === 1
      ? -Math.sign(side || 1)
      : side;

    return {
      role: rng.pick(roles),
      column: resolvedSide * rng.integer(3, 11),
      rotationY: rng.next() * Math.PI * 2,
      scale: 0.85 + rng.next() * 0.3,
    };
  });
}
