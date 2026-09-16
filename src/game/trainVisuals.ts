import type { GameAssetRole } from './kenneyAssets.ts';
import { createRng } from './random.ts';
import type { Hazard } from './types.ts';

type LocomotiveRole =
  | 'hazard.train'
  | Extract<GameAssetRole, `hazard.train.locomotive.${string}`>;

type CarriageRole =
  Extract<GameAssetRole, `hazard.train.carriage.${string}`>;

export type TrainConsistPart = {
  role: LocomotiveRole | CarriageRole;
  offset: number;
};

const TRAIN_PART_PITCH = 2;

const LOCOMOTIVES: readonly LocomotiveRole[] = [
  'hazard.train',
  'hazard.train.locomotive.diesel-b',
  'hazard.train.locomotive.diesel-c',
  'hazard.train.locomotive.a',
];

const CARRIAGES: readonly CarriageRole[] = [
  // Common, visually conventional freight cars.
  'hazard.train.carriage.box',
  'hazard.train.carriage.box',
  'hazard.train.carriage.box',
  'hazard.train.carriage.box',
  'hazard.train.carriage.box',

  'hazard.train.carriage.container-blue',
  'hazard.train.carriage.container-blue',
  'hazard.train.carriage.container-blue',
  'hazard.train.carriage.container-blue',
  'hazard.train.carriage.container-blue',

  'hazard.train.carriage.wood',
  'hazard.train.carriage.wood',
  'hazard.train.carriage.wood',

  'hazard.train.carriage.tank-large',
  'hazard.train.carriage.tank-large',

  // Specialty cargo appears occasionally.
  'hazard.train.carriage.coal',
  'hazard.train.carriage.dirt',
  'hazard.train.carriage.flatbed-wood',
  'hazard.train.carriage.lumber',
];

export function trainConsistParts(
  seed: string,
  row: number,
  hazard: Hazard,
): readonly TrainConsistPart[] {
  if (hazard.kind !== 'train') {
    throw new Error('Expected a train hazard.');
  }

  const rng = createRng(
    [
      'train-consist:v1',
      seed,
      row,
      hazard.position,
      hazard.size,
    ].join(':'),
  );

  const partCount = Math.max(
    2,
    Math.round(hazard.size / TRAIN_PART_PITCH),
  );
  const firstOffset = (partCount - 1) * TRAIN_PART_PITCH / 2;

  return Array.from({ length: partCount }, (_, index) => ({
    role: index === 0
      ? rng.pick(LOCOMOTIVES)
      : rng.pick(CARRIAGES),
    offset: firstOffset - index * TRAIN_PART_PITCH,
  }));
}