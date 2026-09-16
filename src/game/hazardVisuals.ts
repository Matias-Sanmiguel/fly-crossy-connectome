import type { GameAssetRole } from './kenneyAssets.ts';
import { createRng } from './random.ts';
import type { Hazard } from './types.ts';

type CarAssetRole =
  Extract<GameAssetRole, `hazard.car.${string}`>;

type TruckAssetRole =
  Extract<GameAssetRole, `hazard.truck.${string}`>;

type LogAssetRole =
  Extract<GameAssetRole, `hazard.log.${string}`>;

const CAR_VISUAL_POOL: readonly CarAssetRole[] = [
  'hazard.car.sedan',
  'hazard.car.sedan',
  'hazard.car.sedan',
  'hazard.car.sedan',

  'hazard.car.suv',
  'hazard.car.suv',
  'hazard.car.suv',
  'hazard.car.suv',

  'hazard.car.sedan-sports',
  'hazard.car.sedan-sports',

  'hazard.car.hatchback-sports',
  'hazard.car.hatchback-sports',

  'hazard.car.suv-luxury',
  'hazard.car.suv-luxury',

  'hazard.car.taxi',
  'hazard.car.police',
];

const TRUCK_VISUAL_POOL: readonly TruckAssetRole[] = [
  'hazard.truck.truck',
  'hazard.truck.truck',
  'hazard.truck.truck',
  'hazard.truck.truck',

  'hazard.truck.van',
  'hazard.truck.van',
  'hazard.truck.van',
  'hazard.truck.van',

  'hazard.truck.delivery',
  'hazard.truck.delivery',
  'hazard.truck.delivery',
  'hazard.truck.delivery',

  'hazard.truck.garbage-truck',
  'hazard.truck.ambulance',
  'hazard.truck.firetruck',
];

const LOG_VISUAL_POOL: readonly LogAssetRole[] = [
  'hazard.log.small',
  'hazard.log.large',
];

export function visualRoleForHazard(
  seed: string,
  row: number,
  hazard: Hazard,
): GameAssetRole | null {
  if (hazard.kind === 'train') {
    return 'hazard.train';
  }

  const rng = createRng(
    [
      'hazard-visual:v1',
      seed,
      row,
      hazard.kind,
      hazard.position,
      hazard.size,
    ].join(':'),
  );

  if (hazard.kind === 'log') {
    return rng.pick(LOG_VISUAL_POOL);
  }

  if (hazard.kind === 'car') {
    return rng.pick(CAR_VISUAL_POOL);
  }

  return rng.pick(TRUCK_VISUAL_POOL);
}