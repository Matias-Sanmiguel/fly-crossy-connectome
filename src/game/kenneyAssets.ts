export const GAME_ASSET_ROLES = [
  'lane.road',
  'lane.rail',

  'decoration.traffic-light',
  'decoration.rail-warning-light',
  'decoration.tree',
  'decoration.tree.oak',
  'decoration.tree.pine-round',
  'decoration.tree.fat',
  'decoration.rocks',
  'decoration.rocks.small-a',
  'decoration.rocks.small-c',
  'decoration.plant',
  'decoration.plant.small',

  'hazard.car.sedan',
  'hazard.car.sedan-sports',
  'hazard.car.hatchback-sports',
  'hazard.car.suv',
  'hazard.car.suv-luxury',
  'hazard.car.taxi',
  'hazard.car.police',

  'hazard.truck.truck',
  'hazard.truck.van',
  'hazard.truck.delivery',
  'hazard.truck.garbage-truck',
  'hazard.truck.ambulance',
  'hazard.truck.firetruck',

  'hazard.log.small',
  'hazard.log.large',

  'hazard.train',
  'hazard.train.locomotive.diesel-b',
  'hazard.train.locomotive.diesel-c',
  'hazard.train.locomotive.a',
  'hazard.train.carriage.box',
  'hazard.train.carriage.coal',
  'hazard.train.carriage.container-blue',
  'hazard.train.carriage.dirt',
  'hazard.train.carriage.flatbed-wood',
  'hazard.train.carriage.lumber',
  'hazard.train.carriage.tank-large',
  'hazard.train.carriage.wood',
] as const;

export type GameAssetRole =
  typeof GAME_ASSET_ROLES[number];

export type KenneyAssetDefinition = {
  path: string;
  logicalSize: readonly [number, number, number];
  rotation: readonly [number, number, number];
  verticalOffset: number;
};

const carDefinition = (
  path: string,
): KenneyAssetDefinition => ({
  path,
  logicalSize: [1.8, 0.72, 0.9],
  rotation: [0, Math.PI / 2, 0],
  verticalOffset: 0,
});

const truckDefinition = (
  path: string,
): KenneyAssetDefinition => ({
  path,
  logicalSize: [2.8, 1, 0.95],
  rotation: [0, Math.PI / 2, 0],
  verticalOffset: 0,
});

const logDefinition = (
  path: string,
  length: number,
): KenneyAssetDefinition => ({
  path,
  logicalSize: [length, 0.28, 0.25],
  rotation: [0, Math.PI / 2, 0],
  verticalOffset: 0,
});

const trainLocomotiveDefinition = (
  path: string,
): KenneyAssetDefinition => ({
  path,
  logicalSize: [2, 1.3, 0.95],
  rotation: [0, Math.PI / 2, 0],
  verticalOffset: 0,
});

const trainCarriageDefinition = (
  path: string,
): KenneyAssetDefinition => ({
  path,
  logicalSize: [1.9, 1.15, 0.95],
  rotation: [0, Math.PI / 2, 0],
  verticalOffset: 0,
});

export const KENNEY_ASSETS = {
  'lane.road': {
    path: 'assets/kenney/city-roads/road-straight.glb',
    logicalSize: [1, 0.2, 1],
    rotation: [0, Math.PI / 2, 0],
    verticalOffset: -0.02,
  },

  'lane.rail': {
    path: 'assets/kenney/train-kit/track-detailed.glb',
    logicalSize: [1, 0.18, 1],
    rotation: [0, Math.PI / 2, 0],
    verticalOffset: 0,
  },

  'decoration.traffic-light': {
    path: 'assets/kenney/city-roads/traffic-light.glb',
    logicalSize: [0.65, 1.7, 0.65],
    rotation: [0, 0, 0],
    verticalOffset: 0,
  },

  'decoration.rail-warning-light': {
    path: 'assets/kenney/city-roads/road-sign-stop.glb',
    logicalSize: [0.38, 1.15, 0.38],
    rotation: [0, 0, 0],
    verticalOffset: 0,
  },

  'decoration.tree': {
    path: 'assets/kenney/nature-kit/tree-default.glb',
    logicalSize: [1, 2.2, 1],
    rotation: [0, 0, 0],
    verticalOffset: 0,
  },

  'decoration.tree.oak': {
    path: 'assets/kenney/nature-kit/tree-oak.glb',
    logicalSize: [1.05, 1.8, 1.05],
    rotation: [0, 0, 0],
    verticalOffset: 0,
  },

  'decoration.tree.pine-round': {
    path: 'assets/kenney/nature-kit/tree-pine-round-a.glb',
    logicalSize: [1, 2, 1],
    rotation: [0, 0, 0],
    verticalOffset: 0,
  },

  'decoration.tree.fat': {
    path: 'assets/kenney/nature-kit/tree-fat.glb',
    logicalSize: [1.1, 1.7, 1],
    rotation: [0, 0, 0],
    verticalOffset: 0,
  },

  'decoration.rocks': {
    path: 'assets/kenney/nature-kit/rock-large-a.glb',
    logicalSize: [0.95, 0.45, 0.85],
    rotation: [0, 0, 0],
    verticalOffset: 0,
  },

  'decoration.rocks.small-a': {
    path: 'assets/kenney/nature-kit/rock-small-a.glb',
    logicalSize: [0.55, 0.3, 0.55],
    rotation: [0, 0, 0],
    verticalOffset: 0,
  },

  'decoration.rocks.small-c': {
    path: 'assets/kenney/nature-kit/rock-small-c.glb',
    logicalSize: [0.55, 0.24, 0.55],
    rotation: [0, 0, 0],
    verticalOffset: 0,
  },

  'decoration.plant': {
    path: 'assets/kenney/nature-kit/plant-bush.glb',
    logicalSize: [0.72, 0.5, 0.72],
    rotation: [0, 0, 0],
    verticalOffset: 0,
  },

  'decoration.plant.small': {
    path: 'assets/kenney/nature-kit/plant-bush-small.glb',
    logicalSize: [0.58, 0.42, 0.58],
    rotation: [0, 0, 0],
    verticalOffset: 0,
  },

  'hazard.car.sedan': carDefinition(
    'assets/kenney/car-kit/sedan.glb',
  ),

  'hazard.car.sedan-sports': carDefinition(
    'assets/kenney/car-kit/sedan-sports.glb',
  ),

  'hazard.car.hatchback-sports': carDefinition(
    'assets/kenney/car-kit/hatchback-sports.glb',
  ),

  'hazard.car.suv': carDefinition(
    'assets/kenney/car-kit/suv.glb',
  ),

  'hazard.car.suv-luxury': carDefinition(
    'assets/kenney/car-kit/suv-luxury.glb',
  ),

  'hazard.car.taxi': carDefinition(
    'assets/kenney/car-kit/taxi.glb',
  ),

  'hazard.car.police': carDefinition(
    'assets/kenney/car-kit/police.glb',
  ),

  'hazard.truck.truck': truckDefinition(
    'assets/kenney/car-kit/truck.glb',
  ),

  'hazard.truck.van': truckDefinition(
    'assets/kenney/car-kit/van.glb',
  ),

  'hazard.truck.delivery': truckDefinition(
    'assets/kenney/car-kit/delivery.glb',
  ),

  'hazard.truck.garbage-truck': truckDefinition(
    'assets/kenney/car-kit/garbage-truck.glb',
  ),

  'hazard.truck.ambulance': truckDefinition(
    'assets/kenney/car-kit/ambulance.glb',
  ),

  'hazard.truck.firetruck': truckDefinition(
    'assets/kenney/car-kit/firetruck.glb',
  ),

  'hazard.log.small': logDefinition(
    'assets/kenney/survival-kit/tree-log-small.glb',
    0.65,
  ),

  'hazard.log.large': logDefinition(
    'assets/kenney/survival-kit/tree-log.glb',
    1,
  ),

  'hazard.train': trainLocomotiveDefinition(
    'assets/kenney/train-kit/train-diesel-a.glb',
  ),

  'hazard.train.locomotive.diesel-b': trainLocomotiveDefinition(
    'assets/kenney/train-kit/train-diesel-b.glb',
  ),

  'hazard.train.locomotive.diesel-c': trainLocomotiveDefinition(
    'assets/kenney/train-kit/train-diesel-c.glb',
  ),

  'hazard.train.locomotive.a': trainLocomotiveDefinition(
    'assets/kenney/train-kit/train-locomotive-a.glb',
  ),

  'hazard.train.carriage.box': trainCarriageDefinition(
    'assets/kenney/train-kit/train-carriage-box.glb',
  ),

  'hazard.train.carriage.coal': trainCarriageDefinition(
    'assets/kenney/train-kit/train-carriage-coal.glb',
  ),

  'hazard.train.carriage.container-blue': trainCarriageDefinition(
    'assets/kenney/train-kit/train-carriage-container-blue.glb',
  ),

  'hazard.train.carriage.dirt': trainCarriageDefinition(
    'assets/kenney/train-kit/train-carriage-dirt.glb',
  ),

  'hazard.train.carriage.flatbed-wood': trainCarriageDefinition(
    'assets/kenney/train-kit/train-carriage-flatbed-wood.glb',
  ),

  'hazard.train.carriage.lumber': trainCarriageDefinition(
    'assets/kenney/train-kit/train-carriage-lumber.glb',
  ),

  'hazard.train.carriage.tank-large': trainCarriageDefinition(
    'assets/kenney/train-kit/train-carriage-tank-large.glb',
  ),

  'hazard.train.carriage.wood': trainCarriageDefinition(
    'assets/kenney/train-kit/train-carriage-wood.glb',
  ),
} satisfies Record<
  GameAssetRole,
  KenneyAssetDefinition
>;

export const NATIVE_ASSET_ROLES = {
  'lane.grass': 'procedural-grass',
  'lane.river': 'procedural-water',
  fly: 'procedural-fly',
} as const;
