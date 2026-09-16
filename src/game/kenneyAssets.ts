export const GAME_ASSET_ROLES = [
  'lane.road',
  'lane.rail',

  'decoration.traffic-light',
  'decoration.tree',
  'decoration.rocks',
  'decoration.plant',

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

  'hazard.train',
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

export const KENNEY_ASSETS = {
  'lane.road': {
    path: 'assets/kenney/city-roads/road-straight.glb',
    logicalSize: [25, 0.2, 0.94],
    rotation: [0, Math.PI / 2, 0],
    verticalOffset: -0.02,
  },

  'lane.rail': {
    path: 'assets/kenney/train-kit/track-detailed.glb',
    logicalSize: [25, 0.18, 0.94],
    rotation: [0, Math.PI / 2, 0],
    verticalOffset: 0,
  },

  'decoration.traffic-light': {
    path: 'assets/kenney/city-roads/traffic-light.glb',
    logicalSize: [0.65, 1.7, 0.65],
    rotation: [0, 0, 0],
    verticalOffset: 0,
  },

  'decoration.tree': {
    path: 'assets/kenney/mini-forest/tree.glb',
    logicalSize: [1.25, 2.6, 1.25],
    rotation: [0, 0, 0],
    verticalOffset: 0,
  },

  'decoration.rocks': {
    path: 'assets/kenney/mini-forest/rocks-low.glb',
    logicalSize: [1.1, 0.45, 0.85],
    rotation: [0, 0, 0],
    verticalOffset: 0,
  },

  'decoration.plant': {
    path: 'assets/kenney/mini-forest/plant.glb',
    logicalSize: [0.6, 0.55, 0.6],
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

  'hazard.train': {
    path: 'assets/kenney/train-kit/train-diesel-a.glb',
    logicalSize: [5.5, 1.3, 1],
    rotation: [0, Math.PI / 2, 0],
    verticalOffset: 0,
  },
} satisfies Record<
  GameAssetRole,
  KenneyAssetDefinition
>;

export const NATIVE_ASSET_ROLES = {
  'lane.grass': 'procedural-grass',
  'lane.river': 'procedural-water',
  'hazard.log': 'procedural-log',
  fly: 'procedural-fly',
} as const;