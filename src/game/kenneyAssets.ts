export const GAME_ASSET_ROLES = [
  'lane.road',
  'lane.rail',
  'decoration.traffic-light',
  'decoration.tree',
  'decoration.rocks',
  'decoration.plant',
  'hazard.car',
  'hazard.truck',
  'hazard.train',
] as const;

export type GameAssetRole = typeof GAME_ASSET_ROLES[number];

export type KenneyAssetDefinition = {
  path: string;
  logicalSize: readonly [number, number, number];
  rotation: readonly [number, number, number];
  verticalOffset: number;
};

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
  'hazard.car': {
    path: 'assets/kenney/car-kit/sedan.glb',
    logicalSize: [1.8, 0.72, 0.9],
    rotation: [0, Math.PI / 2, 0],
    verticalOffset: 0,
  },
  'hazard.truck': {
    path: 'assets/kenney/car-kit/truck.glb',
    logicalSize: [2.8, 1, 0.95],
    rotation: [0, Math.PI / 2, 0],
    verticalOffset: 0,
  },
  'hazard.train': {
    path: 'assets/kenney/train-kit/train-diesel-a.glb',
    logicalSize: [5.5, 1.3, 1],
    rotation: [0, Math.PI / 2, 0],
    verticalOffset: 0,
  },
} as const satisfies Record<GameAssetRole, KenneyAssetDefinition>;

export const NATIVE_ASSET_ROLES = {
  'lane.grass': 'procedural-grass',
  'lane.river': 'procedural-water',
  'hazard.log': 'procedural-log',
  fly: 'procedural-fly',
} as const;
