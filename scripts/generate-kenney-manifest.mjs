import { createHash } from 'node:crypto';
import {
  readFile,
  writeFile,
} from 'node:fs/promises';

const publicRoot =
  new URL('../public/', import.meta.url);

const PACKS = {
  'car-kit': {
    source: 'https://kenney.nl/assets/car-kit',
    archiveSha256:
      'fac7dacac5c7874348cf19729af3ef205f3d366493edaf0a827d93f4fdf3d0c4',
  },

  'city-roads': {
    source:
      'https://kenney.nl/assets/city-kit-roads',
    archiveSha256:
      '22058af3d68173a7cf9bda9f0e243a8cef6bd68168c302ebc76327063849674e',
  },

  'train-kit': {
    source:
      'https://kenney.nl/assets/train-kit',
    archiveSha256:
      'cf50d77e8cbacbf38dd50826d4bce5392db8e4f67373d3c4e583b0ed0e474475',
  },

  'nature-kit': {
    source:
      'https://kenney.nl/assets/nature-kit',
    archiveSha256:
      'fa7974a0d342bfe63c38664ba9f8ec1a4aab8ea25f099bdc56870e33588c4d9d',
    textureless: true,
  },
};

const ASSETS = [
  // City Kit Roads
  ['city-roads', 'road-straight.glb', 'lane.road'],
  [
    'city-roads',
    'traffic-light.glb',
    'decoration.traffic-light',
  ],
  [
    'city-roads',
    'road-sign-stop.glb',
    'decoration.rail-warning-light',
  ],
  [
    'city-roads',
    'road-sign-warning.glb',
    'decoration.road-sign',
  ],
  [
    'city-roads',
    'light-curved.glb',
    'decoration.street-light',
  ],
  [
    'city-roads',
    'construction-barrier.glb',
    'decoration.barrier',
  ],

  // Car Kit — cars
  ['car-kit', 'sedan.glb', 'hazard.car'],
  ['car-kit', 'sedan-sports.glb', 'hazard.car'],
  [
    'car-kit',
    'hatchback-sports.glb',
    'hazard.car',
  ],
  ['car-kit', 'suv.glb', 'hazard.car'],
  ['car-kit', 'suv-luxury.glb', 'hazard.car'],
  ['car-kit', 'taxi.glb', 'hazard.car'],
  ['car-kit', 'police.glb', 'hazard.car'],

  // Car Kit — larger vehicles
  ['car-kit', 'truck.glb', 'hazard.truck'],
  ['car-kit', 'van.glb', 'hazard.truck'],
  ['car-kit', 'delivery.glb', 'hazard.truck'],
  [
    'car-kit',
    'garbage-truck.glb',
    'hazard.truck',
  ],
  ['car-kit', 'ambulance.glb', 'hazard.truck'],
  ['car-kit', 'firetruck.glb', 'hazard.truck'],

  // Train Kit
  ['train-kit', 'track-detailed.glb', 'lane.rail'],

  [
    'train-kit',
    'train-diesel-a.glb',
    'hazard.train',
  ],
  [
    'train-kit',
    'train-diesel-b.glb',
    'hazard.train',
  ],
  [
    'train-kit',
    'train-diesel-c.glb',
    'hazard.train',
  ],
  [
    'train-kit',
    'train-locomotive-a.glb',
    'hazard.train',
  ],

  [
    'train-kit',
    'train-carriage-box.glb',
    'hazard.train',
  ],
  [
    'train-kit',
    'train-carriage-coal.glb',
    'hazard.train',
  ],
  [
    'train-kit',
    'train-carriage-container-blue.glb',
    'hazard.train',
  ],
  [
    'train-kit',
    'train-carriage-dirt.glb',
    'hazard.train',
  ],
  [
    'train-kit',
    'train-carriage-flatbed-wood.glb',
    'hazard.train',
  ],
  [
    'train-kit',
    'train-carriage-lumber.glb',
    'hazard.train',
  ],
  [
    'train-kit',
    'train-carriage-tank-large.glb',
    'hazard.train',
  ],
  [
    'train-kit',
    'train-carriage-wood.glb',
    'hazard.train',
  ],

  // Nature Kit (no palms selected)
  [
    'nature-kit', 'log.glb', 'hazard.log',
    'Models/GLTF format/log.glb',
  ],
  [
    'nature-kit', 'log-large.glb', 'hazard.log',
    'Models/GLTF format/log_large.glb',
  ],
  [
    'nature-kit', 'tree-default.glb', 'decoration.tree',
    'Models/GLTF format/tree_default.glb',
  ],
  [
    'nature-kit', 'tree-oak.glb', 'decoration.tree',
    'Models/GLTF format/tree_oak.glb',
  ],
  [
    'nature-kit', 'tree-pine-round-a.glb', 'decoration.tree',
    'Models/GLTF format/tree_pineRoundA.glb',
  ],
  [
    'nature-kit', 'tree-fat.glb', 'decoration.tree',
    'Models/GLTF format/tree_fat.glb',
  ],
  [
    'nature-kit', 'tree-simple-dark.glb', 'decoration.tree',
    'Models/GLTF format/tree_simple_dark.glb',
  ],
  [
    'nature-kit', 'tree-oak-fall.glb', 'decoration.tree',
    'Models/GLTF format/tree_oak_fall.glb',
  ],
  [
    'nature-kit', 'rock-large-a.glb', 'decoration.rocks',
    'Models/GLTF format/rock_largeA.glb',
  ],
  [
    'nature-kit', 'rock-small-a.glb', 'decoration.rocks',
    'Models/GLTF format/rock_smallA.glb',
  ],
  [
    'nature-kit', 'rock-small-c.glb', 'decoration.rocks',
    'Models/GLTF format/rock_smallC.glb',
  ],
  [
    'nature-kit', 'plant-bush.glb', 'decoration.plant',
    'Models/GLTF format/plant_bush.glb',
  ],
  [
    'nature-kit', 'plant-bush-small.glb', 'decoration.plant',
    'Models/GLTF format/plant_bushSmall.glb',
  ],
];

function sha256(bytes) {
  return createHash('sha256')
    .update(bytes)
    .digest('hex');
}

async function assetEntry(
  pack,
  filename,
  role,
  upstreamPath = `Models/GLB format/${filename}`,
) {
  const metadata = PACKS[pack];

  const path =
    `assets/kenney/${pack}/${filename}`;

  const bytes =
    await readFile(
      new URL(path, publicRoot),
    );

  return {
    role,
    path,
    bytes: bytes.byteLength,
    sha256: sha256(bytes),
    source: metadata.source,
    archiveSha256:
      metadata.archiveSha256,
    upstreamPath,
    license: 'CC0-1.0',
  };
}

async function textureEntry(pack) {
  const metadata = PACKS[pack];

  const path =
    `assets/kenney/${pack}`
    + '/Textures/colormap.png';

  const bytes =
    await readFile(
      new URL(path, publicRoot),
    );

  return {
    path,
    bytes: bytes.byteLength,
    sha256: sha256(bytes),
    source: metadata.source,
    archiveSha256:
      metadata.archiveSha256,
    upstreamPath:
      'Models/GLB format/Textures/colormap.png',
    license: 'CC0-1.0',
  };
}

const assets = [];

for (const [
  pack,
  filename,
  role,
  upstreamPath,
] of ASSETS) {
  assets.push(
    await assetEntry(
      pack,
      filename,
      role,
      upstreamPath,
    ),
  );
}

const textures = [];

for (const [pack, metadata] of Object.entries(PACKS)) {
  if (metadata.textureless === true) continue;
  textures.push(
    await textureEntry(pack),
  );
}

const manifest = {
  version: 1,
  assets,
  textures,
};

await writeFile(
  new URL(
    'assets/kenney/manifest.json',
    publicRoot,
  ),
  `${JSON.stringify(
    manifest,
    null,
    2,
  )}\n`,
  'utf8',
);

console.log(
  `Kenney manifest generated: `
  + `${assets.length} GLB, `
  + `${textures.length} textures.`,
);