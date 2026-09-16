import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const publicRoot = new URL('../public/', import.meta.url);
const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const KENNEY_PATH_PATTERN = /^assets\/kenney\/[a-z0-9-]+\/[a-z0-9-]+\.glb$/;
const KENNEY_TEXTURE_PATTERN = /^assets\/kenney\/[a-z0-9-]+\/Textures\/colormap\.png$/;
const KENNEY_SOURCE_PATTERN = /^https:\/\/kenney\.nl\/assets\/[a-z0-9-]+$/;
const KENNEY_ROLES = new Set([
  'lane.road',
  'lane.rail',

  'decoration.traffic-light',
  'decoration.rail-warning-light',
  'decoration.road-sign',
  'decoration.street-light',
  'decoration.barrier',

  'decoration.tree',
  'decoration.rocks',
  'decoration.plant',

  'hazard.car',
  'hazard.truck',
  'hazard.train',
  'hazard.log',
]);
const TEXTURELESS_PACKS = new Set(['nature-kit']);
const MANIFEST_KEYS = new Set(['version', 'assets', 'textures']);
const ASSET_KEYS = new Set([
  'role',
  'path',
  'bytes',
  'sha256',
  'source',
  'archiveSha256',
  'upstreamPath',
  'license',
]);
const TEXTURE_KEYS = new Set([
  'path',
  'bytes',
  'sha256',
  'source',
  'archiveSha256',
  'upstreamPath',
  'license',
]);

function digest(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function requirePlainObject(value, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
}

function rejectUnexpectedKeys(value, expected, label) {
  for (const key of Object.keys(value)) {
    if (!expected.has(key)) {
      throw new Error(`${label} has unexpected key: ${key}`);
    }
  }
  for (const key of expected) {
    if (!(key in value)) {
      throw new Error(`${label} is missing key: ${key}`);
    }
  }
}

export async function validateKenneyManifest(manifest, root = publicRoot) {
  requirePlainObject(manifest, 'Kenney manifest');
  rejectUnexpectedKeys(manifest, MANIFEST_KEYS, 'Kenney manifest');

  if (manifest.version !== 1) {
    throw new Error('Kenney manifest version must be 1.');
  }
  if (
    !Array.isArray(manifest.assets)
    || manifest.assets.length !== 45
  ) {
    throw new Error(
      'Kenney manifest must contain exactly 45 assets.',
    );
  }
  if (!Array.isArray(manifest.textures) || manifest.textures.length !== 3) {
    throw new Error('Kenney manifest must contain exactly 3 required textures.');
  }

  const roles = new Set();
  const paths = new Set();

  for (const [index, asset] of manifest.assets.entries()) {
    const label = `Kenney asset ${index}`;
    requirePlainObject(asset, label);
    rejectUnexpectedKeys(asset, ASSET_KEYS, label);

    if (!KENNEY_ROLES.has(asset.role)) {
      throw new Error(`${label} has unknown role: ${asset.role}`);
    }
  
    roles.add(asset.role);

    if (typeof asset.path !== 'string' || !KENNEY_PATH_PATTERN.test(asset.path)) {
      throw new Error(`${label} has an invalid local path.`);
    }
    if (paths.has(asset.path)) {
      throw new Error(`Kenney manifest has duplicate path: ${asset.path}`);
    }
    paths.add(asset.path);

    if (!Number.isInteger(asset.bytes) || asset.bytes <= 0) {
      throw new Error(`${label} has an invalid byte count.`);
    }
    if (typeof asset.sha256 !== 'string' || !SHA256_PATTERN.test(asset.sha256)) {
      throw new Error(`${label} has an invalid checksum.`);
    }
    if (
      typeof asset.archiveSha256 !== 'string'
      || !SHA256_PATTERN.test(asset.archiveSha256)
    ) {
      throw new Error(`${label} has an invalid archive checksum.`);
    }
    if (typeof asset.source !== 'string' || !KENNEY_SOURCE_PATTERN.test(asset.source)) {
      throw new Error(`${label} must use an official Kenney source URL.`);
    }
    if (
      typeof asset.upstreamPath !== 'string'
      || !/^Models\/(?:GLB|GLTF) format\/[A-Za-z0-9_-]+\.glb$/.test(asset.upstreamPath)
    ) {
      throw new Error(`${label} has an invalid upstream path.`);
    }
    if (asset.license !== 'CC0-1.0') {
      throw new Error(`${label} must declare CC0-1.0.`);
    }

    const assetUrl = new URL(asset.path, root);
    if (!assetUrl.href.startsWith(root.href)) {
      throw new Error(`${label} path escapes the public asset root.`);
    }
    const bytes = await readFile(assetUrl);
    if (bytes.byteLength !== asset.bytes) {
      throw new Error(`${label} byte count does not match the manifest.`);
    }
    if (digest(bytes) !== asset.sha256) {
      throw new Error(`${label} checksum does not match the manifest.`);
    }
  }

  for (const role of KENNEY_ROLES) {
    if (!roles.has(role)) {
      throw new Error(
        `Kenney manifest is missing role: ${role}`,
      );
    }
  }

  const texturePaths = new Set();
  for (const [index, texture] of manifest.textures.entries()) {
    const label = `Kenney texture ${index}`;
    requirePlainObject(texture, label);
    rejectUnexpectedKeys(texture, TEXTURE_KEYS, label);

    if (
      typeof texture.path !== 'string'
      || !KENNEY_TEXTURE_PATTERN.test(texture.path)
    ) {
      throw new Error(`${label} has an invalid local path.`);
    }
    if (texturePaths.has(texture.path)) {
      throw new Error(`Kenney manifest has duplicate texture path: ${texture.path}`);
    }
    texturePaths.add(texture.path);

    if (!Number.isInteger(texture.bytes) || texture.bytes <= 0) {
      throw new Error(`${label} has an invalid byte count.`);
    }
    if (typeof texture.sha256 !== 'string' || !SHA256_PATTERN.test(texture.sha256)) {
      throw new Error(`${label} has an invalid checksum.`);
    }
    if (
      typeof texture.archiveSha256 !== 'string'
      || !SHA256_PATTERN.test(texture.archiveSha256)
    ) {
      throw new Error(`${label} has an invalid archive checksum.`);
    }
    if (
      typeof texture.source !== 'string'
      || !KENNEY_SOURCE_PATTERN.test(texture.source)
    ) {
      throw new Error(`${label} must use an official Kenney source URL.`);
    }
    if (texture.upstreamPath !== 'Models/GLB format/Textures/colormap.png') {
      throw new Error(`${label} has an invalid upstream path.`);
    }
    if (texture.license !== 'CC0-1.0') {
      throw new Error(`${label} must declare CC0-1.0.`);
    }

    const textureUrl = new URL(texture.path, root);
    if (!textureUrl.href.startsWith(root.href)) {
      throw new Error(`${label} path escapes the public asset root.`);
    }
    const bytes = await readFile(textureUrl);
    if (bytes.byteLength !== texture.bytes) {
      throw new Error(`${label} byte count does not match the manifest.`);
    }
    if (digest(bytes) !== texture.sha256) {
      throw new Error(`${label} checksum does not match the manifest.`);
    }
  }

  for (const asset of manifest.assets) {
    const pack = asset.path.split('/')[2];
    if (TEXTURELESS_PACKS.has(pack)) continue;
    const directory = asset.path.slice(0, asset.path.lastIndexOf('/'));
    const requiredTexture = `${directory}/Textures/colormap.png`;
    if (!texturePaths.has(requiredTexture)) {
      throw new Error(`Kenney asset is missing its required texture: ${requiredTexture}`);
    }
  }
}

async function verifyAnatomicalAssets() {
  for (const [directory, manifestFile, field] of [
    ['brain-atlas', 'manifest.json', 'exportSha256'],
    ['flybody', 'checksums.json', 'sha256'],
  ]) {
    const manifest = JSON.parse(
      await readFile(new URL(`data/${directory}/${manifestFile}`, publicRoot), 'utf8'),
    );
    for (const [name, expectedDigest] of Object.entries(manifest[field])) {
      const bytes = await readFile(new URL(`data/${directory}/${name}`, publicRoot));
      if (digest(bytes) !== expectedDigest) {
        throw new Error(`Asset checksum mismatch: ${directory}/${name}`);
      }
    }
  }
}

async function verifyConnectomeAssets() {
  const manifest = JSON.parse(
    await readFile(new URL('data/connectome/manifest.json', publicRoot), 'utf8'),
  );
  const bytes = await readFile(new URL('data/connectome/graph.json', publicRoot));
  if (digest(bytes) !== manifest.graphSha256) {
    throw new Error('Asset checksum mismatch: connectome/graph.json');
  }

  const notice = await readFile(new URL('data/connectome/NOTICE.md', publicRoot), 'utf8');
  const protocolPath = 'docs/experiments/reduced-connectome-v1.md';
  if (!notice.includes(protocolPath)) {
    throw new Error(`Connectome notice must link to ${protocolPath}`);
  }
  await readFile(new URL(`../${protocolPath}`, publicRoot));

  const upstreamBuilder = 'https://github.com/cobanov/flyjump/blob/c08c86bc18efd8125964b1d2ca4fc1df59700f30/scripts/build-connectome.py';
  if (!notice.includes(upstreamBuilder)) {
    throw new Error('Connectome notice must pin its upstream derivation script.');
  }

  const bundledPolicy = await readFile(
    new URL('models/reduced-connectome-policy-v3.json', publicRoot),
  );
  const releasedPolicy = await readFile(
    new URL('../release/eval-v1/training/connectome/policy.json', publicRoot),
  );
  if (digest(bundledPolicy) !== digest(releasedPolicy)) {
    throw new Error('Bundled autoplay policy must match the released connectome policy.');
  }
}

export async function checkAssets() {
  await verifyAnatomicalAssets();
  await verifyConnectomeAssets();
  const kenneyManifest = JSON.parse(
    await readFile(new URL('assets/kenney/manifest.json', publicRoot), 'utf8'),
  );
  await validateKenneyManifest(kenneyManifest, publicRoot);
}

const isMain = process.argv[1]
  && pathToFileURL(process.argv[1]).href === import.meta.url;

if (isMain) {
  await checkAssets();
  console.log('Anatomical, connectome, and Kenney asset hashes verified.');
}
