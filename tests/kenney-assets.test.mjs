import assert from 'node:assert/strict';
import {
  appendFile,
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rm,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import test from 'node:test';

import { validateKenneyManifest } from '../scripts/check-assets.mjs';

const root = new URL('../', import.meta.url);
const expectedRoles = [
  'decoration.barrier',
  'decoration.plant',
  'decoration.rail-warning-light',
  'decoration.road-sign',
  'decoration.rocks',
  'decoration.street-light',
  'decoration.traffic-light',
  'decoration.tree',
  'hazard.car',
  'hazard.log',
  'hazard.train',
  'hazard.truck',
  'lane.rail',
  'lane.road',
];

test('Kenney manifest pins exactly the approved CC0 inventory', async () => {
  const manifest = JSON.parse(
    await readFile(new URL('public/assets/kenney/manifest.json', root), 'utf8'),
  );

  assert.equal(manifest.version, 1);
  assert.equal(manifest.assets.length, 43);
  assert.equal(manifest.textures.length, 4);
  assert.deepEqual(
    [...new Set(
      manifest.assets.map(
        (item) => item.role,
      ),
    )].sort(),
    expectedRoles,
  );
  assert.equal(new Set(manifest.assets.map((item) => item.path)).size, 43);

  for (const item of manifest.assets) {
    assert.match(item.path, /^assets\/kenney\/.+\.glb$/);
    assert.match(item.source, /^https:\/\/kenney\.nl\/assets\//);
    assert.match(item.sha256, /^[a-f0-9]{64}$/);
    assert.match(item.archiveSha256, /^[a-f0-9]{64}$/);
    assert.ok(Number.isInteger(item.bytes) && item.bytes > 0);
    assert.equal(item.license, 'CC0-1.0');
  }

  assert.equal(new Set(manifest.textures.map((item) => item.path)).size, 4);
  for (const item of manifest.textures) {
    assert.match(item.path, /^assets\/kenney\/.+\/Textures\/colormap\.png$/);
    assert.match(item.sha256, /^[a-f0-9]{64}$/);
    assert.ok(Number.isInteger(item.bytes) && item.bytes > 0);
    assert.equal(item.license, 'CC0-1.0');
  }
});

test('Kenney validator rejects schema and inventory mutations', async () => {
  const manifest = JSON.parse(
    await readFile(new URL('public/assets/kenney/manifest.json', root), 'utf8'),
  );
  const publicRoot = new URL('public/', root);
  const clone = () => structuredClone(manifest);

  const extraKey = clone();
  extraKey.assets[0].unexpected = true;
  await assert.rejects(
    validateKenneyManifest(extraKey, publicRoot),
    /unexpected key/i,
  );

  const escapedPath = clone();
  escapedPath.assets[0].path = '../outside.glb';
  await assert.rejects(
    validateKenneyManifest(escapedPath, publicRoot),
    /path/i,
  );

  const unknownRole = clone();

  unknownRole.assets[0].role =
    'hazard.unknown';

  await assert.rejects(
    validateKenneyManifest(
      unknownRole,
      publicRoot,
    ),
    /unknown role/i,
  );

  const duplicatePath = clone();
  duplicatePath.assets[1].path = duplicatePath.assets[0].path;
  await assert.rejects(
    validateKenneyManifest(duplicatePath, publicRoot),
    /duplicate path/i,
  );

  const shortInventory = clone();
  shortInventory.assets.pop();
  await assert.rejects(
    validateKenneyManifest(shortInventory, publicRoot),
    /exactly 43/i,
  );

  const wrongBytes = clone();
  wrongBytes.assets[0].bytes += 1;
  await assert.rejects(
    validateKenneyManifest(wrongBytes, publicRoot),
    /byte count/i,
  );
});

test('Kenney validator detects a tampered GLB in an isolated copy', async (t) => {
  const manifest = JSON.parse(
    await readFile(new URL('public/assets/kenney/manifest.json', root), 'utf8'),
  );
  const temporaryRoot = await mkdtemp(join(tmpdir(), 'fly-crossy-kenney-'));
  t.after(() => rm(temporaryRoot, { recursive: true, force: true }));

  for (const file of [...manifest.assets, ...manifest.textures]) {
    const source = fileURLToPath(new URL(`public/${file.path}`, root));
    const destination = join(temporaryRoot, file.path);
    await mkdir(dirname(destination), { recursive: true });
    await copyFile(source, destination);
  }

  const firstAsset = join(temporaryRoot, manifest.assets[0].path);
  await appendFile(firstAsset, new Uint8Array([0]));

  await assert.rejects(
    validateKenneyManifest(manifest, pathToFileURL(`${temporaryRoot}/`)),
    /(byte count|checksum)/i,
  );
});
