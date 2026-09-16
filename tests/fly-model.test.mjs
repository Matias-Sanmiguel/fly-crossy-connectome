import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import * as THREE from 'three';

const moduleUrl = new URL('../src/game/createFlyModel.ts', import.meta.url);
const { createFlyModel } = await import(moduleUrl);

function fakePreparedFly() {
  const source = new THREE.Group();
  source.name = 'world';

  const body = new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 2),
    new THREE.MeshStandardMaterial(),
  );
  body.name = 'body-dark';
  source.add(body);

  const left = new THREE.Group();
  left.name = 'wing-left-pivot';
  left.position.set(-0.14, 3.25, -2.3);
  left.add(new THREE.Mesh(
    new THREE.BoxGeometry(2, 0.05, 1),
    new THREE.MeshStandardMaterial(),
  ));
  source.add(left);

  const right = new THREE.Group();
  right.name = 'wing-right-pivot';
  right.position.set(0.14, 3.25, -2.3);
  right.add(new THREE.Mesh(
    new THREE.BoxGeometry(2, 0.05, 1),
    new THREE.MeshStandardMaterial(),
  ));
  source.add(right);

  return source;
}

test('prepared fly asset contains independently animatable wing pivots', async () => {
  const bytes = await readFile(
    new URL('../public/assets/fly/fly-jeremy-rigged.glb', import.meta.url),
  );
  assert.equal(bytes.subarray(0, 4).toString('ascii'), 'glTF');

  const jsonLength = bytes.readUInt32LE(12);
  const jsonType = bytes.readUInt32LE(16);
  assert.equal(jsonType, 0x4e4f534a);

  const gltf = JSON.parse(
    bytes.subarray(20, 20 + jsonLength).toString('utf8').trim(),
  );
  const names = new Set((gltf.nodes ?? []).map((node) => node.name));
  assert.ok(names.has('body-dark'));
  assert.ok(names.has('wing-left-pivot'));
  assert.ok(names.has('wing-right-pivot'));
  assert.ok(names.has('wing-left'));
  assert.ok(names.has('wing-right'));
});

test('fly opens during a hop and folds again at rest', async () => {
  let requestedUrl = '';
  const model = createFlyModel(async (url) => {
    requestedUrl = url;
    return fakePreparedFly();
  });
  await model.ready;

  assert.equal(requestedUrl, 'assets/fly/fly-jeremy-rigged.glb');
  assert.equal(model.group.name, 'fly-jeremy');

  const left = model.group.getObjectByName('wing-left-pivot');
  const right = model.group.getObjectByName('wing-right-pivot');
  assert.ok(left);
  assert.ok(right);

  model.update(0, 0, false);
  assert.ok(Math.abs(left.rotation.y) < 1e-9);
  assert.ok(Math.abs(right.rotation.y) < 1e-9);
  assert.ok(Math.abs(left.rotation.z) < 1e-9);
  assert.ok(Math.abs(right.rotation.z) < 1e-9);

  model.update(100, 0.5, true);
  assert.ok(Math.abs(left.rotation.y) < 1e-9);
  assert.ok(Math.abs(right.rotation.y) < 1e-9);
  assert.ok(Math.abs(left.rotation.z) > 0.1);
  assert.ok(Math.abs(right.rotation.z + left.rotation.z) < 1e-9);

  model.update(200, 1, true);
  assert.ok(Math.abs(left.rotation.y) < 1e-9);
  assert.ok(Math.abs(right.rotation.y) < 1e-9);
  assert.ok(Math.abs(left.rotation.z) < 1e-9);
  assert.ok(Math.abs(right.rotation.z) < 1e-9);

  model.dispose();
});
