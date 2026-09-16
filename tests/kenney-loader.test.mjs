import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';

import {
  KENNEY_ASSETS,
  NATIVE_ASSET_ROLES,
} from '../src/game/kenneyAssets.ts';
import { loadKenneyAssets } from '../src/game/kenneyLoader.ts';

test('semantic asset registry pins mappings and finite transforms', () => {
  assert.equal(
    KENNEY_ASSETS['hazard.car.sedan'].path,
    'assets/kenney/car-kit/sedan.glb',
  );
  assert.equal(
    KENNEY_ASSETS['hazard.train'].path,
    'assets/kenney/train-kit/train-diesel-a.glb',
  );
  assert.equal(
    KENNEY_ASSETS['decoration.rail-warning-light'].path,
    'assets/kenney/city-roads/road-sign-stop.glb',
  );
  assert.deepEqual(KENNEY_ASSETS['lane.road'].logicalSize, [1, 0.2, 1]);
  assert.deepEqual(KENNEY_ASSETS['lane.rail'].logicalSize, [1, 0.18, 1]);
  assert.equal(NATIVE_ASSET_ROLES['hazard.log'], undefined);
  assert.equal(
    KENNEY_ASSETS['hazard.log.small'].path,
    'assets/kenney/survival-kit/tree-log-small.glb',
  );
  assert.equal(
    KENNEY_ASSETS['hazard.log.large'].path,
    'assets/kenney/survival-kit/tree-log.glb',
  );

  for (const asset of Object.values(KENNEY_ASSETS)) {
    assert.equal(asset.logicalSize.length, 3);
    assert.ok(asset.logicalSize.every((value) => Number.isFinite(value) && value > 0));
    assert.equal(asset.rotation.length, 3);
    assert.ok(asset.rotation.every(Number.isFinite));
    assert.ok(Number.isFinite(asset.verticalOffset));
  }
});

test('loader caches each URL, clones templates, and contains partial failure', async () => {
  const calls = new Map();
  const fakeLoader = async (url) => {
    calls.set(url, (calls.get(url) ?? 0) + 1);
    if (url.includes('train-diesel-a')) throw new Error('offline');

    const group = new THREE.Group();
    group.add(new THREE.Mesh(new THREE.BoxGeometry(2, 1, 4)));
    return group;
  };

  const firstLibrary = await loadKenneyAssets(fakeLoader);
  const secondLibrary = await loadKenneyAssets(fakeLoader);

  assert.equal(firstLibrary.failures.get('hazard.train'), 'offline');
  assert.equal(firstLibrary.templates.has('hazard.car.sedan'), true);
  assert.equal(secondLibrary.templates.has('hazard.car.sedan'), true);
  assert.equal(calls.size, Object.keys(KENNEY_ASSETS).length);
  assert.ok([...calls.values()].every((count) => count === 1));

  const carTemplate = firstLibrary.templates.get('hazard.car.sedan');
  assert.ok(carTemplate);
  assert.deepEqual(carTemplate.position.toArray(), [0, 0, 0]);
  const carBounds = new THREE.Box3().setFromObject(carTemplate);
  const carCenter = carBounds.getCenter(new THREE.Vector3());
  assert.ok(Math.abs(carCenter.x) < 1e-9);
  assert.ok(Math.abs(carCenter.z) < 1e-9);
  assert.ok(Math.abs(carBounds.min.y) < 1e-9);

  const firstCar = firstLibrary.clone('hazard.car.sedan');
  const secondCar = firstLibrary.clone('hazard.car.sedan');
  assert.ok(firstCar);
  assert.ok(secondCar);
  assert.notEqual(firstCar, secondCar);
  assert.notEqual(firstCar.children[0], secondCar.children[0]);
  const secondCarX = secondCar.position.x;
  firstCar.position.x = 12;
  assert.equal(secondCar.position.x, secondCarX);
  assert.equal(firstLibrary.clone('hazard.train'), null);
});

test('loader rejects empty geometry per role without rejecting the library', async () => {
  const library = await loadKenneyAssets(async () => new THREE.Group());

  assert.equal(library.templates.size, 0);
  assert.equal(library.failures.size, Object.keys(KENNEY_ASSETS).length);
  assert.match(library.failures.get('lane.road'), /geometry|bounds/i);
});

test('normalization preserves model proportions while fitting its visual envelope', async () => {
  const source = new THREE.Group();
  source.add(new THREE.Mesh(new THREE.BoxGeometry(2, 1, 4)));

  const library = await loadKenneyAssets(async () => source);
  const car = library.templates.get('hazard.car.sedan');
  assert.ok(car);

  const size = new THREE.Box3().setFromObject(car).getSize(new THREE.Vector3());
  assert.ok(Math.abs(size.x / size.y - 4) < 1e-9);
  assert.ok(Math.abs(size.z / size.y - 2) < 1e-9);
  assert.ok(size.x <= 1.8 + 1e-9);
  assert.ok(size.y <= 0.72 + 1e-9);
  assert.ok(size.z <= 0.9 + 1e-9);
});
