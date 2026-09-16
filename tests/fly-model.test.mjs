import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';

test('stylized fly has recognizable low-poly anatomy', async () => {
  const moduleUrl = new URL('../src/game/createFlyModel.ts', import.meta.url);
  const imported = await import(moduleUrl);
  const model = imported.createFlyModel();

  const named = [];
  model.group.traverse((object) => named.push(object.name));

  assert.equal(named.filter((name) => name.startsWith('fly-leg-')).length, 12);
  assert.equal(named.filter((name) => name.startsWith('fly-wing-')).length, 2);
  assert.equal(named.filter((name) => name.startsWith('fly-eye-')).length, 2);
  assert.equal(named.filter((name) => name.startsWith('fly-antenna-')).length, 2);
  assert.ok(named.includes('fly-thorax'));
  assert.ok(named.includes('fly-abdomen'));
  assert.ok(named.includes('fly-head'));

  model.group.updateMatrixWorld(true);
  const head = model.group.getObjectByName('fly-head');
  const abdomen = model.group.getObjectByName('fly-abdomen');
  assert.ok(head);
  assert.ok(abdomen);
  assert.ok(
    head.getWorldPosition(new THREE.Vector3()).z
      < abdomen.getWorldPosition(new THREE.Vector3()).z,
    'the fly must face the forward (negative-z) crossing direction',
  );

  const bounds = new THREE.Box3().setFromObject(model.group);
  assert.equal(bounds.isEmpty(), false);
  assert.ok(bounds.getSize(new THREE.Vector3()).toArray().every(Number.isFinite));

  model.dispose();
});
