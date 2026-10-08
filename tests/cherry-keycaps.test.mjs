import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { createCherryKeycapLabel, createCherryKeycapLibrary, CHERRY_KEYCAP_NAMES } from '../src/expo/cherryKeycaps.ts';

const path = new URL('../public/assets/expo/keycaps/', import.meta.url);
async function readAsset() {
  const bytes = await readFile(new URL('cherry-wasd.glb', path));
  const gltf = await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), '');
  return { bytes, scene: gltf.scene };
}

test('Cherry asset keeps MIT attribution, exact hash and only the two approved 1u profiles', async () => {
  const manifest = JSON.parse(await readFile(new URL('manifest.json', path), 'utf8'));
  const license = await readFile(new URL('LICENSE', path), 'utf8');
  const { bytes, scene } = await readAsset();
  assert.match(license, /MIT License/);
  assert.match(license, /2020 Santiago Castelo/);
  assert.equal(manifest.sourceUrl, 'https://github.com/endeavoursc/cherry-mx-keycaps');
  assert.equal(createHash('sha256').update(bytes).digest('hex'), manifest.output.sha256);
  assert.ok(bytes.length < 200_000);
  assert.deepEqual(manifest.models.map(v => v.sourceName), ['R2_1u', 'R3_1u']);
  const meshes = [];
  scene.traverse(object => { if (object.isMesh) meshes.push(object) });
  assert.equal(meshes.length, 2);
  for (const mesh of meshes) {
    mesh.geometry.computeBoundingBox();
    const size = mesh.geometry.boundingBox.getSize(new THREE.Vector3());
    assert.ok(Math.abs(size.x - 0.086) < 1e-6 && Math.abs(size.z - 0.086) < 1e-6);
    assert.ok(size.y > 0.03 && size.y < 0.04);
    assert.ok(mesh.geometry.index.count / 3 < 2500);
    assert.ok(mesh.geometry.boundingBox.getCenter(new THREE.Vector3()).length() < 1e-6);
  }
});

test('one cached load reuses the R3 geometry for A/S/D and preserves independent key groups', async () => {
  const { scene } = await readAsset();
  let calls = 0;
  const load = createCherryKeycapLibrary(async () => { calls++; return scene });
  const [first, second] = await Promise.all([load(), load()]);
  assert.equal(calls, 1);
  assert.equal(first, second);
  assert.equal(first.a, first.s);
  assert.equal(first.s, first.d);
  assert.notEqual(first.w, first.a);
  assert.equal(CHERRY_KEYCAP_NAMES.w, 'cherry-r2-1u');
  const a = new THREE.Mesh(first.a), s = new THREE.Mesh(first.s);
  a.position.y = 0.036;
  s.position.y = 0.05;
  assert.notEqual(a.position.y, s.position.y);
});

test('letters follow the actual curved cap surface and remain finite', async () => {
  const { scene } = await readAsset();
  const caps = await createCherryKeycapLibrary(async () => scene)();
  for (const geometry of [caps.w, caps.a]) {
    const label = createCherryKeycapLabel(geometry);
    const positions = label.attributes.position;
    const heights = [];
    const surface = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
    const ray = new THREE.Raycaster();
    for (let i = 0; i < positions.count; i++) {
      const point = new THREE.Vector3().fromBufferAttribute(positions, i);
      assert.ok(point.toArray().every(Number.isFinite));
      ray.set(new THREE.Vector3(point.x, 0.1, point.z), new THREE.Vector3(0, -1, 0));
      const hit = ray.intersectObject(surface, false)[0];
      assert.ok(hit);
      assert.ok(Math.abs(point.y - hit.point.y - 0.0006) < 1e-7);
      heights.push(point.y);
    }
    assert.ok(Math.max(...heights) - Math.min(...heights) > 0.0001);
    label.dispose();
    surface.material.dispose();
  }
});

test('missing or failed assets reject cleanly for the procedural fallback, without a retry loop', async () => {
  let calls = 0;
  const failing = createCherryKeycapLibrary(async () => { calls++; throw Error('offline') });
  await assert.rejects(failing(), /offline/);
  await assert.rejects(failing(), /offline/);
  assert.equal(calls, 1);
  await assert.rejects(createCherryKeycapLibrary(async () => new THREE.Group())(), /Missing Cherry/);
});
