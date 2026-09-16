import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';

import {
  createTiledLaneRow,
  railSurfaceMode,
} from '../src/game/laneSurface.ts';

const libraryWith = (availableRole) => ({
  templates: new Map([[availableRole, new THREE.Group()]]),
  failures: new Map(),
  clone(role) {
    if (role !== availableRole) return null;
    const clone = new THREE.Group();
    clone.name = role;
    return clone;
  },
});

test('Kenney road composite contains exactly 25 pooled-row tiles at integer positions', () => {
  const row = createTiledLaneRow('lane.road', libraryWith('lane.road'), 25);

  assert.ok(row);
  assert.equal(row.name, 'tiled-row:lane.road');
  assert.equal(row.children.length, 25);
  assert.deepEqual(
    row.children.map((tile) => tile.position.x),
    Array.from({ length: 25 }, (_, index) => index - 12),
  );
});

test('tiled lane construction cleanly reports an unavailable asset', () => {
  assert.equal(
    createTiledLaneRow('lane.road', libraryWith('lane.rail'), 25),
    null,
  );
});

test('Kenney rail composite reuses the same exact 25-tile layout', () => {
  const row = createTiledLaneRow('lane.rail', libraryWith('lane.rail'), 25);

  assert.ok(row);
  assert.equal(row.name, 'tiled-row:lane.rail');
  assert.equal(row.children.length, 25);
  assert.deepEqual(
    row.children.map((tile) => tile.position.x),
    Array.from({ length: 25 }, (_, index) => index - 12),
  );
});

test('rail detail mode never enables Kenney and procedural tracks together', () => {
  assert.equal(railSurfaceMode(true), 'kenney');
  assert.equal(railSurfaceMode(false), 'procedural');
});
