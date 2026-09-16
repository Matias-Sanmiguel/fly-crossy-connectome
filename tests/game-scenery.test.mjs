import assert from 'node:assert/strict';
import test from 'node:test';

import { decorationsForRow } from '../src/game/scenery.ts';

const rolesByLane = {
  grass: new Set(['decoration.tree', 'decoration.rocks', 'decoration.plant']),
  road: new Set(),
  rail: new Set(),
  river: new Set(['decoration.rocks', 'decoration.plant']),
};

test('scenery is deterministic and leaves the center five columns clear', () => {
  const first = decorationsForRow('visual-seed', 12, 'grass');
  const second = decorationsForRow('visual-seed', 12, 'grass');

  assert.deepEqual(first, second);
  assert.ok(first.every((item) => Math.abs(item.column) > 2));
});

test('scenery stays bounded, finite, and appropriate to each lane', () => {
  for (const laneKind of Object.keys(rolesByLane)) {
    for (let row = -20; row <= 40; row += 1) {
      const decorations = decorationsForRow('bounded-seed', row, laneKind);
      assert.ok(decorations.length <= 2);
      for (const item of decorations) {
        assert.ok(rolesByLane[laneKind].has(item.role));
        assert.ok(Number.isFinite(item.column));
        assert.ok(Number.isFinite(item.rotationY));
        assert.ok(Number.isFinite(item.scale) && item.scale > 0);
        assert.ok(Math.abs(item.column) > 2 && Math.abs(item.column) <= 11);
      }
    }
  }
});

test('different rows do not collapse to one decoration layout', () => {
  const layouts = new Set(
    Array.from({ length: 12 }, (_, row) => (
      JSON.stringify(decorationsForRow('varied-seed', row, 'grass'))
    )),
  );

  assert.ok(layouts.size > 4);
});

test('road and rail rows never receive generic traffic-light scenery', () => {
  for (const laneKind of ['road', 'rail']) {
    for (let row = -100; row <= 100; row += 1) {
      assert.deepEqual(decorationsForRow('traffic-light-audit', row, laneKind), []);
    }
  }
});
