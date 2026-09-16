import assert from 'node:assert/strict';
import test from 'node:test';

import {
  decorationsForRow,
  isSceneryBlocked,
  obstacleColumnsForRow,
} from '../src/game/scenery.ts';

const laneKinds = ['grass', 'road', 'rail', 'river'];

test('Nature Kit scenery is deterministic and appears only on grass', () => {
  const first = decorationsForRow('visual-seed', 12, 'grass');
  const second = decorationsForRow('visual-seed', 12, 'grass');
  assert.deepEqual(first, second);
  assert.ok(first.length >= 7);

  for (const laneKind of laneKinds.filter((kind) => kind !== 'grass')) {
    assert.deepEqual(decorationsForRow('visual-seed', 12, laneKind), []);
  }
});

test('selected Nature Kit scenery never contains palms', () => {
  for (let row = -30; row <= 60; row += 1) {
    for (const decoration of decorationsForRow('no-palms', row, 'grass')) {
      assert.equal(decoration.role.toLowerCase().includes('palm'), false);
    }
  }
});

test('grass boundaries are densely marked by trees outside playable columns', () => {
  for (let row = 0; row <= 30; row += 1) {
    const boundary = decorationsForRow('boundary-audit', row, 'grass')
      .filter((item) => !item.blocking);
    assert.equal(boundary.length, 6);
    assert.ok(boundary.every((item) => item.role.startsWith('decoration.tree')));
    assert.ok(boundary.every((item) => Math.abs(item.column) >= 6));
  }
});

test('interior grass obstacles are sparse, never use column zero, and match collision lookup', () => {
  for (let row = -30; row <= 80; row += 1) {
    const columns = obstacleColumnsForRow('obstacle-audit', row, 'grass');
    const recovery = ((row - 3) % 7 + 7) % 7 === 6;
    const opening = row >= 0 && row < 3;
    if (recovery || opening) {
      assert.deepEqual(columns, []);
      continue;
    }
    assert.ok(columns.length >= 1 && columns.length <= 3);
    assert.equal(new Set(columns).size, columns.length);
    assert.ok(columns.every((column) => column !== 0 && Math.abs(column) <= 4));
    for (const column of columns) {
      assert.equal(isSceneryBlocked('obstacle-audit', row, 'grass', column), true);
    }
    assert.equal(isSceneryBlocked('obstacle-audit', row, 'grass', 0), false);
  }
});
