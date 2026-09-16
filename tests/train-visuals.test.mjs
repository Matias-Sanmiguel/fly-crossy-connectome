import assert from 'node:assert/strict';
import test from 'node:test';

import { trainConsistParts } from '../src/game/trainVisuals.ts';

test('eighteen-unit trains render as one locomotive plus eight carriages', () => {
  const hazard = {
    kind: 'train',
    position: 4,
    size: 18,
  };

  const first = trainConsistParts('train-test', 12, hazard);
  const second = trainConsistParts('train-test', 12, hazard);

  assert.deepEqual(first, second);
  assert.equal(first.length, 9);
  assert.deepEqual(
    first.map((part) => part.offset),
    [8, 6, 4, 2, 0, -2, -4, -6, -8],
  );
  assert.match(first[0].role, /^hazard\.train(?:\.locomotive\.)?/);
  assert.ok(
    first.slice(1).every(
      (part) => /^hazard\.train\.carriage\./.test(part.role),
    ),
  );
});

test('train appearance varies deterministically across identities', () => {
  const appearances = new Set();

  for (let row = 0; row < 40; row += 1) {
    appearances.add(
      trainConsistParts(
        'train-variety',
        row,
        { kind: 'train', position: row - 20, size: 18 },
      ).map((part) => part.role).join('|'),
    );
  }

  assert.ok(appearances.size >= 6);
});