import test from 'node:test';
import assert from 'node:assert/strict';

import { observe } from '../src/game/observation.ts';

const baseState = () => ({
  version: 1,
  seed: 'observation-fixture',
  step: 4,
  time: 0,
  fly: { row: 0, column: 0 },
  score: 2,
  terminal: null,
  previousAction: 'left',
  lanes: [
    {
      row: 0,
      kind: 'road',
      direction: -1,
      speed: 3,
      phase: 0,
      hazards: [{ kind: 'truck', position: 0, size: 1 }],
    },
    {
      row: 1,
      kind: 'river',
      direction: 1,
      speed: 1,
      phase: 0,
      hazards: [{ kind: 'log', position: 1, size: 1 }],
    },
    {
      row: 6,
      kind: 'rail',
      direction: 1,
      speed: 3,
      phase: 0,
      hazards: [{ kind: 'train', position: 0, size: 4 }],
    },
  ],
});

test('observation is a fixed radius-five egocentric window', () => {
  const observation = observe(baseState());

  assert.equal(observation.version, 2);
  assert.equal(observation.radius, 5);
  assert.equal(observation.cells.length, 11);
  assert.ok(observation.cells.every((row) => row.length === 11));
  assert.equal(observation.motion.length, 11);
  assert.ok(observation.motion.every((row) => row.length === 11));
  assert.ok(observation.motion.flat().every((vector) => vector.length === 2));
});

test('observation encodes local occupancy and normalized motion only', () => {
  const observation = observe(baseState());

  assert.equal(observation.cells[5][5], 5, 'vehicle occupancy');
  assert.deepEqual(observation.motion[5][5], [-1, 3 / 5]);
  assert.equal(observation.cells[6][6], 7, 'log occupancy');
  assert.deepEqual(observation.motion[6][6], [1, 1 / 5]);
  assert.equal(observation.cells.flat().includes(6), false, 'row six is outside the radius-five window');
});

test('observation reports previous action and normalized edge distance', () => {
  const state = baseState();
  state.fly.column = 3;

  const observation = observe(state);

  assert.equal(observation.previousAction, 'left');
  assert.equal(observation.edgeDistance, 0.4);
});

test('support is set only while a river platform holds the fly', () => {
  const state = baseState();
  state.fly.row = 1;
  state.fly.column = 1;

  assert.equal(observe(state).support, 1);
  state.fly.column = 3;
  assert.equal(observe(state).support, 0);
});
