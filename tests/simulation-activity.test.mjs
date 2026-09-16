import assert from 'node:assert/strict';
import test from 'node:test';

import { neuralActivityFrame } from '../src/simulation/activity.ts';

test('neural activity converts body IDs without inventing values', () => {
  assert.equal(neuralActivityFrame(null), null);
  assert.deepEqual(
    neuralActivityFrame({ revision: 2, simulationTime: 0.5, updates: [] }),
    { time: 0.5, values: [] },
  );
  assert.deepEqual(
    neuralActivityFrame({
      revision: 3,
      simulationTime: 1.25,
      updates: [{ neuronId: 42, value: 0.75 }],
    }),
    { time: 1.25, values: [[42, 0.75]] },
  );
});

test('duplicate IDs and invalid numbers fail closed', () => {
  assert.equal(neuralActivityFrame({
    revision: 1,
    simulationTime: 1,
    updates: [
      { neuronId: 42, value: 0.25 },
      { neuronId: 42, value: 0.75 },
    ],
  }), null);
  assert.equal(neuralActivityFrame({
    revision: 1,
    simulationTime: Number.NaN,
    updates: [],
  }), null);
  assert.equal(neuralActivityFrame({
    revision: 1,
    simulationTime: 1,
    updates: [{ neuronId: 42, value: Number.POSITIVE_INFINITY }],
  }), null);
  assert.equal(neuralActivityFrame({
    revision: 1,
    simulationTime: 1,
    updates: [{ neuronId: -1, value: 0.5 }],
  }), null);
});
