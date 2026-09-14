import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import { parsePolicy, runDenseNetwork } from '../src/game/model.ts';

const fixture = JSON.parse(await readFile(new URL('./fixtures/policy-v1.json', import.meta.url)));
const visibleIds = new Set([101, 102, 103]);

test('policy parser rejects incompatible schema versions', () => {
  assert.throws(() => parsePolicy({ version: 99 }, visibleIds), /version 1/i);
  assert.throws(
    () => parsePolicy({ ...fixture, observationVersion: 2 }, visibleIds),
    /observation version 1/i,
  );
});

test('policy parser accepts a strictly shaped dense policy', () => {
  const policy = parsePolicy(fixture, visibleIds);

  assert.equal(policy.version, 1);
  assert.equal(policy.network.kind, 'dense');
  assert.equal(policy.source.name, 'Conventional PPO baseline');
  assert.match(policy.source.checkpointHash, /^[a-f0-9]{64}$/);
});

test('policy parser rejects unknown MaleCNS activity IDs and duplicates', () => {
  assert.throws(
    () => parsePolicy({ ...fixture, activityBodyIds: [999] }, visibleIds),
    /visible MaleCNS/i,
  );
  assert.throws(
    () => parsePolicy({ ...fixture, activityBodyIds: [101, 101] }, visibleIds),
    /duplicate/i,
  );
});

test('policy parser enforces canonical actions and source provenance', () => {
  assert.throws(
    () => parsePolicy({ ...fixture, actions: ['wait', 'forward', 'backward', 'left', 'right'] }, visibleIds),
    /action order/i,
  );
  assert.throws(
    () => parsePolicy({ ...fixture, source: { ...fixture.source, normalization: ' ' } }, visibleIds),
    /source/i,
  );
});

test('policy parser rejects malformed layer shapes, activations, and numeric values', () => {
  const badShape = structuredClone(fixture);
  badShape.network.layers[0].weights.pop();
  assert.throws(() => parsePolicy(badShape, visibleIds), /weights/i);

  const badActivation = structuredClone(fixture);
  badActivation.network.layers[0].activation = 'sigmoid';
  assert.throws(() => parsePolicy(badActivation, visibleIds), /activation/i);

  const badNumber = structuredClone(fixture);
  badNumber.network.layers[0].bias[0] = Number.POSITIVE_INFINITY;
  assert.throws(() => parsePolicy(badNumber, visibleIds), /finite/i);
});

test('policy parser validates fixed graph topology and array shapes', () => {
  const fixed = {
    ...fixture,
    network: {
      kind: 'fixed-graph',
      inputSize: 3,
      bodyIds: [101, 102],
      activation: 'tanh',
      sensoryWeights: [1, 0, 0, 0, 1, 0],
      recurrentSource: [0],
      recurrentTarget: [1],
      recurrentWeights: [0.5],
      actorWeights: Array(10).fill(0),
      actorBias: [0, 0, 0, 0, 0]
    },
    activityBodyIds: [101, 102]
  };

  assert.equal(parsePolicy(fixed, visibleIds).network.kind, 'fixed-graph');
  fixed.network.recurrentTarget = [3];
  assert.throws(() => parsePolicy(fixed, visibleIds), /topology/i);
});

test('browser dense logits match the fixed Python export fixture', () => {
  const policy = parsePolicy(fixture, visibleIds);
  assert.equal(policy.network.kind, 'dense');

  const { output } = runDenseNetwork(policy.network, fixture.parity.input);

  assert.equal(output.length, fixture.parity.logits.length);
  output.forEach((value, index) => {
    assert.ok(Math.abs(value - fixture.parity.logits[index]) <= 1e-5);
  });
});
