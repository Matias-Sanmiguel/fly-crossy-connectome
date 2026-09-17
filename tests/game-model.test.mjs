import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import { parsePolicy, runDenseNetwork, runFixedGraphNetwork } from '../src/game/model.ts';

const fixture = JSON.parse(await readFile(new URL('./fixtures/policy-v1.json', import.meta.url)));
const visibleIds = new Set([101, 102, 103]);
const fixedSource = {
  kind: 'predicted',
  name: 'Fixed graph fixture',
  normalization: 'Raw tanh activity mapped to [0, 1]',
  datasetVersion: 'MaleCNS fixture v1',
  sourceUrl: 'https://example.test/malecns',
  license: 'CC BY 4.0',
  selectionRule: 'Declared fixture cells.',
  graphSourceHash: '1'.repeat(64),
  graphArtifactHash: '2'.repeat(64),
};

test('policy parser preserves historical v1 and current v2 observation versions', () => {
  assert.throws(() => parsePolicy({ version: 99 }, visibleIds), /version 1/i);
  assert.equal(parsePolicy(fixture, visibleIds).observationVersion, 1);
  assert.equal(
    parsePolicy({ ...fixture, observationVersion: 2 }, visibleIds).observationVersion,
    2,
  );
  assert.throws(
    () => parsePolicy({ ...fixture, observationVersion: 3 }, visibleIds),
    /observation version/i,
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
    source: fixedSource,
    network: {
      kind: 'fixed-graph',
      inputSize: 3,
      bodyIds: [101, 102],
      activation: 'tanh',
      sensoryWeights: [1, 0, 0, 0, 1, 0],
      recurrentSource: [0],
      recurrentTarget: [1],
      recurrentWeights: [0.5],
      recurrentGain: 1,
      timeConstant: 1,
      actorWeights: Array(10).fill(0),
      actorBias: [0, 0, 0, 0, 0]
    },
    activityBodyIds: [101, 102]
  };

  assert.equal(parsePolicy(fixed, visibleIds).network.kind, 'fixed-graph');
  fixed.network.recurrentTarget = [3];
  assert.throws(() => parsePolicy(fixed, visibleIds), /topology/i);
});

test('policy parser rejects unsafe fixed graph dynamics', () => {
  const fixedFixture = structuredClone({
    ...fixture,
    source: fixedSource,
    network: {
      kind: 'fixed-graph', inputSize: 3, bodyIds: [101], activation: 'tanh',
      sensoryWeights: [0, 0, 0], recurrentSource: [], recurrentTarget: [],
      recurrentWeights: [], recurrentGain: 1, timeConstant: 1,
      actorWeights: Array(5).fill(0), actorBias: Array(5).fill(0),
    },
    activityBodyIds: [101],
  });

  fixedFixture.network.timeConstant = 0;
  assert.throws(() => parsePolicy(fixedFixture, visibleIds), /time constant/i);
  fixedFixture.network.timeConstant = 1;
  fixedFixture.network.recurrentGain = Number.NaN;
  assert.throws(() => parsePolicy(fixedFixture, visibleIds), /recurrent gain/i);
});

test('browser fixed graph one-step inference matches the Python fixture', async () => {
  const fixedFixture = JSON.parse(await readFile(
    new URL('./fixtures/fixed-graph-policy-v1.json', import.meta.url),
  ));
  const policy = parsePolicy(fixedFixture, visibleIds);
  assert.equal(policy.network.kind, 'fixed-graph');

  const result = runFixedGraphNetwork(
    policy.network,
    fixedFixture.parity.input,
    fixedFixture.parity.hidden,
  );

  result.logits.forEach((value, index) => {
    assert.ok(Math.abs(value - fixedFixture.parity.logits[index]) <= 1e-5);
  });
  result.activity.forEach((value, index) => {
    assert.ok(Math.abs(value - fixedFixture.parity.activity[index]) <= 1e-5);
  });
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
