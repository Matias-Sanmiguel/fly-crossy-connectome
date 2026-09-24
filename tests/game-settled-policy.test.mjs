import assert from 'node:assert/strict';
import test from 'node:test';

import { runFixedGraphNetwork } from '../src/game/model.ts';

const network = {
  kind: 'fixed-graph',
  interfaceMode: 'population',
  inputSize: 1,
  bodyIds: [101, 102],
  activation: 'tanh',
  sensoryWeights: [1, 0],
  recurrentSource: [0],
  recurrentTarget: [1],
  recurrentWeights: [1],
  recurrentGain: 1,
  timeConstant: 1,
  actorWeights: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  actorBias: [0, 0, 0, 0, 0],
};

test('two internal updates propagate current input to downstream readout', () => {
  const one = runFixedGraphNetwork(network, [1], [0, 0]);
  const two = runFixedGraphNetwork({ ...network, interfaceMode: 'population-settled', internalSteps: 2 }, [1], [0, 0]);
  assert.equal(one.logits[0], 0);
  assert.ok(two.logits[0] > 0);
});
