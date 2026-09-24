import assert from 'node:assert/strict';
import test from 'node:test';

import { runFixedGraphNetwork } from '../src/game/model.ts';

test('Controller V2 fuses learned action risk and route predictions at inference', () => {
  const nodes = 2;
  const actions = 5;
  const network = {
    kind: 'fixed-graph',
    interfaceMode: 'controller-v2',
    inputSize: 1,
    bodyIds: [101, 102],
    activation: 'tanh',
    sensoryWeights: [0, 0],
    recurrentSource: [],
    recurrentTarget: [],
    recurrentWeights: [],
    recurrentGain: 1,
    timeConstant: 1,
    actorWeights: Array(actions * nodes).fill(0),
    actorBias: Array(actions).fill(0),
    riskWeights: Array(actions * nodes).fill(0),
    riskBias: [10, 0, -10, 0, 0],
    routeWeights: Array(actions * nodes).fill(0),
    routeBias: [0, 0, 10, 0, 0],
    safetyGain: 4,
    routeGain: 0.75,
  };
  const result = runFixedGraphNetwork(network, [0], [0, 0]);
  assert.ok(result.risk[0] > 0.99);
  assert.ok(result.risk[2] < 0.01);
  assert.ok(result.route[2] > 0.99);
  assert.equal(result.logits.indexOf(Math.max(...result.logits)), 2);
});
