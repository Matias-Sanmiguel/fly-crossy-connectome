import assert from 'node:assert/strict';
import test from 'node:test';

import { stepGame } from '../src/game/simulation.ts';
import { WORLD_VERSION } from '../src/game/world.ts';

const makeState = (lanes) => ({
  version: WORLD_VERSION,
  seed: 'physics-v11-test',
  step: 0,
  time: 0,
  fly: { row: 10, column: 0 },
  score: 10,
  lanes,
  terminal: null,
  previousAction: 'wait',
});

const sweptRoad = {
  row: 10,
  kind: 'road',
  direction: 1,
  speed: 10,
  phase: 0,
  hazards: [{ kind: 'car', position: -3, size: 2 }],
};

test('v11 moving off road escapes traffic that reaches old cell later', () => {
  const result = stepGame(makeState([
    sweptRoad,
    { row: 11, kind: 'road', direction: 1, speed: 1, phase: 0, hazards: [] },
  ]), 'forward');
  assert.equal(result.state.fly.row, 11);
  assert.equal(result.state.terminal, null);
});

test('v11 waiting on road remains exposed to swept traffic', () => {
  const result = stepGame(makeState([sweptRoad]), 'wait');
  assert.equal(result.state.terminal, 'vehicle');
});

test('v11 entering road is hit when traffic sweeps destination this tick', () => {
  const result = stepGame(makeState([
    { row: 10, kind: 'grass', hazards: [] },
    { ...sweptRoad, row: 11 },
  ]), 'forward');
  assert.equal(result.state.fly.row, 11);
  assert.equal(result.state.terminal, 'vehicle');
});
