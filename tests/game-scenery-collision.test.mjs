import assert from 'node:assert/strict';
import test from 'node:test';

import { stepGame } from '../src/game/simulation.ts';
import { obstacleColumnsForRow } from '../src/game/scenery.ts';
import { WORLD_VERSION, generateRows } from '../src/game/world.ts';

const stateAt = (seed, row, column) => ({
  version: WORLD_VERSION,
  seed,
  step: row,
  time: row * 0.2,
  fly: { row, column },
  score: row,
  lanes: generateRows(seed, row - 8, 24),
  terminal: null,
  previousAction: 'wait',
});

test('hitting a grass obstacle blocks the move without killing or passing through it', () => {
  const seed = 'obstacle-test';
  const row = 12;
  assert.equal(generateRows(seed, row, 1)[0].kind, 'grass');
  assert.deepEqual(obstacleColumnsForRow(seed, row, 'grass'), [-1]);

  const result = stepGame(stateAt(seed, row, 0), 'left');
  assert.deepEqual(result.state.fly, { row, column: 0 });
  assert.equal(result.state.terminal, null);
  assert.ok(result.events.some((event) => (
    event.type === 'blocked' && event.reason === 'scenery'
  )));
});

test('manual movement beyond the side limit is blocked instead of causing bounds death', () => {
  const result = stepGame(stateAt('boundary-test', 0, 5), 'right');
  assert.deepEqual(result.state.fly, { row: 0, column: 5 });
  assert.equal(result.state.terminal, null);
  assert.ok(result.events.some((event) => (
    event.type === 'blocked' && event.reason === 'bounds'
  )));
});
