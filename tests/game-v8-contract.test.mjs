import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { encodeObservation } from '../src/game/model.ts';
import { CELL_ENCODING, OBSERVATION_VERSION, observe } from '../src/game/observation.ts';
import { createGame, stepGame } from '../src/game/simulation.ts';
import { obstacleColumnsForRow } from '../src/game/scenery.ts';
import {
  WORLD_LAYOUT_VERSION,
  WORLD_VERSION,
  generateRows,
} from '../src/game/world.ts';

test('v8 changes reward contract without changing v6 world layout', () => {
  assert.equal(WORLD_VERSION, 8);
  assert.equal(WORLD_LAYOUT_VERSION, 6);

  const fixture = JSON.parse(readFileSync(
    new URL('./fixtures/world-generation-v6.json', import.meta.url),
    'utf8',
  ));
  for (const parityCase of fixture.cases) {
    assert.deepEqual(
      generateRows(parityCase.seed, parityCase.from, parityCase.count),
      parityCase.rows,
      parityCase.seed,
    );
  }
});

test('ObservationV2 exposes static blockers and keeps the 370-value boundary', () => {
  const seed = 'v8-blocker-contract';
  let row = 3;
  while (obstacleColumnsForRow(seed, row, 'grass').length === 0) row += 1;
  const blocker = obstacleColumnsForRow(seed, row, 'grass')[0];

  const state = {
    ...createGame(seed),
    fly: { row, column: 0 },
    score: row,
    lanes: [{ row, kind: 'grass', hazards: [] }],
  };
  const observation = observe(state);
  const columnIndex = blocker + 5;

  assert.equal(observation.version, OBSERVATION_VERSION);
  assert.equal(observation.version, 2);
  assert.equal(observation.cells[5][columnIndex], CELL_ENCODING.blocker);
  assert.equal(CELL_ENCODING.blocker, 8);
  assert.equal(encodeObservation(observation).length, 370);
  assert.equal(encodeObservation(observation)[5 * 11 + columnIndex], 1);
});

test('Reward v4 exempts safe carried waits but keeps other costs', () => {
  const waiting = stepGame(createGame('v8-wait-cost'), 'wait');
  assert.equal(waiting.reward, -0.11);

  const edgeState = {
    ...createGame('v8-block-cost'),
    fly: { row: 0, column: 5 },
    lanes: [{ row: 0, kind: 'grass', hazards: [] }],
  };
  const blocked = stepGame(edgeState, 'right');
  assert.equal(blocked.state.terminal, null);
  assert.ok(blocked.events.some((event) => event.type === 'blocked'));
  assert.equal(blocked.reward, -0.21000000000000002);


  const carriedState = {
    ...createGame('v8-safe-carry'),
    fly: { row: 3, column: 0 },
    score: 3,
    lanes: [{
      row: 3,
      kind: 'river',
      hazards: [{ kind: 'log', position: -0.5, size: 2 }],
      direction: 1,
      speed: 1,
      phase: 0,
    }],
  };
  const carried = stepGame(carriedState, 'wait');
  assert.equal(carried.state.terminal, null);
  assert.ok(carried.events.some((event) => event.type === 'carried'));
  assert.equal(carried.reward, -0.01);

  const unsafeCarryState = {
    ...createGame('v8-unsafe-carry'),
    fly: { row: 3, column: 4.9 },
    score: 3,
    lanes: [{
      row: 3,
      kind: 'river',
      hazards: [{ kind: 'log', position: 4.4, size: 2 }],
      direction: 1,
      speed: 1,
      phase: 0,
    }],
  };
  const unsafeCarry = stepGame(unsafeCarryState, 'wait');
  assert.ok(unsafeCarry.events.some((event) => event.type === 'carried'));
  assert.equal(unsafeCarry.state.terminal, 'bounds');
  assert.equal(unsafeCarry.reward, -10.11);
});
