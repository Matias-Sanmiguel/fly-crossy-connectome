import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { encodeObservation } from '../src/game/model.ts';
import { CELL_ENCODING, OBSERVATION_VERSION, observe } from '../src/game/observation.ts';
import { createGame, stepGame } from '../src/game/simulation.ts';
import {
  WORLD_LAYOUT_VERSION,
  WORLD_VERSION,
  generateRows,
} from '../src/game/world.ts';

test('v10 changes only the observation contract and preserves the v6 world layout', () => {
  assert.equal(WORLD_VERSION, 10);
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

test('ObservationV4 exposes signed lateral position and 517 values', () => {
  const state = {
    ...createGame('v10-position-contract'),
    fly: { row: 0, column: 3 },
  };
  const observation = observe(state);

  assert.equal(observation.version, OBSERVATION_VERSION);
  assert.equal(observation.version, 4);
  assert.equal(observation.signedColumn, 0.6);
  assert.equal(observation.edgeDistance, 0.4);
  assert.equal(encodeObservation(observation).length, 517);
});

test('ObservationV4 exposes continuous sub-cell hazard phase without future information', () => {
  const state = {
    ...createGame('v10-hazard-offset'),
    fly: { row: 2, column: 0 },
    score: 2,
    time: 0,
    lanes: [
      { row: 2, kind: 'grass', hazards: [] },
      {
        row: 3,
        kind: 'road',
        direction: 1,
        speed: 1,
        phase: 0,
        hazards: [{ kind: 'car', position: -0.75, size: 2 }],
      },
    ],
  };
  const observation = observe(state);

  assert.equal(observation.cells[6][4], CELL_ENCODING.vehicle);
  assert.equal(observation.cells[6][5], CELL_ENCODING.vehicle);
  assert.equal(observation.hazardOffset[6][4], 0.25);
  assert.equal(observation.hazardOffset[6][5], -0.75);
  assert.deepEqual(observation.motion[6][5], [1, 0.2]);
});

test('Reward v5 distinguishes waiting from navigation in environment v10', () => {
  const waiting = stepGame(createGame('v10-wait-cost'), 'wait');
  assert.equal(waiting.reward, -0.04);

  const carriedState = {
    ...createGame('v10-safe-carry'),
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
});
