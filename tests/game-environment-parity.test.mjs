import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

import { encodeObservation } from '../src/game/model.ts';
import { observe } from '../src/game/observation.ts';
import { createGame, hazardPositionAt, stepGame } from '../src/game/simulation.ts';
import { generateRows } from '../src/game/world.ts';

const fixture = JSON.parse(readFileSync(
  new URL('./fixtures/environment-parity-v1.json', import.meta.url),
  'utf8',
));

const stateFrom = (specification) => {
  if (specification.base === 'createGame') {
    return { ...createGame(specification.seed), ...specification.overrides };
  }
  return specification.value;
};

const observationSha256 = (state) => {
  const values = encodeObservation(observe(state));
  const bytes = Buffer.alloc(values.length * Float32Array.BYTES_PER_ELEMENT);
  values.forEach((value, index) => bytes.writeFloatLE(
    value,
    index * Float32Array.BYTES_PER_ELEMENT,
  ));
  return createHash('sha256').update(bytes).digest('hex');
};

test('environment parity fixture matches authoritative browser behavior', () => {
  assert.equal(fixture.version, 1);
  assert.equal(fixture.observationEncoding, 'float32-le-sha256');

  for (const check of fixture.positionChecks) {
    const lane = generateRows(check.seed, check.row, 1)[0];
    assert.equal(
      hazardPositionAt(lane, lane.hazards[check.hazardIndex], check.time),
      check.expected,
      `${check.seed} row ${check.row} wrapped position`,
    );
  }

  for (const parityCase of fixture.cases) {
    const state = stateFrom(parityCase.state);
    const result = stepGame(state, parityCase.action);
    const expected = parityCase.expected;

    assert.equal(
      observationSha256(state),
      expected.beforeObservationSha256,
      `${parityCase.name} before observation`,
    );
    assert.equal(
      observationSha256(result.state),
      expected.afterObservationSha256,
      `${parityCase.name} after observation`,
    );
    assert.equal(result.reward, expected.reward, `${parityCase.name} reward`);
    assert.deepEqual(result.state.fly, expected.fly, `${parityCase.name} fly`);
    assert.equal(result.state.step, expected.step, `${parityCase.name} step`);
    assert.equal(result.state.time, expected.time, `${parityCase.name} time`);
    assert.equal(result.state.score, expected.score, `${parityCase.name} score`);
    assert.equal(
      result.state.terminal,
      expected.terminalReason,
      `${parityCase.name} terminal reason`,
    );
    assert.equal(
      result.state.previousAction,
      expected.previousAction,
      `${parityCase.name} previous action`,
    );
    assert.deepEqual(result.events, expected.events, `${parityCase.name} events`);
    assert.deepEqual(
      result.state.lanes.map((lane) => lane.row),
      expected.laneRows,
      `${parityCase.name} retained world rows`,
    );
  }
});
