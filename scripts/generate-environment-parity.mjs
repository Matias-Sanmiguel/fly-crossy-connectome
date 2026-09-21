import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';

import { encodeObservation } from '../src/game/model.ts';
import { observe } from '../src/game/observation.ts';
import { createGame, hazardPositionAt, stepGame } from '../src/game/simulation.ts';
import { generateRows, WORLD_VERSION } from '../src/game/world.ts';

const fixtureUrl = new URL('../tests/fixtures/environment-parity-v2.json', import.meta.url);
const fixture = JSON.parse(await readFile(fixtureUrl, 'utf8'));

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

fixture.environmentVersion = WORLD_VERSION;
for (const check of fixture.positionChecks) {
  const lane = generateRows(check.seed, check.row, 1)[0];
  check.expected = hazardPositionAt(lane, lane.hazards[check.hazardIndex], check.time);
}
for (const parityCase of fixture.cases) {
  const state = stateFrom(parityCase.state);
  const result = stepGame(state, parityCase.action);
  parityCase.expected = {
    beforeObservationSha256: observationSha256(state),
    afterObservationSha256: observationSha256(result.state),
    reward: result.reward,
    fly: result.state.fly,
    step: result.state.step,
    time: result.state.time,
    score: result.state.score,
    terminalReason: result.state.terminal,
    previousAction: result.state.previousAction,
    events: result.events,
    laneRows: result.state.lanes.map((lane) => lane.row),
  };
}

await writeFile(fixtureUrl, `${JSON.stringify(fixture, null, 2)}\n`, 'utf8');

const worldFixtureUrl = new URL('../tests/fixtures/world-generation-v8.json', import.meta.url);
const worldFixture = {
  version: 1,
  environmentVersion: WORLD_VERSION,
  cases: [
    { seed: 'solvability-probe-87', from: 2, count: 6 },
    { seed: 'world-parity-opening', from: -3, count: 13 },
    { seed: 'world-parity-middle', from: 52, count: 11 },
    { seed: 'world-parity-far', from: 152, count: 11 },
  ].map((specification) => ({
    ...specification,
    rows: generateRows(specification.seed, specification.from, specification.count),
  })),
};
await writeFile(worldFixtureUrl, `${JSON.stringify(worldFixture, null, 2)}\n`, 'utf8');
