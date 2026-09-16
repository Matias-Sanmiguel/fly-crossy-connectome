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

const findBlockingScenario = () => {
  for (let seedIndex = 0; seedIndex < 30; seedIndex += 1) {
    const seed = `collision-audit-${seedIndex}`;
    for (let row = 3; row <= 80; row += 1) {
      const lane = generateRows(seed, row, 1)[0];
      if (lane.kind !== 'grass') continue;
      const columns = obstacleColumnsForRow(seed, row, 'grass');
      if (columns.length === 0) continue;
      const target = columns[0];
      return {
        seed,
        row,
        target,
        from: target - 1,
        action: 'right',
      };
    }
  }
  throw new Error('expected a deterministic blocking scenery scenario');
};

test('hitting a grass obstacle blocks the move without killing or passing through it', () => {
  const scenario = findBlockingScenario();
  const result = stepGame(
    stateAt(scenario.seed, scenario.row, scenario.from),
    scenario.action,
  );
  assert.deepEqual(result.state.fly, {
    row: scenario.row,
    column: scenario.from,
  });
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
