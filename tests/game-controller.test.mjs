import assert from 'node:assert/strict';
import test from 'node:test';

import { selectRenderableInstances } from '../src/game/rendering.ts';
import { createGame } from '../src/game/simulation.ts';
import { generateRows } from '../src/game/world.ts';
import { reduceGameCommand } from '../src/hooks/useGame.ts';

test('human action commands advance the authoritative game state', () => {
  const run = reduceGameCommand(
    { game: createGame('manual') },
    { type: 'action', action: 'forward' },
  );

  assert.equal(run.game.fly.row, 1);
});

test('long forward-then-backtrack states stay within renderer instance capacity', () => {
  const retainedLanes = generateRows('manual', -8, 249);
  const backtracked = {
    ...createGame('manual'),
    step: 420,
    fly: { row: 20, column: 0 },
    score: 220,
    lanes: retainedLanes,
  };
  const retainedLogCount = retainedLanes
    .flatMap((lane) => lane.hazards)
    .filter((hazard) => hazard.kind === 'log')
    .length;

  assert.ok(retainedLanes.length > 32);
  assert.ok(retainedLogCount > 128);

  const rendered = selectRenderableInstances(backtracked);
  const hazardCounts = Object.groupBy(rendered.hazards, ({ hazard }) => hazard.kind);

  assert.ok(rendered.lanes.length <= 32);
  assert.ok(rendered.lanes.some((lane) => lane.row === backtracked.fly.row));
  assert.ok(rendered.lanes.every((lane) => Math.abs(lane.row - backtracked.fly.row) <= 16));
  for (const hazards of Object.values(hazardCounts)) assert.ok(hazards.length <= 128);
});
