import assert from 'node:assert/strict';
import test from 'node:test';

import { createGame } from '../src/game/simulation.ts';
import { reduceGameCommand } from '../src/hooks/useGame.ts';

test('human action commands advance the authoritative game state', () => {
  const run = reduceGameCommand(
    { game: createGame('manual') },
    { type: 'action', action: 'forward' },
  );

  assert.equal(run.game.fly.row, 1);
});
