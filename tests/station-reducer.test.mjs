import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createStation,
  reduceStation,
} from '../src/simulation/station.ts';

const envelope = {
  version: 2,
  sessionId: 's-00000001',
  episodeId: 'e-00000001',
  sequence: 1,
  simulationTime: 0,
};

const intention = (
  id = 'i-00000001',
  action = 'forward',
) => ({
  ...envelope,
  type: 'intention',
  intentionId: id,
  action,
  motorPhase:
    action === 'wait'
      ? 'neutral'
      : 'targeting',
});

const result = (
  id = 'i-00000001',
  outcome = 'confirmed',
) => ({
  ...envelope,
  sequence: 2,
  type: 'action_result',
  intentionId: id,
  result: outcome,
});

test(
  'intention alone never advances the game',
  () => {
    const initial = createStation('gate-test');

    const requested = reduceStation(
      initial,
      intention(),
    );

    assert.equal(initial.game.step, 0);
    assert.equal(requested.game.step, 0);
    assert.equal(requested.pending?.requestedAction, 'forward');
    assert.equal(requested.pending?.requestedKey, 'W');
  },
);

test(
  'matching confirmed result advances exactly once',
  () => {
    const requested = reduceStation(
      createStation('gate-test'),
      intention(),
    );

    const confirmed = reduceStation(
      requested,
      result(),
    );

    const duplicate = reduceStation(
      confirmed,
      result(),
    );

    assert.equal(confirmed.game.step, 1);
    assert.equal(
      confirmed.lastAppliedAction,
      'forward',
    );

    assert.equal(
      duplicate.game.step,
      confirmed.game.step,
    );
  },
);

test(
  'failed physical press advances as wait',
  () => {
    const requested = reduceStation(
      createStation('gate-test'),
      intention('i-00000002', 'left'),
    );

    const failed = reduceStation(
      requested,
      result('i-00000002', 'failed'),
    );

    assert.equal(failed.game.step, 1);
    assert.equal(
      failed.lastAppliedAction,
      'wait',
    );
  },
);

test(
  'result for another intention fails closed',
  () => {
    const requested = reduceStation(
      createStation('gate-test'),
      intention('i-00000003', 'right'),
    );

    const invalid = reduceStation(
      requested,
      result('i-00000004'),
    );

    assert.equal(invalid.phase, 'error');
    assert.equal(invalid.game.step, 0);
    assert.match(
      invalid.error ?? '',
      /does not match/i,
    );
  },
);