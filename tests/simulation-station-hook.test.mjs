import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildSimulationObservation,
  seedToUint32,
} from '../src/hooks/useSimulationStation.ts';

import {
  createStation,
} from '../src/simulation/station.ts';


test(
  'string seed maps deterministically to uint32',
  () => {
    const first =
      seedToUint32('experiment-001');

    const second =
      seedToUint32('experiment-001');

    const other =
      seedToUint32('experiment-002');

    assert.equal(first, second);

    assert.notEqual(
      first,
      other,
    );

    assert.ok(
      first >= 0
      && first <= 0xffffffff,
    );
  },
);


test(
  'station observation uses the authoritative game state',
  () => {
    const station =
      createStation('observation-test');

    const observation =
      buildSimulationObservation(station);

    assert.equal(
      observation.gameStep,
      station.game.step,
    );

    assert.equal(
      observation.simulationTime,
      station.game.time,
    );

    assert.equal(
      observation.reward,
      0,
    );

    assert.equal(
      observation.observation.length,
      517,
    );

    assert.ok(
      observation.observation.every(
        Number.isFinite,
      ),
    );
  },
);