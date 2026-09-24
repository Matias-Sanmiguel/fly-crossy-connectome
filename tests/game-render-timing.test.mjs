import assert from 'node:assert/strict';
import test from 'node:test';

import { interpolatedSimulationTime } from '../src/game/renderTiming.ts';

test('hazards visually replay the same authoritative interval as the fly hop', () => {
  assert.equal(interpolatedSimulationTime(0.2, 1, 0), 0);
  assert.equal(interpolatedSimulationTime(0.2, 1, 100), 0.1);
  assert.equal(interpolatedSimulationTime(0.2, 1, 200), 0.2);
  assert.equal(interpolatedSimulationTime(0.2, 1, 500), 0.2);
});

test('initial render stays at the initial authoritative time', () => {
  assert.equal(interpolatedSimulationTime(0, 0, 0), 0);
  assert.equal(interpolatedSimulationTime(0, 0, 500), 0);
});
