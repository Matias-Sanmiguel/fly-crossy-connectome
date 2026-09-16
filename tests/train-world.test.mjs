import assert from 'node:assert/strict';
import test from 'node:test';

import { generateRows } from '../src/game/world.ts';

test('generated trains are long and rail lanes stay in the slower speed band', () => {
  const railLanes = [];

  for (let seedIndex = 0; seedIndex < 12; seedIndex += 1) {
    railLanes.push(
      ...generateRows(`train-world-${seedIndex}`, 0, 420)
        .filter((lane) => lane.kind === 'rail'),
    );
  }

  assert.ok(railLanes.length > 0);
  assert.ok(
    railLanes.every(
      (lane) => lane.speed >= 10 && lane.speed <= 12,
    ),
  );
  assert.ok(
    railLanes.every(
      (lane) => lane.hazards.length === 1
        && lane.hazards[0].kind === 'train'
        && lane.hazards[0].size === 18,
    ),
  );
});