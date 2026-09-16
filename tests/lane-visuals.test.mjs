import assert from 'node:assert/strict';
import test from 'node:test';

test('lane substrates cover one complete logical row without gaps', async () => {
  const moduleUrl = new URL('../src/game/laneVisuals.ts', import.meta.url);
  const { LANE_ROW_DEPTH, laneSubstrateLayout } = await import(moduleUrl);
  const layout = laneSubstrateLayout(25);

  assert.equal(LANE_ROW_DEPTH, 1);
  assert.deepEqual(layout.position, [0, -0.12, 0]);
  assert.deepEqual(layout.size, [25, 0.2, 1]);
  assert.equal(layout.size[2], 1);
  assert.throws(() => laneSubstrateLayout(0), /positive finite/i);
});

test('rail detail is built from two rails and repeated sleepers, not a stretched strip', async () => {
  const moduleUrl = new URL('../src/game/laneVisuals.ts', import.meta.url);
  const { laneDetails } = await import(moduleUrl);
  const details = laneDetails('rail', 25);

  assert.equal(details.filter((detail) => detail.kind === 'rail').length, 2);
  assert.ok(details.filter((detail) => detail.kind === 'sleeper').length >= 20);
  assert.equal(details.some((detail) => detail.kind === 'strip'), false);
  assert.ok(details.every((detail) => detail.size.every(Number.isFinite)));
});

test('road surface has no full-width white overlay', async () => {
  const moduleUrl = new URL('../src/game/laneVisuals.ts', import.meta.url);
  const { laneDetails } = await import(moduleUrl);

  assert.deepEqual(laneDetails('road', 25), []);
});
