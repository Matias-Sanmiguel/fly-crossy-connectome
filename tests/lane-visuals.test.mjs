import assert from 'node:assert/strict';
import test from 'node:test';

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
