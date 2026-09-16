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

test('one-unit Kenney tiles cover the complete 25-unit lane exactly', async () => {
  const moduleUrl = new URL('../src/game/laneVisuals.ts', import.meta.url);
  const { tiledLaneRowLayout } = await import(moduleUrl);
  const tiles = tiledLaneRowLayout(25);

  assert.equal(tiles.length, 25);
  assert.deepEqual(
    tiles.map((tile) => tile.position[0]),
    Array.from({ length: 25 }, (_, index) => index - 12),
  );
  assert.ok(tiles.every((tile) => tile.size === 1));
  assert.equal(tiles[0].position[0] - tiles[0].size / 2, -12.5);
  assert.equal(tiles.at(-1).position[0] + tiles.at(-1).size / 2, 12.5);
  for (let index = 1; index < tiles.length; index += 1) {
    const previousRight = tiles[index - 1].position[0] + tiles[index - 1].size / 2;
    const nextLeft = tiles[index].position[0] - tiles[index].size / 2;
    assert.equal(nextLeft, previousRight);
  }
  assert.throws(() => tiledLaneRowLayout(25, 2), /exactly divisible/i);
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
