import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const root = new URL('../', import.meta.url);

test('GameScene is a React lifecycle adapter, not a Three renderer', async () => {
  const component = await readFile(
    new URL('src/components/GameScene.tsx', root),
    'utf8',
  );

  assert.match(component, /createGameRenderer/);
  assert.doesNotMatch(component, /new THREE\.(Scene|WebGLRenderer|Mesh)/);
  assert.doesNotMatch(component, /from ['"]three['"]/);
});

test('game renderer owns simulation selection, Kenney loading, and scenery', async () => {
  const renderer = await readFile(
    new URL('src/game/createGameRenderer.ts', root),
    'utf8',
  );

  assert.match(renderer, /selectRenderableInstances/);
  assert.match(renderer, /loadKenneyAssets/);
  assert.match(renderer, /decorationsForRow/);
  assert.match(renderer, /assetStatus/);
  assert.match(renderer, /dispose/);
});
