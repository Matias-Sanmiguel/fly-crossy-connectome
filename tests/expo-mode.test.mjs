import assert from 'node:assert/strict';
import { access, readFile, stat } from 'node:fs/promises';
import test from 'node:test';

import { keyboardHighlightTargets } from '../src/expo/keyboardLayout.ts';

const root = new URL('../', import.meta.url);
const read = (path) => readFile(new URL(path, root), 'utf8');

test('expo query mode selects the dedicated public presentation', async () => {
  const [main, expo] = await Promise.all([
    read('src/main.tsx'),
    read('src/ExpoApp.tsx'),
  ]);

  assert.match(main, /get\('expo'\) === '1'/);
  assert.match(main, /expoMode[\s\S]*?<ExpoApp \/>/);
  assert.match(expo, /<GameScene/);
  assert.match(expo, /<BrainScene/);
  assert.match(expo, /createFixedGraphPolicyController/);
  assert.match(expo, /activityPresentation/);
  assert.match(expo, /Anatomía medida/);
  assert.match(expo, /Actividad neuronal en tiempo real/);
  assert.match(expo, /165,122/);
  assert.match(expo, /25,563,197/);
  assert.match(expo, /708/);
  assert.doesNotMatch(expo, /Math\.random/);
});

test('expo keyboard maps actions to W-A-D and arrow pairs deterministically', () => {
  assert.deepEqual(keyboardHighlightTargets('forward'), ['w', 'arrow-up']);
  assert.deepEqual(keyboardHighlightTargets('left'), ['a', 'arrow-left']);
  assert.deepEqual(keyboardHighlightTargets('right'), ['d', 'arrow-right']);
  assert.deepEqual(keyboardHighlightTargets('wait'), []);
  assert.deepEqual(keyboardHighlightTargets('backward'), []);
});

test('expo brain uses a perceptible continuous rotation and the keyboard overlays only active keys', async () => {
  const [expo, brain, keyboard] = await Promise.all([
    read('src/ExpoApp.tsx'),
    read('src/components/BrainScene.tsx'),
    read('src/components/ExpoKeyboardScene.tsx'),
  ]);

  assert.match(expo, /orbitSpeed=\{\.06\}/);
  assert.match(brain, /rotation\.y \+= dt \* orbitSpeed/);
  assert.doesNotMatch(expo, /gentleMotion/);
  assert.match(keyboard, /opacity:\s*0,/);
  assert.match(keyboard, /selected \? 0\.94 : 0/);
  assert.match(keyboard, /const scale = 0\.76 \/ size\.x/);
});

test('expo uses the supplied local media, model, brand, and fonts', async () => {
  const files = [
    ['public/assets/expo/fly-typing.mp4', 1_000_000],
    ['public/assets/expo/keyboard.glb', 1_000_000],
    ['public/assets/expo/imas-logo-figma.svg', 10_000],
    ['public/assets/expo/legend-anatomy.svg', 1_000],
    ['public/assets/expo/legend-activity.svg', 1_000],
    ['public/assets/expo/action-arrow.svg', 250],
    ['public/assets/expo/fonts/Axiforma-Book.woff2', 10_000],
    ['public/assets/expo/fonts/Axiforma-Bold.woff2', 10_000],
    ['public/assets/expo/fonts/Axiforma-ExtraBold.woff2', 10_000],
    ['public/assets/expo/fonts/Axiforma-Light.woff2', 10_000],
    ['public/assets/expo/fonts/Axiforma-Black.woff2', 10_000],
  ];
  for (const [path, minimumBytes] of files) {
    await access(new URL(path, root));
    assert.ok((await stat(new URL(path, root))).size > minimumBytes, path);
  }
  assert.ok((await stat(new URL('public/assets/expo/keyboard.glb', root))).size < 2_000_000);

  const expo = await read('src/ExpoApp.tsx');
  assert.match(expo, /autoPlay muted loop playsInline/);
  assert.match(expo, /assets\/expo\/keyboard\.glb|ExpoKeyboardScene/);
  assert.match(expo, /assets\/expo\/imas-logo-figma\.svg/);
});

test('expo layout preserves the Figma desktop composition and adds a responsive stack', async () => {
  const css = await read('src/style.css');
  assert.match(css, /\.expo-frame[\s\S]*?aspect-ratio:\s*16\s*\/\s*10/);
  assert.match(css, /\.expo-dashboard[\s\S]*?grid-template-columns:\s*661fr\s+637fr/);
  assert.match(css, /\.expo-right-column[\s\S]*?grid-template-rows:\s*294fr\s+170fr\s+192fr/);
  assert.match(css, /@media\s*\(max-width:\s*900px\)[\s\S]*?\.expo-dashboard[\s\S]*?grid-template-columns:\s*1fr/);
});
