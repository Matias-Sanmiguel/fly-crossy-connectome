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
  assert.match(expo, /createNeuralReplayController/);
  assert.doesNotMatch(expo, /createFixedGraphPolicyController/);
  assert.match(expo, /mode: 'scripted'/);
  assert.doesNotMatch(expo, /controller\?\.kind \?\? 'human'/);
  assert.match(expo, /activityPresentation/);
  assert.match(expo, /Anatomía medida/);
  assert.match(expo, /Actividad neuronal en tiempo real/);
  assert.match(expo, /165,122/);
  assert.match(expo, /25,563,197/);
  assert.match(expo, /708/);
  assert.doesNotMatch(expo, /Math\.random/);
});

test('expo keyboard maps movement actions to individual W-A-S-D keys deterministically', () => {
  assert.deepEqual(keyboardHighlightTargets('forward'), ['w']);
  assert.deepEqual(keyboardHighlightTargets('backward'), ['s']);
  assert.deepEqual(keyboardHighlightTargets('left'), ['a']);
  assert.deepEqual(keyboardHighlightTargets('right'), ['d']);
  assert.deepEqual(keyboardHighlightTargets('wait'), []);
});

test('expo brain matches the main rotation and its legend uses the shader colors', async () => {
  const [expo, brain, css] = await Promise.all([
    read('src/ExpoApp.tsx'),
    read('src/components/BrainScene.tsx'),
    read('src/style.css'),
  ]);

  assert.match(expo, /<BrainScene atlas=\{atlas\} frame=\{activity\.frame\} activityMode=\{activity\.kind\} activityContrast="expo" \/>/);
  assert.doesNotMatch(expo, /orbitSpeed=/);
  assert.match(brain, /rotation\.y \+= dt \* orbitSpeed/);
  assert.doesNotMatch(expo, /legend-(?:anatomy|activity)\.svg/);
  assert.match(css, /\.expo-legend-anatomy i[\s\S]*?rgb\(31 89 191\)/);
  assert.match(css, /\.expo-legend-activity i[\s\S]*?rgb\(51 242 255\)/);
});

test('expo renders four separate pressable key meshes and animates their travel', async () => {
  const keyboard = await read('src/components/ExpoKeyboardScene.tsx');
  assert.match(keyboard, /w: \[0, -0\.06\]/);
  assert.match(keyboard, /a: \[-0\.1, 0\.05\]/);
  assert.match(keyboard, /s: \[0, 0\.05\]/);
  assert.match(keyboard, /d: \[0\.1, 0\.05\]/);
  assert.match(keyboard, /new RoundedBoxGeometry/);
  assert.match(keyboard, /pressed \? PRESSED_KEY_Y : RESTING_KEY_Y/);
  assert.match(keyboard, /THREE\.MathUtils\.damp/);
  assert.match(keyboard, /loadCherryKeycaps/);
  assert.match(keyboard, /key\.mesh\.geometry = caps\[target\]/);
  assert.doesNotMatch(keyboard, /keyboard\.glb/);
});

test('expo keycaps use a centered view, dark matte caps and no keyboard base', async () => {
  const keyboard = await read('src/components/ExpoKeyboardScene.tsx');
  assert.match(keyboard, /camera\.position\.set\(0, 0\.68, 0\.24\)/);
  assert.match(keyboard, /context\.fillStyle = '#eef1ff'/);
  assert.match(keyboard, /color: 0x222b55/);
  assert.match(keyboard, /const idleColor = new THREE\.Color\(0x222b55\)/);
  assert.match(keyboard, /roughness: 0\.72/);
  assert.match(keyboard, /const activeColor = new THREE\.Color\(0xb3ff3b\)/);
  assert.doesNotMatch(keyboard, /baseGeometry|baseMaterial/);
});

test('expo lower-right panel matches the compact Figma controls and credits', async () => {
  const [expo, css] = await Promise.all([read('src/ExpoApp.tsx'), read('src/style.css')]);
  assert.match(expo, /className="expo-action-area"/);
  assert.match(expo, /onClick=\{togglePause\} aria-pressed=\{game.paused\}/);
  assert.match(expo, /onClick=\{restart\}><span>Reiniciar<\/span>/);
  assert.doesNotMatch(expo, /Actividad motora|expo-motor-status/);
  assert.match(expo, /Desarrollado por/);
  assert.match(expo, /Nicolás Luca Giordano y Matías Adrián Sanmiguel\./);
  assert.match(css, /\.expo-action-card\s*\{[^}]*height: 7\.4306cqw/);
  assert.match(css, /\.expo-config-card\s*\{[^}]*height: 5\.4167cqw[^}]*background: #fff/);
  assert.match(css, /\.expo-config-card button:last-child\s*\{ background: #2033ff; color: #fff;/);
  assert.match(css, /\.expo-credits\s*\{[^}]*margin: 1\.1111cqw 0 0/);
  assert.match(css, /\.expo-config-card button\s*\{ min-height: 44px/);
});

test('expo centers button labels and replaces the direction arrow with a dash for WAIT', async () => {
  const [expo, css] = await Promise.all([read('src/ExpoApp.tsx'), read('src/style.css')]);
  assert.match(expo, /action === 'wait'\s*\? <span className="expo-action-dash" aria-hidden="true">-<\/span>\s*: <img/);
  assert.match(expo, /<span>\{game.paused \? 'Continuar' : 'Pausa'\}<\/span>/);
  assert.match(css, /\.expo-config-card button\s*\{[^}]*display: grid;[^}]*place-items: center/);
  assert.match(css, /\.expo-config-card button > span\s*\{ transform: translateY\(0\.08em\)/);
  assert.doesNotMatch(css, /expo-action-arrow-wait/);
  assert.match(css, /\.expo-action-current \.expo-action-dash\s*\{[^}]*color: #b3ff3b/);
});

test('expo uses the supplied local media, brand, and fonts', async () => {
  const files = [
    ['public/assets/expo/fly-typing.mp4', 1_000_000],
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
  const expo = await read('src/ExpoApp.tsx');
  assert.match(expo, /autoPlay muted loop playsInline/);
  assert.match(expo, /ExpoKeyboardScene/);
  assert.match(expo, /assets\/expo\/imas-logo-figma\.svg/);
});

test('expo layout preserves the Figma desktop composition and adds a responsive stack', async () => {
  const css = await read('src/style.css');
  assert.match(css, /\.expo-frame[\s\S]*?aspect-ratio:\s*16\s*\/\s*10/);
  assert.match(css, /\.expo-dashboard[\s\S]*?grid-template-columns:\s*661fr\s+637fr/);
  assert.match(css, /\.expo-right-column[\s\S]*?grid-template-rows:\s*294fr\s+170fr\s+192fr/);
  assert.match(css, /@media\s*\(max-width:\s*900px\)[\s\S]*?\.expo-dashboard[\s\S]*?grid-template-columns:\s*1fr/);
  assert.match(css, /\.expo-score[\s\S]*?-webkit-text-stroke:\s*0\.5556cqw[\s\S]*?text-shadow:\s*0\.3472cqw/);
  assert.match(css, /\.expo-record-value[\s\S]*?-webkit-text-stroke:\s*0\.4167cqw[\s\S]*?text-shadow:\s*0\.2778cqw/);
});
