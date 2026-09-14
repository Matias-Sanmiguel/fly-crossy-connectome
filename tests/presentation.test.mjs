import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), 'utf8');

test('project identity and modified-template provenance are explicit', async () => {
  const [packageText, page, app, attribution, readme] = await Promise.all([
    read('package.json'),
    read('index.html'),
    read('src/App.tsx'),
    read('src/components/Attribution.tsx'),
    read('README.md'),
  ]);
  const packageJson = JSON.parse(packageText);

  assert.equal(packageJson.name, 'fly-crossy-connectome');
  assert.match(page, /<title>Fly Crossy Connectome<\/title>/);
  assert.match(app, /Fly Crossy Connectome/);
  assert.match(attribution, /Modified for Fly Crossy Connectome/);
  assert.match(readme, /^# Fly Crossy Connectome/m);
  for (const source of [attribution, readme]) {
    assert.match(source, /https:\/\/github\.com\/cobanov\/fly-connectome-template/);
    assert.match(source, /https:\/\/github\.com\/cobanov/);
  }
});

test('semantic tokens, desktop split, mobile stack, and accessible control sizing are declared', async () => {
  const css = await read('src/style.css');

  assert.match(css, /--color-primary:\s*#00bd7d/i);
  assert.match(css, /--color-warning:\s*#d97706/i);
  assert.match(css, /--color-danger:\s*#dc2626/i);
  assert.match(css, /--color-surface:\s*#fff(?:fff)?/i);
  assert.match(css, /--color-text:\s*#111827/i);
  assert.match(css, /grid-template-columns:\s*minmax\(0,\s*3fr\)\s+minmax\(340px,\s*2fr\)/i);
  assert.match(css, /\.game-controls button[^}]*min-(?:block-size|height):\s*44px/is);
  assert.match(css, /\.game-status-bar button[^}]*min-(?:block-size|height):\s*44px/is);
  assert.match(css, /button,\s*select,\s*\.header-link\s*\{[^}]*min-width:\s*44px/is);
  assert.match(css, /a\s*\{[^}]*min-width:\s*44px/is);
  assert.match(css, /a\s*\{[^}]*min-(?:block-size|height):\s*44px/is);
  assert.match(css, /@media\s*\(max-width:\s*760px\)[\s\S]*?\.eyebrow\s*\{[^}]*font-size:\s*var\(--text-xs\)/i);
  assert.match(css, /@media\s*\(max-width:\s*760px\)[\s\S]*grid-template-areas:\s*['"]environment['"]\s*['"]brain['"]\s*['"]telemetry['"]\s*['"]fly['"]/i);
  assert.doesNotMatch(css, /(?:environment|brain)-(?:panel|viewport)[^}]*display:\s*none/is);
});

test('terminal controls, hidden-tab scheduling, and controller disposal are wired', async () => {
  const [app, controls, hook] = await Promise.all([
    read('src/App.tsx'),
    read('src/components/GameControls.tsx'),
    read('src/hooks/useGame.ts'),
  ]);

  assert.match(app, /terminal=\{game\.state\.terminal !== null\}/);
  assert.match(controls, /terminal:\s*boolean/);
  assert.match(controls, /disabled=\{paused \|\| terminal\}/);
  assert.match(hook, /document\.hidden/);
  assert.match(hook, /visibilitychange/);
  assert.match(hook, /if \(!visible\)[\s\S]{0,160}dispatch\(\{ type: 'controller-cancel' \}\)/);
  assert.match(hook, /if \(isInteractiveKeyboardTarget\(event\.target\)\) return;[\s\S]{0,240}event\.preventDefault\(\)/);
  assert.match(hook, /controller\?\.dispose\(\)/);
});
