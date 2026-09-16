import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const root = new URL('../', import.meta.url);

test('generated Python output is not tracked', () => {
  const tracked = execFileSync(
    'git',
    ['ls-files', 'python/.pytest-tmp/**', 'python/runs/**'],
    { cwd: root, encoding: 'utf8' },
  ).trim();

  assert.equal(tracked, '');
});

test('root ignore owns cross-language output', async () => {
  const ignore = await readFile(new URL('.gitignore', root), 'utf8');

  for (const pattern of [
    '.pytest-tmp/',
    'python/runs/',
    '*.py[cod]',
    '.ruff_cache/',
    'coverage/',
  ]) {
    assert.ok(ignore.includes(pattern), pattern);
  }
});
