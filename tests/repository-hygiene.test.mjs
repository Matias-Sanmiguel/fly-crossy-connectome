import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const root = new URL('../', import.meta.url);

test('generated Python output is not tracked', () => {
  const tracked = execFileSync(
    'git',
    [
      'ls-files',
      'python/.pytest-tmp/**',
      'python/.pytest_tmp*/**',
      'python/datasets/**',
      'python/runs/**',
      'runs/crossy-v7-*/**',
    ],
    { cwd: root, encoding: 'utf8' },
  ).trim();

  assert.equal(tracked, '');
});

test('root ignore owns cross-language output', async () => {
  const ignore = await readFile(new URL('.gitignore', root), 'utf8');

  for (const pattern of [
    '.pytest-tmp/',
    'python/.pytest_tmp*/',
    'python/datasets/',
    'python/runs/',
    'runs/crossy-v7-*/',
    '*.py[cod]',
    '.ruff_cache/',
    'coverage/',
  ]) {
    assert.ok(ignore.includes(pattern), pattern);
  }
});

test('V7 documentation names the real curriculum module and profiles', async () => {
  const readme = await readFile(new URL('README.md', root), 'utf8');
  const context = await readFile(new URL('docs/ACTIVE_CONTEXT.md', root), 'utf8');

  for (const text of [readme, context]) {
    assert.ok(text.includes('fly_crossy.v7.curriculum'));
    assert.ok(text.includes('--profile smoke'));
    assert.ok(text.includes('--profile 80'));
    assert.ok(text.includes('--resume'));
    assert.ok(text.includes('--predecessor-report'));
  }
});
