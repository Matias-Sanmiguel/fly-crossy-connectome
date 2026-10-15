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
      'runs/**/*.json',
      'runs/**/*.pt',
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
    '/runs/',
    '*.py[cod]',
    '.ruff_cache/',
    'coverage/',
  ]) {
    assert.ok(ignore.includes(pattern), pattern);
  }
});

test('local training bundles and backups are ignored without hiding published assets', () => {
  const paths = [
    'runs/external-full-model/best_FROZEN_1000.pt',
    'runs/future-experiment/report.json',
    'reports/future-experiment.json',
    'artifacts/downloaded-model.safetensors',
    'checkpoints/external-model.ckpt',
    '.local/repository-cleanup-backup/original.patch',
    'src/example.ts.bak',
    'data/visitor-ranking.sqlite-wal',
  ];
  const ignored = execFileSync('git', ['check-ignore', '--no-index', '--stdin'], {
    cwd: root, input: paths.join('\n'), encoding: 'utf8',
  }).trim().split('\n');
  assert.deepEqual(ignored, paths);
  const tracked = execFileSync('git', ['ls-files',
    'public/assets/expo/full-malecns-*', 'release/eval-v6/training/connectome/checkpoint.pt'],
    { cwd: root, encoding: 'utf8' }).trim().split('\n');
  assert.ok(tracked.includes('public/assets/expo/full-malecns-replay.json'));
  assert.ok(tracked.includes('public/assets/expo/full-malecns-activity.bin.gz'));
  assert.ok(tracked.includes('release/eval-v6/training/connectome/checkpoint.pt'));
});

test('agent documentation, generated reports and duplicate recipe code are not tracked', () => {
  const tracked = execFileSync('git', ['ls-files', 'AGENTS.md', '.superpowers',
    'docs', 'reports', 'runs', 'fly_crossy_v3_flyselfdriving_recipe', 'assets'],
    { cwd: root, encoding: 'utf8' }).trim();
  assert.equal(tracked, '');
});
