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
    '/runs/*',
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
  const tracked = execFileSync('git', ['ls-files', 'runs/README.md',
    'public/assets/expo/full-malecns-*', 'release/eval-v6/training/connectome/checkpoint.pt'],
    { cwd: root, encoding: 'utf8' }).trim().split('\n');
  assert.ok(tracked.includes('runs/README.md'));
  assert.ok(tracked.includes('public/assets/expo/full-malecns-replay.json'));
  assert.ok(tracked.includes('public/assets/expo/full-malecns-activity.bin.gz'));
  assert.ok(tracked.includes('release/eval-v6/training/connectome/checkpoint.pt'));
});

test('historical run index is compact and preserves the published expo report metrics', async () => {
  const index = JSON.parse(await readFile(new URL('docs/experiments/run-index.json', root), 'utf8'));
  assert.equal(index.schemaVersion, 1);
  assert.match(index.sourceRevision, /^[a-f0-9]{40}$/);
  // This must also work in a shallow checkout without the historical blobs.
  assert.ok(index.files.length > 0);
  assert.equal(new Set(index.files.map(file => file.path)).size, index.files.length);
  for (const file of index.files) {
    assert.ok(file.path.startsWith('runs/') && file.path.endsWith('.json'));
    assert.ok(Number.isSafeInteger(file.bytes) && file.bytes > 0);
    assert.match(file.sha256, /^[a-f0-9]{64}$/);
  }
  assert.equal(index.totalBytes, index.files.reduce((total, file) => total + file.bytes, 0));
  assert.ok(JSON.stringify(index).length < 100_000);
  const replay = JSON.parse(await readFile(new URL('public/assets/expo/full-malecns-replay.json', root), 'utf8'));
  const originalReport = index.files.find(file => file.path ===
    'runs/crossy-full-readout-cached-frontier-v12-10-0006-1000/report.json');
  assert.equal(originalReport.summary.seed, replay.seed);
  assert.equal(originalReport.summary.final.score, replay.metrics.score);
  assert.equal(originalReport.summary.final.steps, replay.actions.length);
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
