import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const releaseRoot = new URL('../release/eval-v1/', import.meta.url);
const manifest = JSON.parse(await readFile(new URL('manifest.json', releaseRoot), 'utf8'));

const sha256 = async (url) => createHash('sha256').update(await readFile(url)).digest('hex');

test('tracked release manifest verifies checkpoints, policies, and evaluation evidence', async () => {
  assert.equal(manifest.version, 1);
  assert.equal(manifest.environmentVersion, 3);
  assert.equal(await sha256(new URL(manifest.config.path, releaseRoot)), manifest.config.sha256);
  for (const artifact of Object.values(manifest.reconstructionEnvironment)) {
    assert.match(artifact.sha256, /^[a-f0-9]{64}$/);
    assert.equal(await sha256(new URL(artifact.path, releaseRoot)), artifact.sha256);
  }
  assert.ok(manifest.files.length >= 11);
  for (const file of manifest.files) {
    assert.match(file.sha256, /^[a-f0-9]{64}$/);
    assert.equal(await sha256(new URL(file.path, releaseRoot)), file.sha256, file.path);
  }

  const config = JSON.parse(await readFile(new URL(manifest.config.path, releaseRoot), 'utf8'));
  assert.equal(config.environmentVersion, manifest.environmentVersion);
  for (const controller of ['conventional', 'connectome']) {
    const expected = config.checkpoints[controller];
    const released = manifest.files.find((file) => file.path === `training/${controller}/checkpoint.pt`);
    assert.equal(expected.sha256, released.sha256);
  }
});

test('historical eval-v1 policy remains byte-identical to its preserved public v3 artifact', async () => {
  const released = manifest.files.find(
    (file) => file.path === 'training/connectome/policy.json',
  );
  const bundledUrl = new URL(
    '../public/models/reduced-connectome-policy-v3.json',
    import.meta.url,
  );

  assert.ok(released);
  assert.equal(await sha256(bundledUrl), released.sha256);
  const bundled = JSON.parse(await readFile(bundledUrl, 'utf8'));
  assert.equal(bundled.network.kind, 'fixed-graph');
  assert.equal(bundled.activityBodyIds.length, 80);
});
