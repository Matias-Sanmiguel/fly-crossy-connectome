import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const releaseRoot = new URL('../release/eval-v6/', import.meta.url);
const manifest = JSON.parse(
  await readFile(new URL('manifest.json', releaseRoot), 'utf8'),
);

const sha256 = async (url) => (
  createHash('sha256').update(await readFile(url)).digest('hex')
);

test('eval-v6 release pins the selected 80-neuron checkpoint and evidence', async () => {
  assert.equal(manifest.version, 1);
  assert.equal(manifest.environmentVersion, 6);
  assert.equal(manifest.population, 80);
  assert.equal(
    manifest.environmentFreezeCommit,
    '486a8265cf793daf9e1ea024b1c147b42a7370b3',
  );
  assert.equal(manifest.selection.selectedRun, 'final-80n-v6-train-1m-01');
  assert.equal(manifest.finalTest.episodeCount, 200);
  assert.equal(manifest.finalTest.maxStepsPerEpisode, 500);

  for (const file of manifest.files) {
    assert.match(file.sha256, /^[a-f0-9]{64}$/);
    assert.equal(
      await sha256(new URL(file.path, releaseRoot)),
      file.sha256,
      file.path,
    );
  }

  assert.equal(
    await sha256(new URL(manifest.checkpoint.path, releaseRoot)),
    manifest.checkpoint.sha256,
  );
  assert.equal(
    await sha256(new URL(manifest.policy.path, releaseRoot)),
    manifest.policy.sha256,
  );
  assert.equal(
    await sha256(new URL(manifest.graph.path, releaseRoot)),
    manifest.graph.sha256,
  );

  const finalTest = JSON.parse(
    await readFile(new URL(manifest.finalTest.path, releaseRoot), 'utf8'),
  );
  assert.equal(finalTest.environmentVersion, 6);
  assert.equal(finalTest.trainingRun, 'final-80n-v6-train-1m-01');
  assert.equal(finalTest.checkpointSha256, manifest.checkpoint.sha256);
  assert.equal(finalTest.episodeCount, 200);
  assert.equal(finalTest.maxStepsPerEpisode, 500);
  assert.equal(finalTest.seeds.length, 200);
  assert.deepEqual(finalTest.summary, manifest.finalTest.summary);
});

test('browser autoplay ships the exact eval-v6 released policy', async () => {
  const bundledUrl = new URL(
    '../public/models/reduced-connectome-policy-v6.json',
    import.meta.url,
  );
  assert.equal(await sha256(bundledUrl), manifest.policy.sha256);

  const bundled = JSON.parse(await readFile(bundledUrl, 'utf8'));
  assert.equal(bundled.network.kind, 'fixed-graph');
  assert.equal(bundled.activityBodyIds.length, 80);
  assert.equal(
    bundled.source.checkpointHash,
    manifest.checkpoint.sha256,
  );
});
