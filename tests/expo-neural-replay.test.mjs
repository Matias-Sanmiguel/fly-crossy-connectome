import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { gunzipSync, gzipSync } from 'node:zlib';
import { createNeuralReplayController, decodeNeuralResponse, parseExpoReplay, parseNeuralReplay, sha256 } from '../src/expo/neuralReplay.ts';

const replay = { seed: 'test', actions: ['forward', 'wait'] };
const visible = new Set([1, 2, 3]);
const data = new Uint8Array([0, 255, 128, 0]);
async function manifest() {
  return {
    version: 1, kind: 'full-malecns-neural-state-replay', encoding: 'frame-major-uint8-gzip',
    seed: replay.seed, frameCount: 2, bodyIds: [1, 2],
    checkpointSha256: 'a'.repeat(64), dataSha256: await sha256(data),
    actionsSha256: await sha256(new TextEncoder().encode(JSON.stringify(replay.actions))),
    source: { kind: 'predicted', name: 'checkpoint state', normalization: 'abs(state), uint8 / 255' },
  };
}

test('activity loading handles raw gzip and already HTTP-decoded gzip without double decoding', async () => {
  assert.deepEqual(await decodeNeuralResponse(new Response(gzipSync(data))), data);
  assert.deepEqual(await decodeNeuralResponse(new Response(data, { headers: { 'Content-Encoding': 'gzip' } })), data);
  await assert.rejects(decodeNeuralResponse(new Response(null)), /vacía/);
  await assert.rejects(decodeNeuralResponse(new Response(data)));
});

test('neural replay pairs each action with its actual recorded frame and resets exactly', async () => {
  const neural = await parseNeuralReplay(await manifest(), data, replay, visible);
  const controller = createNeuralReplayController(replay, neural);
  const signal = new AbortController().signal;
  const first = await controller.decide({}, signal);
  assert.deepEqual(first.activity, [[2, 1]]);
  assert.equal(first.action, 'forward');
  const second = await controller.decide({}, signal);
  assert.deepEqual(second.activity, [[1, 128 / 255]]);
  assert.equal(second.action, 'wait');
  assert.equal(controller.kind, 'scripted');
  assert.equal(controller.activityProvenance, 'model-output');
  await assert.rejects(controller.decide({}, signal), /terminado/);
  assert.throws(() => controller.reset('wrong'), /seed/);
  controller.reset(replay.seed);
  assert.deepEqual(await controller.decide({}, signal), first);
});

test('an aborted decision does not consume a neural frame', async () => {
  const controller = createNeuralReplayController(replay, await parseNeuralReplay(await manifest(), data, replay, visible));
  const abort = new AbortController();
  abort.abort();
  await assert.rejects(controller.decide({}, abort.signal), { name: 'AbortError' });
  assert.equal((await controller.decide({}, new AbortController().signal)).action, 'forward');
});

test('neural metadata rejects unknown/duplicate IDs, wrong seed, frame counts and corrupt payloads', async () => {
  const base = await manifest();
  for (const patch of [{ bodyIds: [1, 99] }, { bodyIds: [1, 1] }, { seed: 'wrong' }, { frameCount: 3 }, { checkpointSha256: '' }, { source: { ...base.source, kind: 'measured' } }]) {
    await assert.rejects(parseNeuralReplay({ ...base, ...patch }, data, replay, visible));
  }
  await assert.rejects(parseNeuralReplay(base, data.slice(1), replay, visible));
  await assert.rejects(parseNeuralReplay(base, new Uint8Array([1, 255, 128, 0]), replay, visible), /dañada/);
  await assert.rejects(parseNeuralReplay(base, data, { ...replay, actions: ['wait', 'forward'] }, visible), /corresponde/);
});

test('action replay rejects malformed actions', () => {
  const value = { version: 1, kind: 'full-malecns-checkpoint-action-replay', ...replay };
  assert.deepEqual(parseExpoReplay(value), replay);
  assert.throws(() => parseExpoReplay({ ...value, actions: ['jump'] }));
  assert.throws(() => parseExpoReplay({ ...value, actions: [] }));
});

test('published activity covers the full replay and only measured visible somata', async () => {
  const read = path => readFile(new URL(`../public/${path}`, import.meta.url));
  const actual = parseExpoReplay(JSON.parse(await read('assets/expo/full-malecns-replay.json')));
  const meta = JSON.parse(await read('assets/expo/full-malecns-activity.json'));
  const bytes = gunzipSync(await read('assets/expo/full-malecns-activity.bin.gz'));
  const idsBuffer = await read('data/brain-atlas/ids.bin');
  const groups = await read('data/brain-atlas/groups.bin');
  const ids = new Set();
  for (let i = 0; i < groups.length; i++) if (groups[i] < 3) ids.add(idsBuffer.readUInt32LE(i * 4));
  const neural = await parseNeuralReplay(meta, bytes, actual, ids);
  assert.ok(neural.bodyIds.length >= 512 && neural.bodyIds.length < 20_000);
  assert.ok(bytes.length < 20_000_000);
  assert.equal(neural.frameCount, actual.actions.length);
  assert.equal(meta.displayNeuronLimit, 1536);
  for (let frame = 0; frame < neural.frameCount; frame++) {
    let active = 0;
    for (const value of bytes.subarray(frame * neural.bodyIds.length, (frame + 1) * neural.bodyIds.length)) active += Number(value > 0);
    assert.ok(active > 80 && active <= 1536);
    const counts = [0, 0, 0];
    for (let i = 0; i < neural.bodyIds.length; i++) if (bytes[frame * neural.bodyIds.length + i] > 0) counts[meta.bodyRegions[i]]++;
    assert.ok(counts.every(v => v <= 512));
    assert.ok(counts[0] > 0 && counts[1] > 0 && counts[2] > 0);
  }
  assert.match(neural.source.selectionRule, /actual magnitude changes/);
  assert.deepEqual(meta.regionalDisplayGains, { optic: 1, central: 16, descending: 16 });
  assert.match(meta.sourceFullDataSha256, /^[a-f0-9]{64}$/);
  assert.match(neural.source.normalization, /not biological recordings/);
});

test('hashing a subarray respects its byte offset without requiring a full-payload copy', async () => {
  const padded = new Uint8Array([99, ...data, 88]);
  assert.equal(await sha256(padded.subarray(1, -1)), await sha256(data));
});
