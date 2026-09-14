import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import { parseServerMessage } from '../src/simulation/protocol.ts';

const fixture = JSON.parse(await readFile(new URL('../protocol/v2/valid-session.json', import.meta.url)));

test('protocol v2 fixture parses', () => {
  assert.equal(parseServerMessage(fixture.server).type, 'ready');
  assert.throws(() => parseServerMessage({ ...fixture.server, population: 81 }), /population/i);
});

test('protocol v2 server variants are discriminated and bounded', () => {
  const envelope = {
    version: 2, sessionId: 's-01234567', episodeId: 'e-01234567', sequence: 1, simulationTime: 1,
  };
  const variants = [
    { type: 'reset_complete', ...envelope },
    { type: 'intention', ...envelope, intentionId: 'i-01234567', action: 'left', motorPhase: 'pressing' },
    { type: 'snapshot', ...envelope, body: [0, 0, 0], joints: [0], keys: { W: 0.1 } },
    { type: 'neural_keyframe', ...envelope, updates: [{ neuronId: 1, value: 0.5 }] },
    { type: 'neural_delta', ...envelope, updates: [{ neuronId: 1, value: 0.5 }] },
    { type: 'contact', ...envelope, intentionId: 'i-01234567', requestedKey: 'W', touchedKey: 'W', travel: 0.1, force: 2, debounce: 0.02, confirmed: true },
    { type: 'action_result', ...envelope, intentionId: 'i-01234567', result: 'confirmed' },
    { type: 'metrics', ...envelope, physicsHz: 1000, motorHz: 100, neuralHz: 10, renderHz: 30, latencyMs: 10, droppedRenderFrames: 0 },
    { type: 'paused', ...envelope, code: 'CLIENT_PAUSED', message: 'paused' },
    { type: 'error', ...envelope, code: 'PROTOCOL_ERROR', message: 'bad frame' },
  ];

  assert.deepEqual(variants.map(parseServerMessage).map((message) => message.type), variants.map(({ type }) => type));
  assert.throws(() => parseServerMessage({ ...variants[0], extra: true }), /unexpected/i);
  assert.throws(
    () => parseServerMessage({ ...variants[4], updates: Array.from({ length: 20_001 }, () => ({ neuronId: 1, value: 0 })) }),
    /updates/i,
  );
});

test('protocol v2 rejects committed invalid messages', async () => {
  const invalid = JSON.parse(await readFile(new URL('../protocol/v2/invalid-messages.json', import.meta.url)));
  for (const message of invalid.server) {
    assert.throws(() => parseServerMessage(message));
  }
});
