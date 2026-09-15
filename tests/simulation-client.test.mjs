import assert from 'node:assert/strict';
import test from 'node:test';

import { createSimulationClient } from '../src/simulation/client.ts';
import { OutboundQueue } from '../src/simulation/queue.ts';

const hashes = {
  graphHash: 'a'.repeat(64),
  checkpointHash: 'b'.repeat(64),
};

const envelope = (sequence, overrides = {}) => ({
  version: 2,
  sessionId: 's-current1',
  episodeId: 'e-current1',
  sequence,
  simulationTime: sequence,
  ...overrides,
});

const ready = (overrides = {}) => ({
  type: 'ready',
  ...envelope(0),
  backend: 'cpu',
  population: 80,
  ...hashes,
  ...overrides,
});

class FakeSocket {
  readyState = 0;
  sent = [];
  listeners = new Map();

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type, listener) {
    this.listeners.get(type)?.delete(listener);
  }

  emit(type, event = {}) {
    for (const listener of this.listeners.get(type) ?? []) listener(event);
  }

  open() {
    this.readyState = 1;
    this.emit('open');
  }

  receive(message) {
    this.emit('message', { data: JSON.stringify(message) });
  }

  send(message) {
    this.sent.push(JSON.parse(message));
  }

  close() {
    this.readyState = 3;
  }
}

const createHarness = (sessionIds = ['s-current1'], episodeIds = ['e-current1']) => {
  const sockets = [];
  const scheduled = [];
  const client = createSimulationClient('ws://simulation.test/api/simulation', {
    socketFactory: () => {
      const socket = new FakeSocket();
      sockets.push(socket);
      return socket;
    },
    sessionIdFactory: () => sessionIds.shift() ?? 's-exhausted1',
    episodeIdFactory: () => episodeIds.shift() ?? 'e-exhausted1',
    timerFactory: {
      setTimeout(callback) {
        scheduled.push(callback);
        return callback;
      },
      clearTimeout() {},
    },
  });
  return { client, sockets, scheduled };
};

test('queue drops superseded observations but keeps reset', () => {
  const queue = new OutboundQueue(4);
  queue.enqueue({ type: 'reset', sequence: 1 });
  queue.enqueue({ type: 'observation', sequence: 2 });
  queue.enqueue({ type: 'observation', sequence: 3 });

  assert.deepEqual(queue.drain().map((item) => item.type), ['reset', 'observation']);
});

test('queue makes room by dropping observations before critical messages', () => {
  const queue = new OutboundQueue(2);
  queue.enqueue({ type: 'reset', sequence: 1 });
  queue.enqueue({ type: 'observation', sequence: 2 });
  queue.enqueue({ type: 'request_keyframe', sequence: 3 });

  assert.deepEqual(queue.drain().map((item) => item.type), ['reset', 'request_keyframe']);
});

test('client ignores frames from an old session', () => {
  const { client, sockets } = createHarness();
  sockets[0].open();
  sockets[0].receive(ready({ sessionId: 's-old00001' }));

  assert.equal(client.state.phase, 'connecting');
  client.close();
});

test('client sends zero-sequence hello before configuration and becomes ready', () => {
  const { client, sockets } = createHarness();
  client.configure({ population: 80, backend: 'cpu', seed: 7, speed: 1 });
  sockets[0].open();

  assert.deepEqual(sockets[0].sent.map(({ type, sequence }) => ({ type, sequence })), [
    { type: 'hello', sequence: 0 },
    { type: 'configure', sequence: 1 },
  ]);
  assert.deepEqual(sockets[0].sent[0], {
    type: 'hello', version: 2, sessionId: 's-current1', episodeId: 'e-current1', sequence: 0, simulationTime: 0,
    supportedVersions: [2], uiBuild: 'browser',
  });
  sockets[0].receive(ready());
  assert.deepEqual(client.state, {
    phase: 'ready', sessionId: 's-current1', episodeId: 'e-current1', backend: 'cpu', error: null,
  });
  client.close();
});

test('client permits one observation until its action result arrives', () => {
  const { client, sockets } = createHarness();
  client.configure({ population: 80, backend: 'cpu', seed: 7, speed: 1 });
  sockets[0].open();
  sockets[0].receive(ready());
  client.observe({ gameStep: 3, observation: Array(370).fill(0), reward: 1, simulationTime: 2 });

  assert.throws(
    () => client.observe({ gameStep: 4, observation: Array(370).fill(0), reward: 0, simulationTime: 3 }),
    /pending observation/i,
  );
  sockets[0].receive({ type: 'intention', ...envelope(1), intentionId: 'i-current1', action: 'wait', motorPhase: 'neutral' });
  sockets[0].receive({ type: 'action_result', ...envelope(2), intentionId: 'i-current1', result: 'waited' });

  assert.equal(client.state.phase, 'ready');
  client.close();
});

test('client requests a keyframe after a neural delta sequence gap', () => {
  const { client, sockets } = createHarness();
  client.configure({ population: 80, backend: 'cpu', seed: 7, speed: 1 });
  sockets[0].open();
  sockets[0].receive(ready());
  sockets[0].receive({ type: 'neural_delta', ...envelope(2), updates: [] });

  assert.deepEqual(sockets[0].sent.at(-1), {
    type: 'request_keyframe', version: 2, sessionId: 's-current1', episodeId: 'e-current1', sequence: 2, simulationTime: 2,
  });
  client.close();
});

test('client creates a new session and restarts its sequence on reconnect', () => {
  const { client, sockets, scheduled } = createHarness(['s-current1', 's-current2'], ['e-current1', 'e-current2']);
  client.configure({ population: 80, backend: 'cpu', seed: 7, speed: 1 });
  sockets[0].open();
  sockets[0].receive(ready());
  sockets[0].readyState = 3;
  sockets[0].emit('close');
  scheduled.shift()();
  sockets[1].open();

  assert.deepEqual(sockets[1].sent.map(({ type, sessionId, sequence }) => ({ type, sessionId, sequence })), [
    { type: 'hello', sessionId: 's-current2', sequence: 0 },
    { type: 'configure', sessionId: 's-current2', sequence: 1 },
  ]);
  assert.equal(client.state.sessionId, 's-current2');
  client.close();
});
