import assert from 'node:assert/strict';
import test from 'node:test';

import { selectRenderableInstances } from '../src/game/rendering.ts';
import { observe } from '../src/game/observation.ts';
import { createGame } from '../src/game/simulation.ts';
import { generateRows } from '../src/game/world.ts';
import {
  createDensePolicyController,
  createScriptedController,
} from '../src/game/controllers.ts';
import { createRemoteController } from '../src/game/remote.ts';
import { reduceGameCommand } from '../src/hooks/useGame.ts';

const signal = () => new AbortController().signal;

test('human action commands advance the authoritative game state', () => {
  const run = reduceGameCommand(
    { game: createGame('manual') },
    { type: 'action', action: 'forward' },
  );

  assert.equal(run.game.fly.row, 1);
});

test('long forward-then-backtrack states stay within renderer instance capacity', () => {
  const retainedLanes = generateRows('manual', -8, 249);
  const backtracked = {
    ...createGame('manual'),
    step: 420,
    fly: { row: 20, column: 0 },
    score: 220,
    lanes: retainedLanes,
  };
  const retainedLogCount = retainedLanes
    .flatMap((lane) => lane.hazards)
    .filter((hazard) => hazard.kind === 'log')
    .length;

  assert.ok(retainedLanes.length > 32);
  assert.ok(retainedLogCount > 128);

  const rendered = selectRenderableInstances(backtracked);
  const hazardCounts = Object.groupBy(rendered.hazards, ({ hazard }) => hazard.kind);

  assert.ok(rendered.lanes.length <= 32);
  assert.ok(rendered.lanes.some((lane) => lane.row === backtracked.fly.row));
  assert.ok(rendered.lanes.every((lane) => Math.abs(lane.row - backtracked.fly.row) <= 16));
  for (const hazards of Object.values(hazardCounts)) assert.ok(hazards.length <= 128);
});

test('scripted controller returns actions in order without invented neural activity', async () => {
  const controller = createScriptedController(['forward', 'wait']);
  const observation = observe(createGame('scripted'));

  assert.deepEqual(await controller.decide(observation, signal()), {
    action: 'forward',
    activity: [],
    diagnostics: {},
  });
  assert.equal((await controller.decide(observation, signal())).action, 'wait');
  controller.reset('scripted');
  assert.equal((await controller.decide(observation, signal())).action, 'forward');
});

test('dense policy chooses the maximum logit using the canonical observation encoding', async () => {
  const observation = observe(createGame('dense'));
  const inputSize = 370;
  const policy = {
    version: 1,
    observationVersion: 1,
    actions: ['forward', 'backward', 'left', 'right', 'wait'],
    source: { kind: 'predicted', name: 'Dense fixture', normalization: 'No activity values' },
    network: {
      kind: 'dense',
      inputSize,
      layers: [{
        name: 'actor', inputSize, outputSize: 5, activation: 'linear',
        weights: Array(inputSize * 5).fill(0),
        bias: [0, 2, 1, 0, -1],
      }],
    },
    activityBodyIds: [],
  };
  const decision = await createDensePolicyController(policy).decide(observation, signal());

  assert.equal(decision.action, 'backward');
  assert.deepEqual(decision.activity, []);
  assert.equal(decision.diagnostics['logit.backward'], 2);
});

test('controller reducer ignores stale decisions and clears activity on reset or mode change', () => {
  let run = reduceGameCommand(
    { game: createGame('runtime') },
    { type: 'controller-start', requestId: 7 },
  );
  run = reduceGameCommand(run, {
    type: 'controller-decision',
    requestId: 7,
    decision: { action: 'forward', activity: [[101, 0.75]], diagnostics: {} },
  });
  assert.deepEqual(run.activity?.values, [[101, 0.75]]);

  const reset = reduceGameCommand(run, { type: 'reset' });
  assert.equal(reset.activity, null);
  const stale = reduceGameCommand(reset, {
    type: 'controller-decision',
    requestId: 7,
    decision: { action: 'right', activity: [[101, 1]], diagnostics: {} },
  });
  assert.equal(stale.game.step, 0);
  assert.equal(stale.activity, null);

  const changedMode = reduceGameCommand(run, { type: 'controller-clear' });
  assert.equal(changedMode.activity, null);
  assert.equal(changedMode.decision, null);
  assert.equal(changedMode.reward, 0);
  assert.deepEqual(changedMode.events, []);
});

test('controller failure pauses the run and exposes the error', () => {
  const pending = reduceGameCommand(
    { game: createGame('failure') },
    { type: 'controller-start', requestId: 1 },
  );
  const failed = reduceGameCommand(pending, {
    type: 'controller-failure', requestId: 1, error: 'Controller timed out.',
  });

  assert.equal(failed.paused, true);
  assert.equal(failed.controllerStatus, 'error');
  assert.equal(failed.controllerError, 'Controller timed out.');
  assert.equal(failed.game.step, 0);
});

test('reset restores the status declared by the active controller mode', () => {
  const reset = reduceGameCommand(
    { game: createGame('manual-reset'), controllerStatus: 'terminal' },
    { type: 'reset', status: 'manual' },
  );

  assert.equal(reset.controllerStatus, 'manual');
});

class FakeSocket extends EventTarget {
  readyState = 0;
  sent = [];
  closed = false;

  open() {
    this.readyState = 1;
    this.dispatchEvent(new Event('open'));
  }

  send(payload) { this.sent.push(JSON.parse(payload)); }
  close() { this.closed = true; this.readyState = 3; }
  receive(payload) {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(payload) }));
  }
}

test('remote controller matches a versioned decision to its request and closes on dispose', async () => {
  const socket = new FakeSocket();
  const controller = createRemoteController('ws://fixture', {
    socketFactory: () => socket,
    timeoutMs: 100,
  });
  socket.open();
  controller.reset('remote-seed');
  const pending = controller.decide(observe(createGame('remote')), signal());
  const request = socket.sent.find((message) => message.type === 'observe');
  socket.receive({
    type: 'decision', version: 1, requestId: request.requestId,
    decision: { action: 'wait', activity: [], diagnostics: { latencyMs: 4 } },
  });

  assert.equal((await pending).action, 'wait');
  assert.deepEqual(socket.sent[0], { type: 'reset', version: 1, seed: 'remote-seed' });
  controller.dispose();
  assert.equal(socket.closed, true);
});

test('remote controller queues reset and one observation while its socket connects', async () => {
  const socket = new FakeSocket();
  const controller = createRemoteController('ws://connecting', {
    socketFactory: () => socket,
    timeoutMs: 100,
  });
  controller.reset('connecting-seed');
  const pending = controller.decide(observe(createGame('connecting')), signal());

  assert.deepEqual(socket.sent, []);
  socket.open();
  assert.equal(socket.sent[0].type, 'reset');
  assert.equal(socket.sent[1].type, 'observe');
  socket.receive({
    type: 'decision', version: 1, requestId: socket.sent[1].requestId,
    decision: { action: 'left', activity: [], diagnostics: {} },
  });

  assert.equal((await pending).action, 'left');
  controller.dispose();
});

test('remote controller rejects incompatible protocol messages and times out once', async () => {
  const badSocket = new FakeSocket();
  const bad = createRemoteController('ws://bad', { socketFactory: () => badSocket, timeoutMs: 100 });
  badSocket.open();
  const rejected = bad.decide(observe(createGame('bad')), signal());
  badSocket.receive({ type: 'decision', version: 2, requestId: 1, decision: {} });
  await assert.rejects(rejected, /version 1/i);
  bad.dispose();

  const quietSocket = new FakeSocket();
  const quiet = createRemoteController('ws://quiet', { socketFactory: () => quietSocket, timeoutMs: 5 });
  quietSocket.open();
  await assert.rejects(quiet.decide(observe(createGame('quiet')), signal()), /timed out/i);
  quiet.dispose();
});
