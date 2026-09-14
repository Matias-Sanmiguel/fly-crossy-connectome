import assert from 'node:assert/strict';
import test from 'node:test';

import { selectRenderableInstances } from '../src/game/rendering.ts';
import { observe } from '../src/game/observation.ts';
import { createGame } from '../src/game/simulation.ts';
import { generateRows } from '../src/game/world.ts';
import {
  createDensePolicyController,
  createFixedGraphPolicyController,
  createScriptedController,
  validateControllerDecision,
} from '../src/game/controllers.ts';
import { createRemoteController } from '../src/game/remote.ts';
import { activityPresentation } from '../src/game/provenance.ts';
import {
  autonomousDecisionDelayMs,
  isInteractiveKeyboardTarget,
  normalizeSeed,
  reduceGameCommand,
} from '../src/hooks/useGame.ts';

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
  assert.equal(controller.activityProvenance, 'none');
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
  assert.equal(createDensePolicyController(policy).activityProvenance, 'none');
});

test('fixed graph controller persists recurrence and resets simulated activity', async () => {
  const observation = observe(createGame('fixed'));
  const inputSize = 370;
  const policy = {
    version: 1,
    observationVersion: 1,
    actions: ['forward', 'backward', 'left', 'right', 'wait'],
    source: { kind: 'predicted', name: 'Fixed fixture', normalization: 'tanh mapped to [0, 1]' },
    network: {
      kind: 'fixed-graph', inputSize, bodyIds: [101], activation: 'tanh',
      sensoryWeights: Array(inputSize).fill(0),
      recurrentSource: [0], recurrentTarget: [0], recurrentWeights: [1],
      recurrentGain: 1, timeConstant: 1,
      actorWeights: [1, 0, 0, 0, 0], actorBias: [0, 1, 0, 0, 0],
    },
    activityBodyIds: [101],
  };
  policy.network.sensoryWeights[0] = 1;
  const controller = createFixedGraphPolicyController(policy);

  const first = await controller.decide(observation, signal());
  const second = await controller.decide(observation, signal());

  assert.equal(controller.kind, 'connectome');
  assert.equal(controller.activityProvenance, 'simulated-reduced-circuit');
  assert.equal(first.action, 'backward');
  assert.ok(second.activity[0][1] > first.activity[0][1]);
  controller.reset('fixed');
  assert.deepEqual((await controller.decide(observation, signal())).activity, first.activity);
});

test('activity presentation distinguishes absent, model, and reduced-circuit output', () => {
  const frame = { time: 0.2, values: [[101, 0.75]] };

  assert.deepEqual(activityPresentation(null, frame), {
    kind: 'none',
    frame: null,
    heading: 'NO NEURAL OUTPUT',
  });
  assert.deepEqual(activityPresentation({ kind: 'scripted', activityProvenance: 'none' }, frame), {
    kind: 'none',
    frame: null,
    heading: 'NO NEURAL OUTPUT',
  });
  assert.equal(
    activityPresentation({ kind: 'conventional', activityProvenance: 'model-output' }, frame).kind,
    'model-output',
  );
  assert.equal(
    activityPresentation({ kind: 'connectome', activityProvenance: 'simulated-reduced-circuit' }, frame).kind,
    'simulated-reduced-circuit',
  );
  assert.equal(
    activityPresentation({ kind: 'conventional', activityProvenance: 'model-output' }, { time: 0.2, values: [] }).kind,
    'none',
  );
});

test('seed normalization rejects invisible or overlong values and preserves repeatability', () => {
  assert.equal(normalizeSeed('  experiment-042  '), 'experiment-042');
  assert.throws(() => normalizeSeed('   '), /seed/i);
  assert.throws(() => normalizeSeed('x'.repeat(65)), /64/);
  assert.throws(() => normalizeSeed('line\nbreak'), /printable/i);

  const first = reduceGameCommand(
    { game: createGame('experiment-042') },
    { type: 'action', action: 'forward' },
  );
  const reset = reduceGameCommand(first, { type: 'reset' });
  const repeated = reduceGameCommand(first, { type: 'reset', seed: 'experiment-042' });
  const switched = reduceGameCommand(first, { type: 'reset', seed: 'experiment-043' });
  assert.deepEqual(reset.game, repeated.game);
  assert.equal(switched.game.seed, 'experiment-043');
  assert.equal(switched.game.step, 0);
});

test('autonomous speed maps to a deterministic controller schedule', () => {
  assert.equal(autonomousDecisionDelayMs(0.5), 400);
  assert.equal(autonomousDecisionDelayMs(1), 200);
  assert.equal(autonomousDecisionDelayMs(2), 100);
  assert.equal(autonomousDecisionDelayMs(4), 50);
  assert.throws(() => autonomousDecisionDelayMs(3), /speed/i);
});

test('non-empty controller activity fails closed without atlas membership data', () => {
  const decision = { action: 'wait', activity: [[999, 0.5]], diagnostics: {} };

  assert.throws(() => validateControllerDecision(decision), /atlas.*required/i);
  assert.throws(
    () => validateControllerDecision(decision, new Set([101])),
    /visible MaleCNS/i,
  );
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

test('visibility cancellation clears an in-flight request so a fresh decision can commit', () => {
  let run = reduceGameCommand(
    { game: createGame('visibility-resume'), controllerStatus: 'ready' },
    { type: 'controller-start', requestId: 1 },
  );
  assert.equal(run.pendingRequest, 1);

  run = reduceGameCommand(run, { type: 'controller-cancel' });
  assert.equal(run.pendingRequest, null);
  assert.equal(run.controllerStatus, 'ready');
  assert.equal(run.game.step, 0);

  run = reduceGameCommand(run, { type: 'controller-start', requestId: 2 });
  assert.equal(run.pendingRequest, 2);
  run = reduceGameCommand(run, {
    type: 'controller-decision',
    requestId: 2,
    decision: { action: 'forward', activity: [], diagnostics: {} },
  });

  assert.equal(run.game.step, 1);
  assert.equal(run.pendingRequest, null);
});

test('human keyboard shortcuts ignore native interactive targets and their descendants', () => {
  let selector = '';
  const interactive = {
    closest(value) {
      selector = value;
      return { tagName: 'BUTTON' };
    },
  };
  const nonInteractive = { closest: () => null };
  const summary = {
    tagName: 'SUMMARY',
    closest(value) { return value.includes('summary') ? this : null; },
  };
  const summaryChild = {
    closest(value) { return value.includes('summary') ? summary : null; },
  };

  assert.equal(isInteractiveKeyboardTarget(interactive), true);
  assert.equal(isInteractiveKeyboardTarget(summary), true);
  assert.equal(isInteractiveKeyboardTarget(summaryChild), true);
  assert.equal(isInteractiveKeyboardTarget(nonInteractive), false);
  assert.equal(isInteractiveKeyboardTarget(null), false);
  for (const required of ['a[href]', 'button', 'input', 'select', 'textarea', 'summary', '[contenteditable]']) {
    assert.match(selector, new RegExp(required.replaceAll('[', '\\[').replaceAll(']', '\\]')));
  }
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
  disconnect() {
    this.readyState = 3;
    this.dispatchEvent(new Event('close'));
  }
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

test('idle remote disconnect persists and clears prior runtime activity immediately', () => {
  const socket = new FakeSocket();
  const controller = createRemoteController('ws://idle-close', {
    socketFactory: () => socket,
    timeoutMs: 100,
    visibleIds: new Set([101]),
  });
  socket.open();
  let run = reduceGameCommand(
    { game: createGame('idle-close') },
    { type: 'controller-start', requestId: 3 },
  );
  run = reduceGameCommand(run, {
    type: 'controller-decision',
    requestId: 3,
    decision: { action: 'wait', activity: [[101, 0.8]], diagnostics: {} },
  });
  assert.deepEqual(run.activity?.values, [[101, 0.8]]);

  assert.equal(typeof controller.subscribeFailure, 'function');
  const unsubscribe = controller.subscribeFailure((error) => {
    run = reduceGameCommand(run, {
      type: 'controller-connection-failure',
      error: error.message,
    });
  });
  socket.disconnect();

  assert.equal(run.paused, true);
  assert.equal(run.controllerStatus, 'error');
  assert.match(run.controllerError, /disconnected/i);
  assert.equal(run.activity, null);

  let retainedError = '';
  const unsubscribeLate = controller.subscribeFailure((error) => { retainedError = error.message; });
  assert.match(retainedError, /disconnected/i);
  unsubscribeLate();
  unsubscribe();
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
