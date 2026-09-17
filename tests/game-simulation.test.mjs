import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { createGame, stepGame } from '../src/game/simulation.ts';
import { reward } from '../src/game/reward.ts';

const lane = (row, kind, options = {}) => ({
  row,
  kind,
  hazards: [],
  ...options,
});

const stateWith = ({ fly = { row: 3, column: 0 }, lanes, ...overrides }) => ({
  version: 1,
  seed: 'fixture',
  step: 0,
  time: 0,
  fly,
  score: fly.row,
  lanes,
  terminal: null,
  previousAction: 'wait',
  ...overrides,
});

test('forward movement advances the immutable fixed-step state', () => {
  const initial = createGame('road-test');
  const moved = stepGame(initial, 'forward');

  assert.equal(moved.state.fly.row, initial.fly.row + 1);
  assert.equal(moved.state.step, 1);
  assert.equal(moved.state.time, 0.2);
  assert.equal(moved.state.score, 1);
  assert.equal(initial.fly.row, 0, 'stepGame must not mutate input');
  assert.equal(initial.step, 0, 'stepGame must not mutate input');
});

test('a vehicle crossing the fly between decisions causes a swept collision', () => {
  const initial = stateWith({
    lanes: [lane(3, 'road', {
      direction: 1,
      speed: 10,
      phase: 0,
      hazards: [{ kind: 'car', position: -1.5, size: 1 }],
    })],
  });

  const result = stepGame(initial, 'wait');

  assert.equal(result.state.terminal, 'vehicle');
  assert.ok(result.events.some((event) => event.type === 'terminal' && event.reason === 'vehicle'));
});

test('moving off a road cannot evade a vehicle sweeping the starting cell', () => {
  const initial = stateWith({
    lanes: [
      lane(3, 'road', {
        direction: 1,
        speed: 10,
        phase: 0,
        hazards: [{ kind: 'car', position: -1.5, size: 1 }],
      }),
      lane(4, 'grass'),
    ],
  });

  const result = stepGame(initial, 'forward');

  assert.equal(result.state.terminal, 'vehicle');
  assert.deepEqual(result.state.fly, initial.fly, 'impact occurs before movement');
});

test('moving onto a road is safe when the vehicle passed before arrival', () => {
  const initial = stateWith({
    fly: { row: 2, column: 0 },
    lanes: [
      lane(2, 'grass'),
      lane(3, 'road', {
        direction: 1,
        speed: 15,
        phase: 0,
        hazards: [{ kind: 'car', position: -1.5, size: 1 }],
      }),
    ],
  });

  const result = stepGame(initial, 'forward');

  assert.equal(result.state.terminal, null);
  assert.deepEqual(result.state.fly, { row: 3, column: 0 });
});

test('moving hazards continue circulating during long episodes', () => {
  const initial = stateWith({
    time: 25,
    lanes: [lane(3, 'road', {
      direction: 1,
      speed: 1,
      phase: 0,
      hazards: [{ kind: 'car', position: 0, size: 1 }],
    })],
  });

  assert.equal(stepGame(initial, 'wait').state.terminal, 'vehicle');
});

test('rail lanes warn before a train and terminate on swept impact', () => {
  const warning = stateWith({
    lanes: [lane(3, 'rail', {
      direction: 1,
      speed: 2,
      phase: 0,
      hazards: [{ kind: 'train', position: -4, size: 4 }],
    })],
  });
  const impact = stateWith({
    lanes: [lane(3, 'rail', {
      direction: 1,
      speed: 10,
      phase: 0,
      hazards: [{ kind: 'train', position: -2.5, size: 4 }],
    })],
  });

  const warned = stepGame(warning, 'wait');
  const hit = stepGame(impact, 'wait');

  assert.equal(warned.state.terminal, null);
  assert.ok(warned.events.some((event) => event.type === 'train-warning' && event.row === 3));
  assert.equal(hit.state.terminal, 'train');
});

test('waiting on a river log inherits its displacement', () => {
  const initial = stateWith({
    lanes: [lane(3, 'river', {
      direction: 1,
      speed: 1,
      phase: 0,
      hazards: [{ kind: 'log', position: -0.5, size: 2 }],
    })],
  });

  const result = stepGame(initial, 'wait');

  assert.equal(result.state.terminal, null);
  assert.equal(result.state.fly.column, 0.2);
  assert.ok(result.events.some((event) => event.type === 'carried' && event.displacement === 0.2));
});

test('unsupported water is terminal', () => {
  const initial = stateWith({
    lanes: [lane(3, 'river', {
      direction: 1,
      speed: 1,
      phase: 0,
      hazards: [{ kind: 'log', position: 3, size: 1 }],
    })],
  });

  assert.equal(stepGame(initial, 'wait').state.terminal, 'water');
});

test('manual movement beyond the horizontal playfield is blocked', () => {
  const initial = stateWith({
    fly: { row: 3, column: 5 },
    lanes: [lane(3, 'grass')],
  });

  const result = stepGame(initial, 'right');
  assert.equal(result.state.terminal, null);
  assert.deepEqual(result.state.fly, initial.fly);
  assert.ok(result.events.some((event) => (
    event.type === 'blocked' && event.reason === 'bounds'
  )));
});

test('waiting advances the clock without moving the fly', () => {
  const initial = stateWith({ lanes: [lane(3, 'grass')] });
  const result = stepGame(initial, 'wait');

  assert.deepEqual(result.state.fly, initial.fly);
  assert.equal(result.state.step, 1);
  assert.equal(result.state.time, 0.2);
  assert.equal(result.state.previousAction, 'wait');
});

test('moving backward never decreases the furthest-row score', () => {
  const initial = stateWith({
    fly: { row: 4, column: 0 },
    score: 7,
    lanes: [lane(3, 'grass'), lane(4, 'grass')],
  });

  assert.equal(stepGame(initial, 'backward').state.score, 7);
});

test('world rows extend ahead, retain future rows, and prune only far behind', () => {
  const initial = stateWith({
    fly: { row: 100, column: 0 },
    lanes: [lane(50, 'grass'), lane(100, 'grass'), lane(130, 'grass')],
  });
  const result = stepGame(initial, 'wait');
  const rows = result.state.lanes.map((candidate) => candidate.row);

  assert.equal(rows.includes(50), false);
  assert.equal(rows.includes(115), true);
  assert.equal(rows.includes(130), true, 'rows ahead must not be pruned');
});

test(
  'reward combines progress, time, stagnation, and terminal costs',
  () => {
    const previous = stateWith({
      fly: { row: 0, column: 0 },
      score: 2,
      lanes: [],
    });

    const progress = {
      ...previous,
      score: 4,
      step: 1,
      time: 0.2,
      previousAction: 'forward',
    };

    const stagnant = {
      ...progress,
      step: 2,
      time: 0.4,
      previousAction: 'wait',
    };

    const terminal = {
      ...stagnant,
      terminal: 'vehicle',
    };

    assert.equal(
      reward(previous, progress),
      1.99,
    );

    assert.equal(
      reward(progress, stagnant),
      -0.11,
    );

    assert.equal(
      reward(stagnant, terminal),
      -10.11,
    );
  },
);

test('terminal states do not advance again', () => {
  const initial = stateWith({ lanes: [lane(3, 'grass')], terminal: 'bounds' });
  const result = stepGame(initial, 'forward');

  assert.equal(result.state, initial);
  assert.equal(result.reward, 0);
  assert.deepEqual(result.events, []);
});

test('versioned episode fixture reproduces score, rewards, and terminal reason', () => {
  const fixture = JSON.parse(readFileSync(
    new URL('./fixtures/episode-v2.json', import.meta.url),
    'utf8',
  ));
  let state = createGame(fixture.seed);
  const rewards = [];
  const steps = [];

  for (const action of fixture.actions) {
    const result = stepGame(state, action);
    state = result.state;
    rewards.push(result.reward);
    steps.push({
      reward: result.reward,
      score: state.score,
      terminalReason: state.terminal,
    });
  }

  assert.equal(fixture.version, 2);
  assert.equal(
    rewards.length,
    fixture.rewards.length,
  );

  for (let index = 0; index < rewards.length; index += 1) {
    assert.ok(
      Math.abs(
        rewards[index] - fixture.rewards[index],
      ) < 1e-12,
      `reward mismatch at step ${index}`,
    );
  }
  assert.equal(
    steps.length,
    fixture.steps.length,
  );

  for (let index = 0; index < steps.length; index += 1) {
    assert.ok(
      Math.abs(
        steps[index].reward
        - fixture.steps[index].reward,
      ) < 1e-12,
      `step reward mismatch at step ${index}`,
    );

    assert.equal(
      steps[index].score,
      fixture.steps[index].score,
      `score mismatch at step ${index}`,
    );

    assert.equal(
      steps[index].terminalReason,
      fixture.steps[index].terminalReason,
      `terminal reason mismatch at step ${index}`,
    );
  }
  assert.equal(state.score, fixture.expectedScore);
  assert.equal(state.terminal, fixture.terminalReason);
});
