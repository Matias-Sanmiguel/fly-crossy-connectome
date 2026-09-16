import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const { DECISION_SECONDS, stepGame } = await import('../src/game/simulation.ts');
const {
  WORLD_VERSION,
  difficultyForRow,
  findBoundedGroupWitness,
  generateRows,
  hasBoundedGroupPath,
} = await import('../src/game/world.ts');
const parity = JSON.parse(await readFile(
  new URL('./fixtures/world-generation-v3.json', import.meta.url),
  'utf8',
));

const replayWitness = (seed, rows, actions) => {
  const startRow = rows[0].row;
  let time = 0;
  for (let step = 0; step < Math.max(0, startRow); step += 1) time += DECISION_SECONDS;
  let state = {
    version: WORLD_VERSION,
    seed,
    step: Math.max(0, startRow),
    time,
    fly: { row: startRow, column: 0 },
    score: Math.max(0, startRow),
    lanes: rows,
    terminal: null,
    previousAction: 'wait',
  };
  for (const action of actions) {
    state = stepGame(state, action).state;
    assert.equal(state.terminal, null, `${seed}: ${action} failed at step ${state.step}`);
  }
  assert.equal(state.fly.row, rows.at(-1).row, `${seed}: witness did not reach recovery grass`);
};

const circularGap = (left, right) => {
  const centerDistance = Math.abs(left.position - right.position);
  const wrappedDistance = Math.min(centerDistance, 25 - centerDistance);
  return wrappedDistance - (left.size + right.size) / 2;
};

test('same seed and range produce identical lanes', () => {
  assert.deepEqual(generateRows('lab-7', -5, 40), generateRows('lab-7', -5, 40));
});

test('chunk boundaries do not change generation', () => {
  assert.deepEqual(
    generateRows('lab-7', 0, 40),
    [...generateRows('lab-7', 0, 20), ...generateRows('lab-7', 20, 20)],
  );
});

test('generated lane rows match their requested coordinates', () => {
  assert.deepEqual(generateRows('lab-7', -5, 4).map((lane) => lane.row), [-5, -4, -3, -2]);
});

test('opening rows are safe and hazard groups include recovery rows', () => {
  const rows = generateRows('lab-7', 0, 100);
  assert.deepEqual(rows.slice(0, 3).map((row) => row.kind), ['grass', 'grass', 'grass']);
  assert.ok(rows.every((row, index) => (
    index < 6 || rows.slice(Math.max(0, index - 6), index + 1).some((candidate) => candidate.kind === 'grass')
  )));
});

test('known impassable first group is replaced by a bounded reachable group', () => {
  const group = generateRows('solvability-probe-87', 2, 6);

  assert.equal(hasBoundedGroupPath(group), true);
  assert.deepEqual([group[0].kind, group.at(-1).kind], ['grass', 'grass']);
});

test('audit-replay-21 solver witness survives authoritative floating-point replay', () => {
  const group = generateRows('audit-replay-21', 2, 6);
  const witness = findBoundedGroupWitness(group);

  assert.ok(witness);
  replayWitness('audit-replay-21', group, witness);
});

test('solver witnesses replay through the authoritative transition over a bounded sample', () => {
  for (let seedIndex = 0; seedIndex < 24; seedIndex += 1) {
    for (const groupIndex of [0, 1, 8, 24]) {
      const seed = `audit-replay-${seedIndex}`;
      const start = 2 + groupIndex * 5;
      const group = generateRows(seed, start, 6);
      const witness = findBoundedGroupWitness(group);
      assert.ok(witness, `${seed}, group ${groupIndex}`);
      replayWitness(seed, group, witness);
    }
  }
});

test('generated groups have a bounded route over a deterministic seed sample', () => {
  for (let seedIndex = 0; seedIndex < 24; seedIndex += 1) {
    for (const groupIndex of [0, 1, 8, 24]) {
      const start = 2 + groupIndex * 5;
      assert.equal(
        hasBoundedGroupPath(generateRows(`solvability-property-${seedIndex}`, start, 6)),
        true,
        `seed ${seedIndex}, group ${groupIndex}`,
      );
    }
  }
});

test('documented difficulty parameters increase with forward distance', () => {
  const opening = difficultyForRow(3);
  const middle = difficultyForRow(53);
  const far = difficultyForRow(153);

  assert.ok(middle.minimumSpeed >= opening.minimumSpeed);
  assert.ok(middle.maximumSpeed > opening.maximumSpeed);
  assert.ok(middle.minimumHazards >= opening.minimumHazards);
  assert.ok(far.maximumSpeed > middle.maximumSpeed);
  assert.ok(far.maximumHazards > middle.maximumHazards);
});

test('road traffic uses canonical vehicle sizes with visible space between neighbors', () => {
  const roadLanes = generateRows('experiment-001:auto:e323bb69', -5, 25)
    .filter((lane) => lane.kind === 'road');

  assert.ok(roadLanes.length > 0);
  for (const lane of roadLanes) {
    for (const hazard of lane.hazards) {
      assert.equal(hazard.size, hazard.kind === 'car' ? 2 : 3);
    }
    for (let left = 0; left < lane.hazards.length; left += 1) {
      for (let right = left + 1; right < lane.hazards.length; right += 1) {
        assert.ok(
          circularGap(lane.hazards[left], lane.hazards[right]) >= 0.5,
          `row ${lane.row}: hazards ${left} and ${right} overlap`,
        );
      }
    }
  }
});

test('world generation matches the shared cross-language fixture', () => {
  assert.equal(parity.environmentVersion, WORLD_VERSION);
  for (const parityCase of parity.cases) {
    assert.deepEqual(
      generateRows(parityCase.seed, parityCase.from, parityCase.count),
      parityCase.rows,
    );
  }
});
