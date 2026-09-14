import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const { difficultyForRow, generateRows, hasBoundedGroupPath } = await import('../src/game/world.ts');
const parity = JSON.parse(await readFile(
  new URL('./fixtures/world-generation-v2.json', import.meta.url),
  'utf8',
));

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

test('generated groups have a bounded route over a deterministic seed sample', () => {
  for (let seedIndex = 0; seedIndex < 96; seedIndex += 1) {
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

test('world generation matches the shared cross-language fixture', () => {
  assert.equal(parity.environmentVersion, 2);
  for (const parityCase of parity.cases) {
    assert.deepEqual(
      generateRows(parityCase.seed, parityCase.from, parityCase.count),
      parityCase.rows,
    );
  }
});
