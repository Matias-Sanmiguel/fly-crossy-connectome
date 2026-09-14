import test from 'node:test';
import assert from 'node:assert/strict';

const { generateRows } = await import('../src/game/world.ts');

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
