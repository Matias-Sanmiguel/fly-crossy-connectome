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

const APPROVED_CONTENT_TEMPLATES = new Set([
  'road,road,road,grass,road,road',
  'road,road,grass,grass,road,road',
  'rail,grass,road,road,road,road',
  'rail,grass,grass,road,road,road',
  'river,river,grass,road,road,grass',
]);

const auditSectionGeneration = () => {
  const rowCounts = { grass: 0, road: 0, rail: 0, river: 0 };
  const sectionCounts = { road: 0, rail: 0, river: 0 };
  const sectionLengths = { road: [], rail: [], river: [] };
  let directUnlikeTransitions = 0;
  let fallbackGroups = 0;
  let groupCount = 0;

  for (let seedIndex = 0; seedIndex < 30; seedIndex += 1) {
    const rows = generateRows(`section-audit-${seedIndex}`, 3, 210);
    let sectionKind = null;
    let sectionLength = 0;
    const finishSection = () => {
      if (sectionKind === null) return;
      sectionLengths[sectionKind].push(sectionLength);
      sectionKind = null;
      sectionLength = 0;
    };

    for (const lane of rows) {
      rowCounts[lane.kind] += 1;
      if (lane.kind === 'grass') {
        finishSection();
      } else if (lane.kind === sectionKind) {
        sectionLength += 1;
      } else {
        if (sectionKind !== null) {
          directUnlikeTransitions += 1;
          finishSection();
        }
        sectionKind = lane.kind;
        sectionLength = 1;
        sectionCounts[lane.kind] += 1;
      }
    }
    finishSection();

    for (let offset = 0; offset < rows.length; offset += 7) {
      const group = rows.slice(offset, offset + 7);
      groupCount += 1;
      assert.equal(group.length, 7);
      assert.equal(group[6].kind, 'grass', 'every section group must end with recovery grass');
      const contentKinds = group.slice(0, 6).map((lane) => lane.kind);
      if (contentKinds.every((kind) => kind === 'grass')) {
        fallbackGroups += 1;
      } else {
        assert.ok(
          APPROVED_CONTENT_TEMPLATES.has(contentKinds.join(',')),
          `unexpected section template: ${contentKinds.join(',')}`,
        );
      }
    }
  }

  const totalSections = Object.values(sectionCounts).reduce((sum, count) => sum + count, 0);
  const totalRows = Object.values(rowCounts).reduce((sum, count) => sum + count, 0);
  return {
    rowCounts,
    sectionCounts,
    sectionLengths,
    sectionShares: Object.fromEntries(
      Object.entries(sectionCounts).map(([kind, count]) => [kind, count / totalSections]),
    ),
    rowShares: Object.fromEntries(
      Object.entries(rowCounts).map(([kind, count]) => [kind, count / totalRows]),
    ),
    directUnlikeTransitions,
    fallbackGroups,
    groupCount,
    fallbackRate: fallbackGroups / groupCount,
  };
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

test('section templates enforce grammar, lengths, frequencies, and recovery grass', () => {
  const audit = auditSectionGeneration();

  assert.equal(audit.directUnlikeTransitions, 0, audit);
  assert.ok(audit.sectionLengths.road.every((length) => length >= 2 && length <= 4));
  assert.ok(audit.sectionLengths.rail.every((length) => length === 1));
  assert.ok(audit.sectionLengths.river.every((length) => length === 2));
  assert.ok(audit.sectionShares.road >= 0.65 && audit.sectionShares.road <= 0.75, audit);
  assert.ok(audit.sectionShares.rail >= 0.19 && audit.sectionShares.rail <= 0.29, audit);
  assert.ok(audit.sectionShares.river >= 0.03 && audit.sectionShares.river <= 0.09, audit);
  assert.ok(audit.sectionShares.river < audit.sectionShares.rail, audit);
  assert.ok(audit.rowShares.road >= 0.48 && audit.rowShares.road <= 0.58, audit);
  assert.ok(audit.rowShares.grass >= 0.31 && audit.rowShares.grass <= 0.41, audit);
  assert.ok(audit.rowShares.rail >= 0.04 && audit.rowShares.rail <= 0.09, audit);
  assert.ok(audit.rowShares.river >= 0.025 && audit.rowShares.river <= 0.065, audit);
  assert.ok(audit.rowShares.river < audit.rowShares.rail, audit);
  assert.ok(audit.fallbackRate <= 0.03, audit);
});

test('known impassable first group is replaced by a bounded reachable group', () => {
  const group = generateRows('solvability-probe-87', 2, 8);

  assert.equal(hasBoundedGroupPath(group), true);
  assert.deepEqual([group[0].kind, group.at(-1).kind], ['grass', 'grass']);
});

test('audit-replay-21 solver witness survives authoritative floating-point replay', () => {
  const group = generateRows('audit-replay-21', 2, 8);
  const witness = findBoundedGroupWitness(group);

  assert.ok(witness);
  replayWitness('audit-replay-21', group, witness);
});

test('solver witnesses replay through the authoritative transition over a bounded sample', () => {
  for (let seedIndex = 0; seedIndex < 24; seedIndex += 1) {
    for (const groupIndex of [0, 1, 8, 24]) {
      const seed = `audit-replay-${seedIndex}`;
      const start = 2 + groupIndex * 7;
      const group = generateRows(seed, start, 8);
      const witness = findBoundedGroupWitness(group);
      assert.ok(witness, `${seed}, group ${groupIndex}`);
      replayWitness(seed, group, witness);
    }
  }
});

test('generated groups have a bounded route over a deterministic seed sample', () => {
  for (let seedIndex = 0; seedIndex < 24; seedIndex += 1) {
    for (const groupIndex of [0, 1, 8, 24]) {
      const start = 2 + groupIndex * 7;
      assert.equal(
        hasBoundedGroupPath(generateRows(`solvability-property-${seedIndex}`, start, 8)),
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
  assert.equal(difficultyForRow(52).level, 0);
  assert.equal(difficultyForRow(53).level, 1);
  assert.equal(difficultyForRow(102).level, 1);
  assert.equal(difficultyForRow(103).level, 2);
  assert.equal(difficultyForRow(152).level, 2);
  assert.equal(difficultyForRow(153).level, 3);
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

test('river sections use the approved Candidate B log profile', () => {
  const riverLanes = Array.from({ length: 30 }, (_, seedIndex) => (
    generateRows(`section-audit-${seedIndex}`, 3, 210)
  )).flat().filter((lane) => lane.kind === 'river');

  assert.ok(riverLanes.length > 0);
  for (const lane of riverLanes) {
    assert.equal(lane.hazards.length, 5);
    assert.ok(lane.hazards.every((hazard) => (
      hazard.kind === 'log'
      && Number.isInteger(hazard.size)
      && hazard.size >= 2
      && hazard.size <= 4
    )));
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
