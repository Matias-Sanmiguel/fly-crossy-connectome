import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'node:http';
import { openVisitorStore } from '../server/visitorStore.ts';
import { createVisitorApi } from '../server/visitorApi.ts';
import { createGame, stepGame } from '../src/game/simulation.ts';
import { normalizeVisitorName, validateVisitorActions, VISITOR_MAX_STEPS } from '../src/visitor/protocol.ts';

function completedTrace(seed) {
  let state = createGame(seed);
  const actions = [];
  while (!state.terminal && actions.length < VISITOR_MAX_STEPS) {
    actions.push('forward');
    state = stepGame(state, 'forward').state;
  }
  return { actions, state };
}

test('visitor names are normalized, bounded, Unicode-safe and never rendered as HTML', async () => {
  assert.equal(normalizeVisitorName('  Nicolás   Giordano  '), 'Nicolás Giordano');
  assert.equal(normalizeVisitorName('Mati\u0301as'), 'Matías');
  for (const value of ['', '   ', 4, null, 'a'.repeat(33), 'a\nB', 'a\u202eB']) assert.throws(() => normalizeVisitorName(value));
  assert.throws(() => validateVisitorActions(['fly']));
  assert.throws(() => validateVisitorActions([]));
  assert.throws(() => validateVisitorActions(Array(VISITOR_MAX_STEPS + 1).fill('wait')));
  const source = await readFile(new URL('../src/VisitorApp.tsx', import.meta.url), 'utf8');
  assert.doesNotMatch(source, /dangerouslySetInnerHTML|create.*Controller|useAtlas|Math\.random/);
  assert.match(source, /mode: 'human'/);
  assert.match(source, /\{entry\.name\}/);
});

test('rounds get independent server seeds, unfinished traces cannot enter the ranking, and scores are computed by simulation', () => {
  const store = openVisitorStore(':memory:');
  try {
    const round = store.start('Nico'), second = store.start('Mati');
    assert.notEqual(round.id, second.id);
    assert.notEqual(round.seed, second.seed);
    assert.ok(round.seed.startsWith('visitor:') && round.seed.length <= 64);
    assert.throws(() => store.finish(round.id, ['wait']), /todavía no terminó/);
    assert.deepEqual(store.ranking(), []);
    const { actions, state } = completedTrace(round.seed);
    const entry = store.finish(round.id, actions);
    assert.equal(entry.score, state.score);
    assert.equal(entry.name, 'Nico');
    assert.deepEqual(store.finish(round.id, actions), entry);
    assert.equal(store.ranking().length, 1);
    assert.throws(() => store.finish(round.id, ['left']), /ya tiene un resultado/);
    assert.throws(() => store.finish('unknown', ['wait']), /No se encontró/);
    const otherTrace = completedTrace(second.seed);
    if (otherTrace.state.terminal) assert.throws(() => store.finish(second.id, [...otherTrace.actions, 'wait']), /posteriores/);
  } finally { store.close(); }
});

test('the installation time limit accepts the same bounded trace without changing game rules', () => {
  const store = openVisitorStore(':memory:');
  try {
    const round = store.start('Paciente');
    const actions = Array(VISITOR_MAX_STEPS).fill('wait');
    const original = createGame(round.seed);
    assert.equal(original.terminal, null);
    const result = store.finish(round.id, actions);
    assert.equal(result.score, 0);
    assert.deepEqual(store.finish(round.id, actions), result);
  } finally { store.close(); }
});

test('SQLite retains completed rounds after reopening, orders scores and caps the public ranking at 10', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'fly-visitor-test-'));
  const path = join(directory, 'ranking.sqlite');
  let store = openVisitorStore(path);
  try {
    for (let i = 0; i < 22; i++) {
      const round = store.start(`Visitante ${i}`);
      store.finish(round.id, completedTrace(round.seed).actions);
    }
    const ranking = store.ranking();
    assert.equal(ranking.length, 10);
    assert.ok(ranking.every((entry, i) => i === 0 || ranking[i - 1].score >= entry.score));
    store.close(); store = openVisitorStore(path);
    assert.deepEqual(store.ranking(), ranking);
  } finally { store.close(); await rm(directory, { recursive: true, force: true }); }
});

test('HTTP API validates input, isolates unrelated routes, rejects cross-origin writes and retries saves exactly once', async () => {
  const store = openVisitorStore(':memory:');
  const api = createVisitorApi(store);
  const server = createServer((req, res) => { void api(req, res, () => { res.writeHead(404); res.end(); }); });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}`;
  const post = (path, data, extra = {}) => fetch(url + path, { method: 'POST', headers: { 'Content-Type': 'application/json', ...extra }, body: JSON.stringify(data) });
  try {
    assert.equal((await fetch(url + '/unrelated')).status, 404);
    assert.equal((await post('/api/visitors/rounds', { name: 'Nico' }, { Origin: 'https://untrusted.example' })).status, 403);
    assert.equal((await post('/api/visitors/rounds', { name: '' })).status, 400);
    const malformed = await fetch(url + '/api/visitors/rounds', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{' });
    assert.equal(malformed.status, 400);
    assert.equal((await post('/api/visitors/rounds', { name: 'x'.repeat(40_000) })).status, 413);
    const start = await post('/api/visitors/rounds', { name: '<b>Nico</b>' });
    assert.equal(start.status, 201);
    const round = await start.json();
    const { actions, state } = completedTrace(round.seed);
    const payload = { actions, score: 999999 };
    const result = await post(`/api/visitors/rounds/${round.id}/finish`, payload);
    assert.equal(result.status, 200);
    const entry = await result.json();
    assert.equal(entry.score, state.score);
    assert.equal((await post(`/api/visitors/rounds/${round.id}/finish`, payload)).status, 200);
    const ranking = await fetch(url + '/api/visitors/ranking');
    assert.equal(ranking.headers.get('cache-control'), 'no-store');
    assert.deepEqual((await ranking.json()).entries, [entry]);
  } finally { await new Promise(resolve => server.close(resolve)); store.close(); }
});

test('visitor routing is independent of expo and ranking/controls are scoped to the new screen', async () => {
  const main = await readFile(new URL('../src/main.tsx', import.meta.url), 'utf8');
  const source = await readFile(new URL('../src/VisitorApp.tsx', import.meta.url), 'utf8');
  const css = await readFile(new URL('../src/visitor.css', import.meta.url), 'utf8');
  assert.match(main, /get\('play'\) === '1'/);
  assert.match(main, /visitorMode\s*\? <VisitorApp/);
  assert.match(source, /lastStep\.current \+ 1/);
  assert.match(source, /actions\.current\.push\(game\.state\.previousAction\)/);
  assert.match(source, /submission\.current/);
  assert.match(source, /startPending\.current/);
  assert.match(source, /TOP 10/);
  assert.doesNotMatch(source, /Tu próxima ronda|El camino empieza acá\.|TOP 20/);
  assert.match(css, /@media \(min-width: 768px\)/);
  assert.match(css, /min-height: 48px/);
  assert.doesNotMatch(css, /\.expo-/);
});

test('visitor top ten uses compact rows and fits shorter desktop viewports without hiding results', async () => {
  const css = await readFile(new URL('../src/visitor.css', import.meta.url), 'utf8');
  assert.match(css, /\.visitor-ranking li\s*\{[^}]*padding: 0\.375rem 0/);
  assert.doesNotMatch(css, /\.visitor-ranking li:first-child\s*\{[^}]*padding-block/);
  assert.match(css, /@media \(min-width: 768px\) and \(max-height: 900px\)/);
  assert.match(css, /\.visitor-game-column:has\(\.visitor-round-heading\) \.visitor-stage/);
  assert.doesNotMatch(css, /\.visitor-ranking[^{}]*\{[^}]*overflow:\s*hidden/);
});
