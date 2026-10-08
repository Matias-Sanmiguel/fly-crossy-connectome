import { createHash, randomUUID } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { createGame, stepGame } from '../src/game/simulation.ts';
import { WORLD_VERSION } from '../src/game/world.ts';
import { normalizeVisitorName, validateVisitorActions, VISITOR_MAX_STEPS } from '../src/visitor/protocol.ts';
import type { RankingEntry, VisitorRound } from '../src/visitor/protocol.ts';

// Increment when authoritative gameplay changes, not when this UI changes.
const RULESET = `environment-v11-world-v${WORLD_VERSION}`;
export class VisitorError extends Error {
  status: number;
  constructor(message: string, status = 400) { super(message); this.status = status; }
}

export function openVisitorStore(path: string) {
  if (path !== ':memory:') mkdirSync(dirname(path), { recursive: true });
  const db = new DatabaseSync(path, { timeout: 5000 });
  db.exec(`PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS visitor_rounds (
      id TEXT PRIMARY KEY, name TEXT NOT NULL, seed TEXT NOT NULL UNIQUE,
      ruleset TEXT NOT NULL, created_at TEXT NOT NULL, played_at TEXT,
      score INTEGER, steps INTEGER, trace_hash TEXT
    );
    CREATE INDEX IF NOT EXISTS visitor_ranking ON visitor_rounds(ruleset, score DESC, played_at, id);`);
  const find = db.prepare('SELECT * FROM visitor_rounds WHERE id = ?');

  return {
    start(value: unknown): VisitorRound {
      const name = normalizeVisitorName(value);
      const round = { id: randomUUID(), name, seed: `visitor:${randomUUID()}` };
      db.prepare('INSERT INTO visitor_rounds(id,name,seed,ruleset,created_at) VALUES(?,?,?,?,?)')
        .run(round.id, round.name, round.seed, RULESET, new Date().toISOString());
      return round;
    },
    finish(id: string, value: unknown): RankingEntry {
      const actions = validateVisitorActions(value);
      const row = find.get(id);
      if (!row) throw new VisitorError('No se encontró esta ronda.', 404);
      if (row.ruleset !== RULESET) throw new VisitorError('La versión del juego cambió. Iniciá una nueva ronda.', 409);
      const hash = createHash('sha256').update(JSON.stringify(actions)).digest('hex');
      if (row.trace_hash !== null) {
        if (row.trace_hash !== hash) throw new VisitorError('Esta ronda ya tiene un resultado guardado.', 409);
        return { id, name: String(row.name), score: Number(row.score), playedAt: String(row.played_at) };
      }
      let state = createGame(String(row.seed));
      for (const action of actions) {
        if (state.terminal !== null) throw new VisitorError('Hay acciones posteriores al final de la ronda.');
        state = stepGame(state, action).state;
      }
      if (state.terminal === null && actions.length !== VISITOR_MAX_STEPS) {
        throw new VisitorError('La ronda todavía no terminó.');
      }
      const playedAt = new Date().toISOString();
      const updated = db.prepare('UPDATE visitor_rounds SET score=?, steps=?, played_at=?, trace_hash=? WHERE id=? AND trace_hash IS NULL')
        .run(state.score, actions.length, playedAt, hash, id);
      if (Number(updated.changes) === 0) {
        const saved = find.get(id)!;
        if (saved.trace_hash !== hash) throw new VisitorError('Esta ronda ya tiene un resultado guardado.', 409);
        return { id, name: String(saved.name), score: Number(saved.score), playedAt: String(saved.played_at) };
      }
      return { id, name: String(row.name), score: state.score, playedAt };
    },
    ranking(): RankingEntry[] {
      return db.prepare(`SELECT id,name,score,played_at AS playedAt FROM visitor_rounds
        WHERE ruleset=? AND score IS NOT NULL ORDER BY score DESC, played_at ASC, id ASC LIMIT 10`)
        .all(RULESET) as unknown as RankingEntry[];
    },
    close() { db.close(); },
  };
}

export type VisitorStore = ReturnType<typeof openVisitorStore>;
