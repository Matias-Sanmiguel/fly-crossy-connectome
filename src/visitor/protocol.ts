import type { Action } from '../game/types.ts';

// Installation limit only; the authoritative game rules stay unchanged.
export const VISITOR_MAX_STEPS = 900;
export const VISITOR_API = `${import.meta.env?.BASE_URL ?? '/'}api/visitors`;
export type VisitorRound = { id: string; name: string; seed: string };
export type RankingEntry = { id: string; name: string; score: number; playedAt: string };
export const VISITOR_ACTIONS: readonly Action[] = ['forward', 'backward', 'left', 'right', 'wait'];

export function normalizeVisitorName(value: unknown): string {
  if (typeof value !== 'string' || /\p{C}/u.test(value)) throw Error('Ingresá un nombre válido.');
  const name = value.normalize('NFC').trim().replace(/\s+/gu, ' ');
  if (!name || [...name].length > 32) throw Error('El nombre debe tener entre 1 y 32 caracteres.');
  return name;
}

export function validateVisitorActions(value: unknown): Action[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > VISITOR_MAX_STEPS
    || value.some(action => !VISITOR_ACTIONS.includes(action))) {
    throw Error('La partida contiene acciones inválidas.');
  }
  return value as Action[];
}
