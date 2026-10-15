import type { Action } from '../game/types.ts';

// Installation limit only; the authoritative game rules stay unchanged.
export const VISITOR_MAX_STEPS = 900;
export const VISITOR_API = `${import.meta.env?.BASE_URL ?? '/'}api/visitors`;
export type VisitorRound = { id: string; name: string; seed: string };
export type VisitorIdentity = { name: string; lastName: string; email: string };
export type RankingEntry = { id: string; name: string; score: number; playedAt: string };
export const VISITOR_ACTIONS: readonly Action[] = ['forward', 'backward', 'left', 'right', 'wait'];

export function normalizeVisitorName(value: unknown): string {
  if (typeof value !== 'string' || /\p{C}/u.test(value)) throw Error('Ingresá un nombre válido.');
  const name = value.normalize('NFC').trim().replace(/\s+/gu, ' ');
  if (!name || [...name].length > 32) throw Error('El nombre debe tener entre 1 y 32 caracteres.');
  return name;
}

export function normalizeVisitorIdentity(value: unknown): VisitorIdentity {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error('Ingresá tus datos.');
  const input = value as Record<string, unknown>;
  const name = normalizeVisitorName(input.name);
  let lastName: string;
  try { lastName = normalizeVisitorName(input.lastName); }
  catch { throw Error('El apellido debe tener entre 1 y 32 caracteres válidos.'); }
  if (typeof input.email !== 'string' || /\p{C}/u.test(input.email)) throw Error('Ingresá un mail válido.');
  const email = input.email.trim();
  if (email.length > 254 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/u.test(email)) throw Error('Ingresá un mail válido.');
  return { name, lastName, email };
}

export function validateVisitorActions(value: unknown): Action[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > VISITOR_MAX_STEPS
    || value.some(action => !VISITOR_ACTIONS.includes(action))) {
    throw Error('La partida contiene acciones inválidas.');
  }
  return value as Action[];
}
