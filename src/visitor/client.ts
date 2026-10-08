import { VISITOR_API } from './protocol.ts';
import type { RankingEntry, VisitorRound } from './protocol.ts';
import type { Action } from '../game/types.ts';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${VISITOR_API}${path}`, { cache: 'no-store', ...options });
  const contentType = response.headers.get('content-type');
  if (!contentType?.includes('application/json')) throw Error('El ranking necesita el servidor local. Ejecutá npm run dev o npm run preview.');
  const value = await response.json();
  if (!response.ok) throw Error(value.error ?? 'No se pudo conectar con el ranking.');
  return value as T;
}
export const loadRanking = (signal?: AbortSignal) => request<{ entries: RankingEntry[] }>('/ranking', { signal });
export const startVisitorRound = (name: string) => request<VisitorRound>('/rounds', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }),
});
export const finishVisitorRound = (id: string, actions: readonly Action[]) => request<RankingEntry>(`/rounds/${id}/finish`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ actions }),
});
