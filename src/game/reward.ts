import type { GameState } from './simulation.ts';

export const PROGRESS_REWARD = 1;
export const TERMINAL_PENALTY = -10;
export const STEP_COST = -0.01;
export const STAGNATION_COST = -0.10;
export const BLOCKED_COST = -0.10;

/** Shared reward used by every controller and training environment. */
export function reward(
  previous: GameState,
  next: GameState,
  events: readonly { type: string }[] = [],
): number {
  const progress =
    Math.max(
      0,
      next.score - previous.score,
    ) * PROGRESS_REWARD;

  const terminal =
    previous.terminal === null
    && next.terminal !== null
      ? TERMINAL_PENALTY
      : 0;

  const productiveCarry =
    next.terminal === null
    && events.some((event) => event.type === 'carried');

  const stagnation =
    next.score <= previous.score
    && !productiveCarry
      ? STAGNATION_COST
      : 0;

  const blocked = events.some((event) => event.type === 'blocked')
    ? BLOCKED_COST
    : 0;

  return (
    progress
    + terminal
    + STEP_COST
    + stagnation
    + blocked
  );
}