import type { GameState } from './simulation.ts';

export const PROGRESS_REWARD = 0.35;
export const TERMINAL_PENALTY = -5;
export const STEP_COST = -0.01;
export const STAGNATION_COST = 0;
export const WAIT_COST = -0.03;
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

  const waiting =
    next.previousAction === 'wait'
    && !productiveCarry
      ? WAIT_COST
      : 0;

  const blocked = events.some((event) => event.type === 'blocked')
    ? BLOCKED_COST
    : 0;

  return (
    progress
    + terminal
    + STEP_COST
    + waiting
    + blocked
  );
}