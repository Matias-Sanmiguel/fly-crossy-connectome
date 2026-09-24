import { DECISION_SECONDS } from './transition.ts';

export function interpolatedSimulationTime(
  stateTime: number,
  step: number,
  elapsedMilliseconds: number,
): number {
  if (!Number.isFinite(stateTime) || stateTime < 0) {
    throw Error('Render state time must be a finite non-negative number.');
  }
  if (!Number.isSafeInteger(step) || step < 0) {
    throw Error('Render step must be a non-negative integer.');
  }
  if (!Number.isFinite(elapsedMilliseconds) || elapsedMilliseconds < 0) {
    throw Error('Render elapsed time must be finite and non-negative.');
  }

  if (step === 0) return stateTime;

  const fraction = Math.min(
    1,
    elapsedMilliseconds / (DECISION_SECONDS * 1_000),
  );
  const transitionStart = stateTime - DECISION_SECONDS;
  return transitionStart + fraction * DECISION_SECONDS;
}
