import type { ActivityFrame } from '../lib/replay.ts';
import type { NeuralActivityState } from './client.ts';

export function neuralActivityFrame(
  activity: NeuralActivityState | null,
): ActivityFrame | null {
  if (!activity || !Number.isFinite(activity.simulationTime)) return null;

  const seen = new Set<number>();
  const values: [number, number][] = [];
  for (const update of activity.updates) {
    if (
      !Number.isSafeInteger(update.neuronId)
      || update.neuronId < 0
      || !Number.isFinite(update.value)
      || seen.has(update.neuronId)
    ) {
      return null;
    }
    seen.add(update.neuronId);
    values.push([update.neuronId, update.value]);
  }

  return { time: activity.simulationTime, values };
}
