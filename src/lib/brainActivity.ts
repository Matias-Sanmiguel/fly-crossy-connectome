/** Reuse the measured body-ID lookup and GPU buffer instead of rebuilding a Map per tick. */
export function createBrainActivityBuffer(bodyIds: readonly number[]) {
  const indices = new Map(bodyIds.map((id, index) => [id, index]));
  const values = new Float32Array(bodyIds.length);
  return {
    values,
    write(frame: readonly (readonly [number, number])[]) {
      values.fill(0);
      for (const [id, value] of frame) {
        const index = indices.get(id);
        if (index !== undefined) values[index] = value;
      }
    },
  };
}

export type BrainActivityContrast = 'standard' | 'expo';
export const BRAIN_ACTIVITY_CONTRAST = {
  standard: { gain: 1, floor: 0, activePointSize: 0 },
  expo: { gain: 12, floor: 0.5, activePointSize: 2.6 },
} as const;

/** Display transfer only: zero is blue, weak genuine signals are perceptible. */
export function displayedActivityStrength(value: number, profile: BrainActivityContrast): number {
  if (value <= 0) return 0;
  const { gain, floor } = BRAIN_ACTIVITY_CONTRAST[profile];
  return floor + (1 - floor) * Math.pow(Math.min(1, value * gain), 0.4);
}
