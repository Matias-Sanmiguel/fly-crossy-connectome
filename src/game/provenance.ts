import type { ActivityFrame } from '../lib/replay.ts';

export type ActivityProvenance = 'none' | 'model-output' | 'simulated-reduced-circuit';

type ControllerActivitySource = {
  kind: string;
  activityProvenance: ActivityProvenance;
};

export type ActivityPresentation = {
  kind: ActivityProvenance;
  frame: ActivityFrame | null;
  heading: 'NO NEURAL OUTPUT' | 'MODEL OUTPUT' | 'SIMULATED REDUCED-CIRCUIT ACTIVITY';
};

/** Keep anatomy visible while failing closed on absent or non-neural output. */
export function activityPresentation(
  controller: ControllerActivitySource | null,
  frame: ActivityFrame | null,
): ActivityPresentation {
  if (!controller || controller.activityProvenance === 'none' || !frame?.values.length) {
    return { kind: 'none', frame: null, heading: 'NO NEURAL OUTPUT' };
  }
  if (controller.activityProvenance === 'simulated-reduced-circuit') {
    return {
      kind: 'simulated-reduced-circuit',
      frame,
      heading: 'SIMULATED REDUCED-CIRCUIT ACTIVITY',
    };
  }
  return { kind: 'model-output', frame, heading: 'MODEL OUTPUT' };
}
