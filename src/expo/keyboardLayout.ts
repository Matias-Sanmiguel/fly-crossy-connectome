import type { Action } from '../game/types.ts';

export type KeyboardHighlightTarget = 'w' | 'a' | 's' | 'd';

const ACTION_TARGETS: Readonly<Record<Action, readonly KeyboardHighlightTarget[]>> = {
  forward: ['w'],
  backward: ['s'],
  left: ['a'],
  right: ['d'],
  wait: [],
};

export function keyboardHighlightTargets(action: Action): readonly KeyboardHighlightTarget[] {
  return ACTION_TARGETS[action];
}
