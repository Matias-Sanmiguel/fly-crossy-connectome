import type { Action } from '../game/types.ts';

export type KeyboardHighlightTarget = 'w' | 'a' | 'd' | 'arrow-up' | 'arrow-left' | 'arrow-right';

const ACTION_TARGETS: Readonly<Record<Action, readonly KeyboardHighlightTarget[]>> = {
  forward: ['w', 'arrow-up'],
  backward: [],
  left: ['a', 'arrow-left'],
  right: ['d', 'arrow-right'],
  wait: [],
};

export function keyboardHighlightTargets(action: Action): readonly KeyboardHighlightTarget[] {
  return ACTION_TARGETS[action];
}

