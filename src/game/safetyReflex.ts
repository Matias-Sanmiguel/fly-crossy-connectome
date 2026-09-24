import type { ControllerDecision } from './controllers.ts';
import { stepGame } from './simulation.ts';
import type { GameState, StepResult } from './simulation.ts';
import type { Action } from './types.ts';

const ACTION_ORDER: readonly Action[] = ['forward', 'backward', 'left', 'right', 'wait'];

export const SAFETY_REFLEX_STAGNATION_STEPS = 6;
export const SAFETY_REFLEX_EDGE_COLUMN = 3;

export type SafetyReflexResult = {
  decision: ControllerDecision;
  nextNoProgressSteps: number;
};

function finiteLogit(decision: ControllerDecision, action: Action): number | null {
  const value = decision.diagnostics[`logit.${action}`];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function isBlocked(result: StepResult, reason?: 'bounds' | 'scenery'): boolean {
  return result.events.some((event) => (
    event.type === 'blocked' && (reason === undefined || event.reason === reason)
  ));
}

function moved(state: GameState, result: StepResult): boolean {
  return result.state.fly.row !== state.fly.row
    || result.state.fly.column !== state.fly.column;
}

function viableMovement(state: GameState, result: StepResult): boolean {
  return result.state.terminal === null && !isBlocked(result) && moved(state, result);
}

function highestLogitAction(
  decision: ControllerDecision,
  actions: readonly Action[],
): Action | null {
  let selected: Action | null = null;
  let selectedLogit = Number.NEGATIVE_INFINITY;

  for (const action of actions) {
    const logit = finiteLogit(decision, action);
    if (logit === null) continue;
    if (selected === null || logit > selectedLogit) {
      selected = action;
      selectedLogit = logit;
    }
  }
  return selected;
}

function forwardOpensAfter(result: StepResult): boolean {
  if (result.state.terminal !== null) return false;
  const forward = stepGame(result.state, 'forward');
  return (
    forward.state.terminal === null
    && !isBlocked(forward)
    && forward.state.score > result.state.score
  );
}

function chooseLateralBypass(
  state: GameState,
  decision: ControllerDecision,
  outcomes: ReadonlyMap<Action, StepResult>,
): Action | null {
  const candidates = (['left', 'right'] as const)
    .filter((action) => viableMovement(state, outcomes.get(action)!))
    .map((action) => {
      const result = outcomes.get(action)!;
      return {
        action,
        result,
        opensForward: forwardOpensAfter(result),
        movesInward: Math.abs(result.state.fly.column) < Math.abs(state.fly.column),
        absoluteColumn: Math.abs(result.state.fly.column),
        logit: finiteLogit(decision, action) ?? Number.NEGATIVE_INFINITY,
      };
    });

  if (candidates.length === 0) return null;

  candidates.sort((left, right) => {
    if (left.opensForward !== right.opensForward) return left.opensForward ? -1 : 1;
    if (left.movesInward !== right.movesInward) return left.movesInward ? -1 : 1;
    if (left.absoluteColumn !== right.absoluteColumn) {
      return left.absoluteColumn - right.absoluteColumn;
    }
    return right.logit - left.logit;
  });

  return candidates[0]!.action;
}

/**
 * Minimal motor-reflex layer for the connectome demo.
 *
 * The connectome still supplies the action logits. The reflex only handles
 * obvious one-step motor failures: terminal proposals, blocked scenery,
 * repeated outward drift at the edge, and short no-progress streaks.
 */
export function applyConnectomeSafetyReflex(
  state: GameState,
  decision: ControllerDecision,
  noProgressSteps: number,
): SafetyReflexResult {
  if (!Number.isSafeInteger(noProgressSteps) || noProgressSteps < 0) {
    throw Error('Safety reflex no-progress counter must be a non-negative integer.');
  }
  if (state.terminal !== null) return { decision, nextNoProgressSteps: 0 };

  const proposedResult = stepGame(state, decision.action);
  let executedAction = decision.action;
  let blockedVeto = false;

  // V10: this layer is not allowed to make traffic/navigation decisions.
  if (decision.action !== 'wait' && isBlocked(proposedResult)) {
    const outcomes = new Map<Action, StepResult>();
    for (const action of ACTION_ORDER) outcomes.set(action, stepGame(state, action));
    const viable = ACTION_ORDER.filter((action) => (
      action !== 'wait'
      && viableMovement(state, outcomes.get(action)!)
      && !isBlocked(outcomes.get(action)!)
    ));
    const replacement = highestLogitAction(decision, viable);
    if (replacement !== null) {
      executedAction = replacement;
      blockedVeto = replacement !== decision.action;
    }
  }

  const executedResult = stepGame(state, executedAction);
  const nextNoProgressSteps = executedResult.state.terminal !== null
    ? 0
    : executedResult.state.score > state.score ? 0 : noProgressSteps + 1;

  return {
    decision: {
      ...decision,
      action: executedAction,
      diagnostics: {
        ...decision.diagnostics,
        'reflex.applied': executedAction === decision.action ? 0 : 1,
        'reflex.terminalVeto': 0,
        'reflex.blockedVeto': blockedVeto ? 1 : 0,
        'reflex.sceneryBypass': blockedVeto ? 1 : 0,
        'reflex.edgeCorrection': 0,
        'reflex.stagnationOverride': 0,
        'reflex.noProgressBefore': noProgressSteps,
        'reflex.proposedActionIndex': ACTION_ORDER.indexOf(decision.action),
        'reflex.executedActionIndex': ACTION_ORDER.indexOf(executedAction),
      },
    },
    nextNoProgressSteps,
  };
}
