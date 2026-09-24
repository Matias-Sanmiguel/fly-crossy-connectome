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
  if (state.terminal !== null) {
    return { decision, nextNoProgressSteps: 0 };
  }

  const outcomes = new Map<Action, StepResult>();
  for (const action of ACTION_ORDER) outcomes.set(action, stepGame(state, action));

  const proposedAction = decision.action;
  const proposedResult = outcomes.get(proposedAction)!;
  const viableActions = ACTION_ORDER.filter((action) => (
    action === 'wait'
      ? outcomes.get(action)!.state.terminal === null
      : viableMovement(state, outcomes.get(action)!)
  ));
  const viableMovingActions = ACTION_ORDER.filter(
    (action) => action !== 'wait' && viableMovement(state, outcomes.get(action)!),
  );

  let executedAction = proposedAction;
  let terminalVeto = false;
  let blockedVeto = false;
  let sceneryBypass = false;
  let edgeCorrection = false;
  let stagnationOverride = false;

  // Immediate death veto. The replacement still comes from the connectome logits.
  if (proposedResult.state.terminal !== null && viableActions.length > 0) {
    const safeChoice = highestLogitAction(decision, viableActions);
    if (safeChoice !== null) {
      executedAction = safeChoice;
      terminalVeto = executedAction !== proposedAction;
    }
  }

  // A blocked movement is not useful progress. Prefer a lateral bypass that
  // actually opens a forward cell; otherwise use the best viable model action.
  let executedResult = outcomes.get(executedAction)!;
  if (
    executedAction !== 'wait'
    && isBlocked(executedResult)
    && viableMovingActions.length > 0
  ) {
    const bypass = chooseLateralBypass(state, decision, outcomes)
      ?? highestLogitAction(decision, viableMovingActions);
    if (bypass !== null) {
      executedAction = bypass;
      blockedVeto = executedAction !== proposedAction;
    }
  }

  // If a tree/rock is directly ahead and the model just waits in front of it,
  // sidestep immediately instead of waiting for the stagnation threshold.
  const forwardResult = outcomes.get('forward')!;
  const forwardBlockedByScenery = isBlocked(forwardResult, 'scenery');
  executedResult = outcomes.get(executedAction)!;
  if (
    forwardBlockedByScenery
    && executedResult.state.score <= state.score
    && (executedAction === 'wait' || isBlocked(executedResult))
  ) {
    const bypass = chooseLateralBypass(state, decision, outcomes);
    if (bypass !== null) {
      executedAction = bypass;
      sceneryBypass = true;
    }
  }

  // Stop the learned right-only drift from pinning the fly to a wall.
  // Near an edge, an outward lateral move is replaced by the inward move when
  // the inward move is viable and does not make the local route worse.
  const outward: Action | null = state.fly.column >= SAFETY_REFLEX_EDGE_COLUMN
    ? 'right'
    : state.fly.column <= -SAFETY_REFLEX_EDGE_COLUMN
      ? 'left'
      : null;
  const inward: Action | null = outward === 'right'
    ? 'left'
    : outward === 'left'
      ? 'right'
      : null;

  if (outward !== null && inward !== null && executedAction === outward) {
    const outwardResult = outcomes.get(outward)!;
    const inwardResult = outcomes.get(inward)!;
    if (
      viableMovement(state, inwardResult)
      && (
        !viableMovement(state, outwardResult)
        || forwardOpensAfter(inwardResult)
        || !forwardOpensAfter(outwardResult)
      )
    ) {
      executedAction = inward;
      edgeCorrection = true;
    }
  }

  // Short anti-stagnation reflex. Unlike v1, FORWARD must really advance the
  // score; a blocked FORWARD no longer qualifies as a safe escape.
  executedResult = outcomes.get(executedAction)!;
  if (
    noProgressSteps >= SAFETY_REFLEX_STAGNATION_STEPS
    && executedResult.state.score <= state.score
  ) {
    if (
      forwardResult.state.terminal === null
      && !isBlocked(forwardResult)
      && forwardResult.state.score > state.score
    ) {
      executedAction = 'forward';
      stagnationOverride = true;
    } else if (forwardBlockedByScenery) {
      const bypass = chooseLateralBypass(state, decision, outcomes);
      if (bypass !== null) {
        executedAction = bypass;
        stagnationOverride = true;
        sceneryBypass = true;
      }
    }
  }

  executedResult = outcomes.get(executedAction)!;
  const nextNoProgressSteps = executedResult.state.terminal !== null
    ? 0
    : executedResult.state.score > state.score
      ? 0
      : noProgressSteps + 1;

  const adjustedDecision: ControllerDecision = {
    ...decision,
    action: executedAction,
    diagnostics: {
      ...decision.diagnostics,
      'reflex.applied': executedAction === proposedAction ? 0 : 1,
      'reflex.terminalVeto': terminalVeto ? 1 : 0,
      'reflex.blockedVeto': blockedVeto ? 1 : 0,
      'reflex.sceneryBypass': sceneryBypass ? 1 : 0,
      'reflex.edgeCorrection': edgeCorrection ? 1 : 0,
      'reflex.stagnationOverride': stagnationOverride ? 1 : 0,
      'reflex.noProgressBefore': noProgressSteps,
      'reflex.proposedActionIndex': ACTION_ORDER.indexOf(proposedAction),
      'reflex.executedActionIndex': ACTION_ORDER.indexOf(executedAction),
    },
  };

  return { decision: adjustedDecision, nextNoProgressSteps };
}
