import {
  createGame,
  stepGame,
  type GameEvent,
  type GameState,
} from '../game/simulation.ts';

import type {
  Action,
  ActionResult,
  Contact,
  Intention,
  KeyName,
  ResetComplete,
} from './protocol.ts';

export type StationPhase =
  | 'ready'
  | 'acting'
  | 'paused'
  | 'error';

export type PendingStationIntention = {
  id: string;
  requestedAction: Action;
  requestedKey: KeyName | null;
};

export type StationState = {
  game: GameState;
  events: readonly GameEvent[];
  reward: number;

  phase: StationPhase;

  pending: PendingStationIntention | null;
  completedIds: readonly string[];

  lastContact: Contact | null;
  lastAppliedAction: Action | null;

  error: string | null;
};

export type StationEvent =
  | Intention
  | Contact
  | ActionResult
  | ResetComplete;

const COMPLETION_HISTORY = 256;

const ACTION_TO_KEY: Readonly<
  Record<Exclude<Action, 'wait'>, KeyName>
> = {
  forward: 'W',
  backward: 'S',
  left: 'A',
  right: 'D',
};

export class StationFault extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'StationFault';
  }
}

function requestedKey(action: Action): KeyName | null {
  return action === 'wait'
    ? null
    : ACTION_TO_KEY[action];
}

function rememberCompletion(
  completed: readonly string[],
  intentionId: string,
): readonly string[] {
  const next = [...completed, intentionId];

  if (next.length <= COMPLETION_HISTORY) {
    return next;
  }

  return next.slice(next.length - COMPLETION_HISTORY);
}

function fail(
  state: StationState,
  message: string,
): StationState {
  return {
    ...state,
    phase: 'error',
    pending: null,
    error: message,
  };
}

export function createStation(seed: string): StationState {
  return {
    game: createGame(seed),
    events: [],
    reward: 0,

    phase: 'ready',

    pending: null,
    completedIds: [],

    lastContact: null,
    lastAppliedAction: null,

    error: null,
  };
}

export function reduceStation(
  state: StationState,
  event: StationEvent,
): StationState {
  if (state.phase === 'error') {
    return state;
  }

  if (event.type === 'intention') {
    if (state.pending !== null) {
      return fail(
        state,
        'Simulation sent an overlapping intention.',
      );
    }

    return {
      ...state,
      events: [],
      reward: 0,
      phase: 'acting',
      pending: {
        id: event.intentionId,
        requestedAction: event.action,
        requestedKey: requestedKey(event.action),
      },
      lastContact: null,
      error: null,
    };
  }

  if (event.type === 'contact') {
    if (
      state.completedIds.includes(event.intentionId)
    ) {
      return state;
    }

    if (
      state.pending === null
      || state.pending.id !== event.intentionId
    ) {
      return fail(
        state,
        'Physical contact does not match the active intention.',
      );
    }

    if (
      state.pending.requestedKey !== null
      && event.requestedKey
        !== state.pending.requestedKey
    ) {
      return fail(
        state,
        'Physical contact reports the wrong requested key.',
      );
    }

    return {
      ...state,
      lastContact: event,
    };
  }

  if (event.type === 'action_result') {
    if (
      state.completedIds.includes(event.intentionId)
    ) {
      return state;
    }

    if (state.pending === null) {
      return fail(
        state,
        'Simulation result arrived without an active intention.',
      );
    }

    if (state.pending.id !== event.intentionId) {
      return fail(
        state,
        'Simulation result does not match the active intention.',
      );
    }

    const appliedAction: Action =
      event.result === 'confirmed'
        ? state.pending.requestedAction
        : 'wait';

    const result = stepGame(
      state.game,
      appliedAction,
    );

    return {
      ...state,
      game: result.state,
      events: result.events,
      reward: result.reward,

      phase: result.state.terminal === null
        ? 'ready'
        : 'paused',

      pending: null,

      completedIds: rememberCompletion(
        state.completedIds,
        event.intentionId,
      ),

      lastAppliedAction: appliedAction,
      error: null,
    };
  }

  if (event.type === 'reset_complete') {
    return {
      ...createStation(state.game.seed),
      completedIds: [],
    };
  }

  return state;
}