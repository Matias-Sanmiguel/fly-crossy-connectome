import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { validateControllerDecision } from '../game/controllers.ts';
import type { Controller, ControllerDecision } from '../game/controllers.ts';
import { observe } from '../game/observation.ts';
import { createGame, DECISION_SECONDS, stepGame } from '../game/simulation.ts';
import type { GameEvent, GameState, StepResult } from '../game/simulation.ts';
import type { Action } from '../game/types.ts';
import type { ActivityFrame } from '../lib/replay.ts';

export type ControllerMode = 'human' | Controller['kind'];
export type ControllerStatus = 'manual' | 'ready' | 'pending' | 'paused' | 'error' | 'terminal';

export type GameCommand =
  | { type: 'action'; action: Action }
  | { type: 'reset'; seed?: string; status?: 'manual' | 'ready' }
  | { type: 'toggle-pause' }
  | { type: 'controller-start'; requestId: number }
  | { type: 'controller-decision'; requestId: number; decision: ControllerDecision }
  | { type: 'controller-failure'; requestId: number; error: string }
  | { type: 'controller-connection-failure'; error: string }
  | { type: 'controller-cancel' }
  | { type: 'controller-clear'; status?: 'manual' | 'ready' };

export type GameRun = {
  game: GameState;
  events?: GameEvent[];
  reward?: number;
  paused?: boolean;
  activity?: ActivityFrame | null;
  decision?: ControllerDecision | null;
  pendingRequest?: number | null;
  controllerStatus?: ControllerStatus;
  controllerError?: string | null;
};

export type UseGameOptions = {
  seed: string;
  mode?: ControllerMode;
  controller?: Controller | null;
  visibleIds?: ReadonlySet<number>;
};

const KEY_ACTIONS: Readonly<Record<string, Action>> = {
  ArrowUp: 'forward',
  ArrowDown: 'backward',
  ArrowLeft: 'left',
  ArrowRight: 'right',
  w: 'forward',
  W: 'forward',
  s: 'backward',
  S: 'backward',
  a: 'left',
  A: 'left',
  d: 'right',
  D: 'right',
  ' ': 'wait',
};

export function isInteractiveKeyboardTarget(target: EventTarget | null): boolean {
  const element = target as Element | null;
  if (typeof element?.closest !== 'function') return false;
  return element.closest(
    'a[href], button, input, select, textarea, [contenteditable]:not([contenteditable="false"])',
  ) !== null;
}

export function reduceGameCommand(state: GameRun, command: GameCommand): Required<GameRun> {
  const current: Required<GameRun> = {
    game: state.game,
    events: state.events ?? [],
    reward: state.reward ?? 0,
    paused: state.paused ?? false,
    activity: state.activity ?? null,
    decision: state.decision ?? null,
    pendingRequest: state.pendingRequest ?? null,
    controllerStatus: state.controllerStatus ?? 'manual',
    controllerError: state.controllerError ?? null,
  };

  if (command.type === 'reset') {
    return {
      game: createGame(command.seed ?? current.game.seed),
      events: [],
      reward: 0,
      paused: false,
      activity: null,
      decision: null,
      pendingRequest: null,
      controllerStatus: command.status
        ?? (current.controllerStatus === 'manual' ? 'manual' : 'ready'),
      controllerError: null,
    };
  }

  if (command.type === 'toggle-pause') {
    const paused = !current.paused;
    return {
      ...current,
      paused,
      pendingRequest: paused ? null : current.pendingRequest,
      controllerStatus: current.controllerStatus === 'manual'
        ? 'manual'
        : paused ? 'paused' : 'ready',
    };
  }

  if (command.type === 'controller-clear') {
    return {
      ...current,
      events: [],
      reward: 0,
      activity: null,
      decision: null,
      pendingRequest: null,
      controllerError: null,
      controllerStatus: command.status ?? 'ready',
    };
  }

  if (command.type === 'controller-cancel') {
    return {
      ...current,
      pendingRequest: null,
      controllerStatus: current.controllerStatus === 'pending' ? 'ready' : current.controllerStatus,
    };
  }

  if (command.type === 'controller-start') {
    if (current.pendingRequest !== null || current.paused || current.game.terminal !== null) return current;
    return {
      ...current,
      pendingRequest: command.requestId,
      controllerStatus: 'pending',
      controllerError: null,
    };
  }

  if (command.type === 'controller-failure') {
    if (current.pendingRequest !== command.requestId) return current;
    return {
      ...current,
      paused: true,
      activity: null,
      pendingRequest: null,
      controllerStatus: 'error',
      controllerError: command.error,
    };
  }

  if (command.type === 'controller-connection-failure') {
    return {
      ...current,
      paused: true,
      activity: null,
      pendingRequest: null,
      controllerStatus: 'error',
      controllerError: command.error,
    };
  }

  if (command.type === 'controller-decision') {
    if (current.pendingRequest !== command.requestId || current.paused || current.game.terminal !== null) {
      return current;
    }
    const result = stepGame(current.game, command.decision.action);
    return {
      game: result.state,
      events: result.events,
      reward: result.reward,
      paused: false,
      activity: { time: result.state.time, values: command.decision.activity },
      decision: command.decision,
      pendingRequest: null,
      controllerStatus: result.state.terminal === null ? 'ready' : 'terminal',
      controllerError: null,
    };
  }

  if (current.paused || current.game.terminal !== null) {
    return { ...current, events: [], reward: 0 };
  }

  const result: StepResult = stepGame(current.game, command.action);
  return {
    game: result.state,
    events: result.events,
    reward: result.reward,
    paused: current.paused,
    activity: null,
    decision: { action: command.action, activity: [], diagnostics: {} },
    pendingRequest: null,
    controllerStatus: result.state.terminal === null ? current.controllerStatus : 'terminal',
    controllerError: null,
  };
}

export function useGame({ seed, mode = 'human', controller = null, visibleIds }: UseGameOptions) {
  const [run, dispatch] = useReducer(reduceGameCommand, seed, (initialSeed): Required<GameRun> => ({
    game: createGame(initialSeed),
    events: [],
    reward: 0,
    paused: false,
    activity: null,
    decision: null,
    pendingRequest: null,
    controllerStatus: 'manual',
    controllerError: null,
  }));
  const previousSeed = useRef(seed);
  const pending = useRef<{ requestId: number; abort: AbortController } | null>(null);
  const nextRequestId = useRef(1);
  const [documentVisible, setDocumentVisible] = useState(() => !document.hidden);

  const abortPending = useCallback(() => {
    pending.current?.abort.abort();
    pending.current = null;
  }, []);

  useEffect(() => {
    if (previousSeed.current === seed) return;
    previousSeed.current = seed;
    abortPending();
    controller?.reset(seed);
    dispatch({ type: 'reset', seed, status: mode === 'human' ? 'manual' : 'ready' });
  }, [abortPending, controller, mode, seed]);

  useEffect(() => {
    abortPending();
    controller?.reset(run.game.seed);
    dispatch({ type: 'controller-clear', status: mode === 'human' ? 'manual' : 'ready' });
  }, [abortPending, controller, mode]);

  useEffect(() => () => {
    abortPending();
    controller?.dispose();
  }, [abortPending, controller]);

  useEffect(() => {
    const onVisibilityChange = () => {
      const visible = !document.hidden;
      setDocumentVisible(visible);
      if (!visible) {
        abortPending();
        dispatch({ type: 'controller-cancel' });
      }
    };
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => document.removeEventListener('visibilitychange', onVisibilityChange);
  }, [abortPending]);

  useEffect(() => {
    if (mode === 'human' || !controller?.subscribeFailure) return;
    return controller.subscribeFailure((error) => {
      abortPending();
      dispatch({ type: 'controller-connection-failure', error: error.message });
    });
  }, [abortPending, controller, mode]);

  useEffect(() => {
    if (mode === 'human' || !controller || !documentVisible
      || run.paused || run.game.terminal !== null || pending.current) {
      return;
    }
    const timer = window.setTimeout(() => {
      if (pending.current) return;
      const requestId = nextRequestId.current;
      nextRequestId.current += 1;
      const abort = new AbortController();
      pending.current = { requestId, abort };
      dispatch({ type: 'controller-start', requestId });
      void controller.decide(observe(run.game), abort.signal).then((rawDecision) => {
        const decision = validateControllerDecision(rawDecision, visibleIds);
        if (pending.current?.requestId !== requestId || abort.signal.aborted) return;
        pending.current = null;
        dispatch({ type: 'controller-decision', requestId, decision });
      }).catch((error: unknown) => {
        if (pending.current?.requestId !== requestId || abort.signal.aborted) return;
        pending.current = null;
        dispatch({
          type: 'controller-failure',
          requestId,
          error: error instanceof Error ? error.message : String(error),
        });
      });
    }, DECISION_SECONDS * 1_000);
    return () => window.clearTimeout(timer);
  }, [controller, documentVisible, mode, run.game, run.paused, visibleIds]);

  const onAction = useCallback((action: Action) => {
    if (mode !== 'human') return;
    dispatch({ type: 'action', action });
  }, [mode]);

  const onTogglePause = useCallback(() => {
    abortPending();
    dispatch({ type: 'toggle-pause' });
  }, [abortPending]);

  const reset = useCallback((nextSeed?: string) => {
    abortPending();
    controller?.reset(nextSeed ?? run.game.seed);
    dispatch({
      type: 'reset',
      seed: nextSeed,
      status: mode === 'human' ? 'manual' : 'ready',
    });
  }, [abortPending, controller, mode, run.game.seed]);

  useEffect(() => {
    if (mode !== 'human') return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (isInteractiveKeyboardTarget(event.target)) return;
      const action = KEY_ACTIONS[event.key] ?? (event.code === 'Space' ? 'wait' : undefined);
      const togglesPause = event.key === 'Escape';
      if (!action && !togglesPause) return;

      event.preventDefault();
      if (event.repeat) return;
      if (togglesPause) onTogglePause();
      else onAction(action!);
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [mode, onAction, onTogglePause]);

  return {
    state: run.game,
    events: run.events,
    reward: run.reward,
    paused: run.paused,
    activity: run.activity,
    decision: run.decision,
    controllerStatus: run.controllerStatus,
    controllerError: run.controllerError,
    onAction,
    onTogglePause,
    reset,
  };
}
