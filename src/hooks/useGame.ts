import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { validateControllerDecision } from '../game/controllers.ts';
import type { Controller, ControllerDecision } from '../game/controllers.ts';
import { observe } from '../game/observation.ts';
import { createGame, DECISION_SECONDS, stepGame } from '../game/simulation.ts';
import { applyConnectomeSafetyReflex } from '../game/safetyReflex.ts';
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
  autonomousSpeed?: AutonomousSpeed;
};

export type AutonomousSpeed = 0.5 | 1 | 2 | 4;

export function autonomousDecisionDelayMs(speed: number): number {
  if (speed !== 0.5 && speed !== 1 && speed !== 2 && speed !== 4) {
    throw Error('Autonomous speed must be 0.5, 1, 2, or 4.');
  }
  return DECISION_SECONDS * 1_000 / speed;
}

export function normalizeSeed(value: string): string {
  const normalized = value.trim();
  if (!normalized) throw Error('Seed must not be empty.');
  if (normalized.length > 64) throw Error('Seed must be 64 characters or fewer.');
  if ([...normalized].some((character) => /\p{C}/u.test(character))) {
    throw Error('Seed must contain printable characters only.');
  }
  return normalized;
}

export function nextAutoplaySeed(currentSeed: string, episodeToken: string): string {
  if (!/^[a-f0-9]{8}$/.test(episodeToken)) {
    throw Error('Autoplay episode token must contain eight lowercase hexadecimal characters.');
  }
  const root = normalizeSeed(currentSeed).replace(/:auto:[a-f0-9]{8}$/, '');
  const suffix = `:auto:${episodeToken}`;
  return `${root.slice(0, 64 - suffix.length)}${suffix}`;
}

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
    'a[href], button, input, select, textarea, summary, [contenteditable]:not([contenteditable="false"])',
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

export function useGame({
  seed,
  mode = 'human',
  controller = null,
  visibleIds,
  autonomousSpeed = 1,
}: UseGameOptions) {
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
  const queuedHumanAction = useRef<Action | null>(null);
  const connectomeNoProgressSteps = useRef(0);
  const [documentVisible, setDocumentVisible] = useState(() => !document.hidden);

  const abortPending = useCallback(() => {
    pending.current?.abort.abort();
    pending.current = null;
  }, []);

  useEffect(() => {
    if (previousSeed.current === seed) return;
    previousSeed.current = seed;
    queuedHumanAction.current = null;
    connectomeNoProgressSteps.current = 0;
    abortPending();
    controller?.reset(seed);
    dispatch({ type: 'reset', seed, status: mode === 'human' ? 'manual' : 'ready' });
  }, [abortPending, controller, mode, seed]);

  useEffect(() => {
    queuedHumanAction.current = null;
    connectomeNoProgressSteps.current = 0;
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
        queuedHumanAction.current = null;
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

        const adjusted = controller.kind === 'connectome'
          ? applyConnectomeSafetyReflex(
            run.game,
            decision,
            connectomeNoProgressSteps.current,
          )
          : { decision, nextNoProgressSteps: 0 };

        if (controller.kind === 'connectome') {
          connectomeNoProgressSteps.current = adjusted.nextNoProgressSteps;
        } else {
          connectomeNoProgressSteps.current = 0;
        }

        pending.current = null;
        dispatch({ type: 'controller-decision', requestId, decision: adjusted.decision });
      }).catch((error: unknown) => {
        if (pending.current?.requestId !== requestId || abort.signal.aborted) return;
        pending.current = null;
        dispatch({
          type: 'controller-failure',
          requestId,
          error: error instanceof Error ? error.message : String(error),
        });
      });
    }, autonomousDecisionDelayMs(autonomousSpeed));
    return () => window.clearTimeout(timer);
  }, [autonomousSpeed, controller, documentVisible, mode, run.game, run.paused, visibleIds]);

  const onAction = useCallback((action: Action) => {
    if (mode !== 'human' || run.paused || run.game.terminal !== null) return;
    queuedHumanAction.current = action;
  }, [mode, run.game.terminal, run.paused]);

  const onTogglePause = useCallback(() => {
    queuedHumanAction.current = null;
    abortPending();
    dispatch({ type: 'toggle-pause' });
  }, [abortPending]);

  const reset = useCallback((nextSeed?: string) => {
    queuedHumanAction.current = null;
    connectomeNoProgressSteps.current = 0;
    abortPending();
    controller?.reset(nextSeed ?? run.game.seed);
    dispatch({
      type: 'reset',
      seed: nextSeed,
      status: mode === 'human' ? 'manual' : 'ready',
    });
  }, [abortPending, controller, mode, run.game.seed]);

  useEffect(() => {
    if (mode !== 'human' || !documentVisible
      || run.paused || run.game.terminal !== null) {
      return;
    }

    const timer = window.setInterval(() => {
      const action = queuedHumanAction.current ?? 'wait';
      queuedHumanAction.current = null;
      dispatch({ type: 'action', action });
    }, DECISION_SECONDS * 1_000);

    return () => window.clearInterval(timer);
  }, [documentVisible, mode, run.game.terminal, run.paused]);

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
