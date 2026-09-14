import { useCallback, useEffect, useReducer, useRef } from 'react';
import { createGame, stepGame } from '../game/simulation.ts';
import type { GameEvent, GameState, StepResult } from '../game/simulation.ts';
import type { Action } from '../game/types.ts';

export type ControllerMode = 'human' | 'conventional' | 'connectome';

export type GameCommand =
  | { type: 'action'; action: Action }
  | { type: 'reset'; seed?: string }
  | { type: 'toggle-pause' };

export type GameRun = {
  game: GameState;
  events?: GameEvent[];
  reward?: number;
  paused?: boolean;
};

export type UseGameOptions = {
  seed: string;
  mode?: ControllerMode;
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

export function reduceGameCommand(state: GameRun, command: GameCommand): Required<GameRun> {
  const current: Required<GameRun> = {
    game: state.game,
    events: state.events ?? [],
    reward: state.reward ?? 0,
    paused: state.paused ?? false,
  };

  if (command.type === 'reset') {
    return {
      game: createGame(command.seed ?? current.game.seed),
      events: [],
      reward: 0,
      paused: false,
    };
  }

  if (command.type === 'toggle-pause') {
    return { ...current, paused: !current.paused };
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
  };
}

export function useGame({ seed, mode = 'human' }: UseGameOptions) {
  const [run, dispatch] = useReducer(reduceGameCommand, seed, (initialSeed): Required<GameRun> => ({
    game: createGame(initialSeed),
    events: [],
    reward: 0,
    paused: false,
  }));
  const previousSeed = useRef(seed);

  useEffect(() => {
    if (previousSeed.current === seed) return;
    previousSeed.current = seed;
    dispatch({ type: 'reset', seed });
  }, [seed]);

  const onAction = useCallback((action: Action) => {
    dispatch({ type: 'action', action });
  }, []);

  const onTogglePause = useCallback(() => {
    dispatch({ type: 'toggle-pause' });
  }, []);

  const reset = useCallback((nextSeed?: string) => {
    dispatch({ type: 'reset', seed: nextSeed });
  }, []);

  useEffect(() => {
    if (mode !== 'human') return;

    const onKeyDown = (event: KeyboardEvent) => {
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
    onAction,
    onTogglePause,
    reset,
  };
}
