import { useEffect, useRef } from 'react';
import {
  createGameRenderer,
  type GameAssetStatus,
  type GameRenderer,
} from '../game/createGameRenderer';
import type { GameEvent, GameState } from '../game/simulation';

type GameSceneProps = {
  state: GameState;
  events: readonly GameEvent[];
  onAssetStatus?: (status: GameAssetStatus) => void;
};

export function GameScene({ state, events, onAssetStatus }: GameSceneProps) {
  const host = useRef<HTMLDivElement>(null);
  const renderer = useRef<GameRenderer | null>(null);
  const latest = useRef({ state, events });
  const latestStatusHandler = useRef(onAssetStatus);
  latest.current = { state, events };
  latestStatusHandler.current = onAssetStatus;

  useEffect(() => {
    const element = host.current;
    if (!element) return;

    const gameRenderer = createGameRenderer(element, {
      onAssetStatus: (status) => latestStatusHandler.current?.(status),
    });
    renderer.current = gameRenderer;
    gameRenderer.render(latest.current.state, latest.current.events);

    return () => {
      renderer.current = null;
      gameRenderer.dispose();
    };
  }, []);

  useEffect(() => {
    renderer.current?.render(state, events);
  }, [state, events]);

  return <div ref={host} className="three-viewport game-viewport" aria-label="Isometric fly crossing game" />;
}
