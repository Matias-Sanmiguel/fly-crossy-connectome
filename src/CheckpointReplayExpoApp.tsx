import { useEffect, useMemo, useState } from 'react';

import { BrainScene } from './components/BrainScene.tsx';
import { ExpoKeyboardScene } from './components/ExpoKeyboardScene.tsx';
import { GameScene } from './components/GameScene.tsx';
import {
  createScriptedController,
  type Controller,
} from './game/controllers.ts';
import { activityPresentation } from './game/provenance.ts';
import type { Action } from './game/types.ts';
import { useAtlas } from './hooks/useAtlas.ts';
import { useGame } from './hooks/useGame.ts';
import { asset } from './lib/atlas.ts';

const ACTION_LABELS: Readonly<Record<Action, string>> = {
  forward: 'FORWARD',
  backward: 'BACKWARD',
  left: 'LEFT',
  right: 'RIGHT',
  wait: 'WAIT',
};

const ACTIONS = new Set<Action>(['forward', 'backward', 'left', 'right', 'wait']);
const REPLAY_PATH = 'assets/expo/full-malecns-replay.json';
const FALLBACK_SEED = 'crossy-v4-expo:0006';

type ReplayMetrics = {
  steps: number;
  score: number;
  row: number;
  terminalReason: string | null;
  reachedStepLimit: boolean;
  progressRate: number;
  blockedActions: number;
  blockedRate: number;
  maxNoProgressStreak: number;
  pingPongReturns: number;
};

type CheckpointReplay = {
  version: 1;
  kind: 'full-malecns-checkpoint-action-replay';
  seed: string;
  sourceCheckpoint: string;
  checkpointStage: string | null;
  fullMaleCNS: {
    neurons: number;
    edges: number;
    sensory: number;
    motor: number;
  };
  actions: Action[];
  metrics: ReplayMetrics;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function parseReplay(value: unknown): CheckpointReplay {
  if (!isRecord(value)
    || value.version !== 1
    || value.kind !== 'full-malecns-checkpoint-action-replay'
    || typeof value.seed !== 'string'
    || !Array.isArray(value.actions)
    || !isRecord(value.metrics)
    || !isRecord(value.fullMaleCNS)) {
    throw Error('El replay del checkpoint tiene un formato invalido.');
  }
  if (value.actions.length === 0 || value.actions.some((action) => !ACTIONS.has(action as Action))) {
    throw Error('El replay no contiene una secuencia de acciones valida.');
  }
  const metrics = value.metrics;
  const fullMaleCNS = value.fullMaleCNS;
  for (const key of ['steps', 'score', 'row', 'progressRate', 'blockedActions', 'blockedRate', 'maxNoProgressStreak', 'pingPongReturns']) {
    if (typeof metrics[key] !== 'number' || !Number.isFinite(metrics[key])) {
      throw Error(`Replay invalido: metrics.${key}.`);
    }
  }
  for (const key of ['neurons', 'edges', 'sensory', 'motor']) {
    if (typeof fullMaleCNS[key] !== 'number' || !Number.isFinite(fullMaleCNS[key])) {
      throw Error(`Replay invalido: fullMaleCNS.${key}.`);
    }
  }
  return value as unknown as CheckpointReplay;
}

function checkpointName(path: string): string {
  const normalized = path.replaceAll('\\', '/');
  const pieces = normalized.split('/').filter(Boolean);
  if (pieces.length < 2) return path;
  return pieces.slice(-3).join('/');
}

export function CheckpointReplayExpoApp() {
  const { atlas, error: atlasError } = useAtlas();
  const [replay, setReplay] = useState<CheckpointReplay | null>(null);
  const [loadError, setLoadError] = useState('');
  const [seed, setSeed] = useState(FALLBACK_SEED);

  useEffect(() => {
    const abort = new AbortController();
    void fetch(asset(REPLAY_PATH), { signal: abort.signal, cache: 'no-store' })
      .then((response) => {
        if (!response.ok) {
          throw Error('No se encontro el replay. Genera full-malecns-replay.json primero.');
        }
        return response.json() as Promise<unknown>;
      })
      .then((value) => {
        if (abort.signal.aborted) return;
        const parsed = parseReplay(value);
        setReplay(parsed);
        setSeed(parsed.seed);
        setLoadError('');
      })
      .catch((error: unknown) => {
        if (!abort.signal.aborted) {
          setLoadError(error instanceof Error ? error.message : String(error));
        }
      });
    return () => abort.abort();
  }, []);

  const controller = useMemo<Controller | null>(() => (
    replay ? createScriptedController(replay.actions) : null
  ), [replay]);

  const game = useGame({
    seed,
    mode: controller?.kind ?? 'human',
    controller,
    visibleIds: atlas?.visibleIds,
    autonomousSpeed: 1,
  });

  const activity = activityPresentation(controller, game.activity);
  const action = game.decision?.action ?? 'wait';
  const error = loadError || atlasError || game.controllerError;
  const expectedScore = replay?.metrics.score ?? 0;
  const expectedSteps = replay?.metrics.steps ?? 0;
  const replayMatches = replay !== null
    && game.state.terminal !== null
    && game.state.step === replay.metrics.steps
    && game.state.score === replay.metrics.score;

  const resetReplay = () => {
    game.reset(replay?.seed ?? FALLBACK_SEED);
  };

  useEffect(() => {
    if (!replay || game.paused || game.state.terminal !== null) return;
    if (game.state.step >= replay.actions.length) game.onTogglePause();
  }, [game.onTogglePause, game.paused, game.state.step, game.state.terminal, replay]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as Element | null;
      if (typeof target?.closest === 'function'
        && target.closest('button, input, textarea, select, a[href], [contenteditable]:not([contenteditable=\"false\"])')) {
        return;
      }
      if (event.code === 'Space' || event.key.toLowerCase() === 'p') {
        event.preventDefault();
        if (!event.repeat) game.onTogglePause();
      } else if (event.key.toLowerCase() === 'r') {
        event.preventDefault();
        if (!event.repeat) resetReplay();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [game.onTogglePause, replay]);

  return (
    <main className="expo-page">
      <section className="expo-frame" aria-label="Fly Crossy Connectome replay de checkpoint">
        <header className="expo-header">
          <h1 className="expo-title">fly<span>-</span>crossy</h1>
          <p className="expo-subtitle">Replay visual del checkpoint Full MaleCNS sobre la seed exacta de expo.</p>
          <div className="expo-logo-crop">
            <img src={asset('assets/expo/imas-logo-figma.svg')} alt="imas+ tech club" />
          </div>
        </header>

        <div className="expo-dashboard">
          <div className="expo-left-column">
            <section className="expo-game-card" aria-label="Replay autonomo del checkpoint">
              <GameScene state={game.state} events={game.events} />
              <strong className="expo-score">
                <span className="sr-only">Puntaje </span>{game.state.score}
              </strong>
              <div className="expo-record">
                <span className="expo-record-label">Checkpoint:</span>
                <span className="expo-record-value">{expectedScore || '--'}</span>
              </div>
            </section>

            <section className="expo-explanation">
              <h2>REPLAY DEL MALECNS COMPLETO</h2>
              <p>
                Esta vista reproduce la secuencia cerrada de acciones generada por el checkpoint Full MaleCNS
                sobre <strong>{replay?.seed ?? FALLBACK_SEED}</strong>. El navegador conserva la autoridad del
                mundo Crossy y vuelve a ejecutar las mismas acciones para poder inspeccionar visualmente la partida.
              </p>
            </section>
          </div>

          <div className="expo-right-column">
            <section className="expo-brain-card" aria-label="Anatomia del cerebro digital">
              <div className="expo-brain-legend" aria-hidden="true">
                <span className="expo-legend-anatomy">
                  <i><img src={asset('assets/expo/legend-anatomy.svg')} alt="" /></i>
                  Anatomia medida
                </span>
                <span className="expo-legend-activity">
                  <i><img src={asset('assets/expo/legend-activity.svg')} alt="" /></i>
                  Replay de decisiones del checkpoint
                </span>
              </div>
              <div className="expo-brain-visual">
                {atlas
                  ? <BrainScene atlas={atlas} frame={activity.frame} activityMode={activity.kind} orbitSpeed={.06} />
                  : <span className="expo-media-status" role="status">Cargando cerebro</span>}
              </div>
              <dl className="expo-brain-stats">
                <div><dt>{replay?.fullMaleCNS.neurons.toLocaleString('en-US') ?? '165,122'}</dt><dd>NEURONAS</dd></div>
                <div><dt>{replay?.fullMaleCNS.edges.toLocaleString('en-US') ?? '25,563,197'}</dt><dd>CONEXIONES</dd></div>
                <div><dt>{replay?.fullMaleCNS.motor.toLocaleString('en-US') ?? '708'}</dt><dd>NEURONAS MOTORAS</dd></div>
              </dl>
            </section>

            <div className="expo-media-row">
              <figure className="expo-video-card" role="img" aria-label="Mosca sobre un teclado">
                <video aria-hidden="true" autoPlay muted loop playsInline src={asset('assets/expo/fly-typing.mp4')} />
              </figure>
              <section className="expo-keyboard-card" aria-label="Teclado controlado por la accion actual">
                <ExpoKeyboardScene action={action} />
              </section>
            </div>

            <section className="expo-action-card" aria-live="polite">
              <div className="expo-action-metrics">
                <div className="expo-action-current">
                  <i aria-hidden="true" />
                  <span>Accion actual</span>
                  <img
                    className={`expo-action-arrow expo-action-arrow-${action}`}
                    src={asset('assets/expo/action-arrow.svg')}
                    alt=""
                  />
                  <strong>{ACTION_LABELS[action]}</strong>
                </div>
                <div className="expo-step-current">
                  <i aria-hidden="true" />
                  <span>Paso</span>
                  <strong>{game.state.step}{expectedSteps ? ` / ${expectedSteps}` : ''}</strong>
                </div>
                <div className="expo-motor-status">
                  <i aria-hidden="true" />
                  <span>{game.state.terminal === null ? 'Replay en curso' : replayMatches ? 'Replay verificado' : 'Replay finalizado'}</span>
                </div>
              </div>
              <div className="expo-config-card">
                <p>
                  Full MaleCNS<br />
                  Seed: {replay?.seed ?? FALLBACK_SEED}<br />
                  Score esperado: {expectedScore || '--'}<br />
                  {replay ? checkpointName(replay.sourceCheckpoint) : 'Cargando checkpoint replay...'}
                </p>
                <div style={{ display: 'flex', gap: '.5rem', flexWrap: 'wrap', marginTop: '.65rem' }}>
                  <button type="button" onClick={game.onTogglePause}>
                    {game.paused ? 'REANUDAR' : 'PAUSAR'}
                  </button>
                  <button type="button" onClick={resetReplay}>REINICIAR</button>
                </div>
                <small style={{ display: 'block', marginTop: '.45rem', opacity: .72 }}>
                  Replay controls: SPACE pause/resume · R reset
                </small>
              </div>
            </section>
          </div>
        </div>

        {error && <p className="expo-error" role="alert">{error}</p>}
      </section>
    </main>
  );
}
