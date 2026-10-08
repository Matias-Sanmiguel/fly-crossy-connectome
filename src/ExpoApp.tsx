import { useEffect, useMemo, useState } from 'react';

import { BrainScene } from './components/BrainScene.tsx';
import { ExpoKeyboardScene } from './components/ExpoKeyboardScene.tsx';
import { GameScene } from './components/GameScene.tsx';
import type { Controller } from './game/controllers.ts';
import { createNeuralReplayController, decodeNeuralResponse, parseExpoReplay, parseNeuralReplay, type ExpoReplay, type NeuralReplay } from './expo/neuralReplay.ts';
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

const REPLAY_PATH = 'assets/expo/full-malecns-replay.json';

export function ExpoApp() {
  const { atlas, error: atlasError } = useAtlas();
  const [neuralReplay, setNeuralReplay] = useState<NeuralReplay | null>(null);
  const [replay, setReplay] = useState<ExpoReplay | null>(null);
  const [loadError, setLoadError] = useState('');
  const [seed, setSeed] = useState('crossy-v4-expo:0006');
  const [manuallyPaused, setManuallyPaused] = useState(false);
  const [record, setRecord] = useState(() => {
    try {
      const saved = Number(window.localStorage.getItem('fly-crossy-expo-record'));
      return Number.isSafeInteger(saved) && saved >= 0 ? saved : 0;
    } catch {
      return 0;
    }
  });

  const controller = useMemo<Controller | null>(() => {
    if (!neuralReplay || !replay) return null;
    return createNeuralReplayController(replay, neuralReplay);
  }, [neuralReplay, replay]);

  const game = useGame({
    seed,
    // Do not advance empty human ticks while the larger neural replay is loading.
    mode: 'scripted',
    controller,
    visibleIds: atlas?.visibleIds,
    autonomousSpeed: 1,
  });

  useEffect(() => {
    if (!atlas) return;
    const abort = new AbortController();
    void fetch(asset(REPLAY_PATH), { signal: abort.signal, cache: 'no-store' })
      .then((response) => {
        if (!response.ok) {
          throw Error('No se pudo cargar assets/expo/full-malecns-replay.json.');
        }
        return response.json() as Promise<unknown>;
      })
      .then(async (value) => {
        if (abort.signal.aborted) return;
        const parsed = parseExpoReplay(value);
        const responses = await Promise.all(['full-malecns-activity.json', 'full-malecns-activity.bin.gz'].map(name =>
          fetch(asset(`assets/expo/${name}`), { signal: abort.signal, cache: 'no-store' })));
        if (responses.some(response => !response.ok)) throw Error('No se pudo cargar la actividad Full MaleCNS.');
        const manifest: unknown = await responses[0]!.json();
        const data = await decodeNeuralResponse(responses[1]!);
        const neural = await parseNeuralReplay(manifest, data, parsed, atlas.visibleIds);
        if (abort.signal.aborted) return;
        setNeuralReplay(neural);
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
  }, [atlas]);

  useEffect(() => {
    if (!replay || !controller || manuallyPaused) return;
    const finished = game.state.terminal !== null || game.state.step >= replay.actions.length;
    if (!finished) return;

    if (!game.paused) game.onTogglePause();
    const timer = window.setTimeout(() => {
      game.reset(replay.seed);
    }, 1_000);
    return () => window.clearTimeout(timer);
  }, [
    controller,
    game.onTogglePause,
    game.paused,
    game.reset,
    game.state.step,
    game.state.terminal,
    replay,
    manuallyPaused,
  ]);

  const bestScore = Math.max(record, game.state.score);
  useEffect(() => {
    setRecord(bestScore);
    try {
      window.localStorage.setItem('fly-crossy-expo-record', String(bestScore));
    } catch {
      // Keep the session record when browser storage is unavailable.
    }
  }, [bestScore]);

  const togglePause = () => {
    setManuallyPaused(!game.paused);
    game.onTogglePause();
  };
  const restart = () => {
    setManuallyPaused(false);
    game.reset(replay?.seed ?? seed);
  };

  const activity = activityPresentation(controller, game.activity);
  const action = game.decision?.action ?? 'wait';
  const error = loadError || atlasError || game.controllerError;

  return (
    <main className="expo-page">
      <section className="expo-frame" aria-label="Fly Crossy Connectome modo expo">
        <header className="expo-header">
          <h1 className="expo-title">fly<span>-</span>crossy</h1>
          <p className="expo-subtitle">Un cerebro biologico intentando cruzar un mundo simulado.</p>
          <div className="expo-logo-crop">
            <img src={asset('assets/expo/imas-logo-figma.svg')} alt="imas+ tech club" />
          </div>
        </header>

        <div className="expo-dashboard">
          <div className="expo-left-column">
            <section className="expo-game-card" aria-label="Videojuego autónomo">
              <GameScene state={game.state} events={game.events} />
              <strong className="expo-score">
                <span className="sr-only">Puntaje </span>{game.state.score}
              </strong>
              <div className="expo-record">
                <span className="expo-record-label">Record:</span>
                <span className="expo-record-value">{bestScore}</span>
              </div>
            </section>

            <section className="expo-explanation">
              <h2>DE UN MAPA BIOLÓGICO A UN CEREBRO EN ACCIÓN</h2>
              <p>
                En 2026 se publicó el connectoma completo del sistema nervioso central de una mosca macho,
                reconstruyendo más de 166.000 neuronas y sus conexiones.
                <br aria-hidden="true" />
                Nosotros lo llevamos a un entorno simulado para explorar si esa estructura puede percibir
                un mundo, tomar decisiones y transformarlas en movimiento.
              </p>
            </section>
          </div>

          <div className="expo-right-column">
            <section className="expo-brain-card" aria-label="Actividad del cerebro digital">
              <div className="expo-brain-legend" aria-hidden="true">
                <span className="expo-legend-anatomy">
                  <i />
                  Anatomía medida
                </span>
                <span className="expo-legend-activity">
                  <i />
                  Actividad neuronal en tiempo real
                </span>
              </div>
              <div className="expo-brain-visual">
                {atlas
                  ? <BrainScene atlas={atlas} frame={activity.frame} activityMode={activity.kind} activityContrast="expo" />
                  : <span className="expo-media-status" role="status">Cargando cerebro</span>}
              </div>
              <dl className="expo-brain-stats">
                <div><dt>165,122</dt><dd>NEURONAS</dd></div>
                <div><dt>25,563,197</dt><dd>CONEXIONES</dd></div>
                <div><dt>708</dt><dd>NEURONAS MOTORAS</dd></div>
              </dl>
            </section>

            <div className="expo-media-row">
              <figure className="expo-video-card" role="img" aria-label="Mosca sobre un teclado">
                <video aria-hidden="true" autoPlay muted loop playsInline src={asset('assets/expo/fly-typing.mp4')} />
              </figure>
              <section className="expo-keyboard-card" aria-label="Teclado controlado por la acción actual">
                <ExpoKeyboardScene action={action} />
              </section>
            </div>

            <div className="expo-action-area">
              <section className="expo-action-card" aria-live="polite">
                <div className="expo-action-metrics">
                  <div className="expo-action-current">
                    <i aria-hidden="true" />
                    <span>Acción actual</span>
                    {action === 'wait'
                      ? <span className="expo-action-dash" aria-hidden="true">-</span>
                      : <img
                          className={`expo-action-arrow expo-action-arrow-${action}`}
                          src={asset('assets/expo/action-arrow.svg')}
                          alt=""
                        />}
                    <strong>{ACTION_LABELS[action]}</strong>
                  </div>
                  <div className="expo-step-current">
                    <i aria-hidden="true" />
                    <span>Paso</span>
                    <strong>{game.state.step}</strong>
                  </div>
                </div>
                <div className="expo-config-card">
                  <button type="button" onClick={togglePause} aria-pressed={game.paused}>
                    <span>{game.paused ? 'Continuar' : 'Pausa'}</span>
                  </button>
                  <button type="button" onClick={restart}><span>Reiniciar</span></button>
                </div>
              </section>
              <p className="expo-credits">
                <strong>Desarrollado por </strong>
                Nicolás Luca Giordano y Matías Adrián Sanmiguel.
              </p>
            </div>
          </div>
        </div>

        {error && <p className="expo-error" role="alert">{error}</p>}
      </section>
    </main>
  );
}
