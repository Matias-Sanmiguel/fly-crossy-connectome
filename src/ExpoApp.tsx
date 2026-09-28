import { useEffect, useMemo, useState } from 'react';

import { BrainScene } from './components/BrainScene.tsx';
import { ExpoKeyboardScene } from './components/ExpoKeyboardScene.tsx';
import { GameScene } from './components/GameScene.tsx';
import {
  BUNDLED_CONNECTOME_POLICY_PATH,
  createFixedGraphPolicyController,
  parseBundledConnectomePolicy,
  type Controller,
} from './game/controllers.ts';
import type { ExportedPolicyV1 } from './game/model.ts';
import { activityPresentation } from './game/provenance.ts';
import type { Action } from './game/types.ts';
import { useAtlas } from './hooks/useAtlas.ts';
import { nextAutoplaySeed, useGame } from './hooks/useGame.ts';
import { asset } from './lib/atlas.ts';

const ACTION_LABELS: Readonly<Record<Action, string>> = {
  forward: 'FORWARD',
  backward: 'BACKWARD',
  left: 'LEFT',
  right: 'RIGHT',
  wait: 'WAIT',
};

export function ExpoApp() {
  const { atlas, error: atlasError } = useAtlas();
  const [policy, setPolicy] = useState<ExportedPolicyV1 | null>(null);
  const [loadError, setLoadError] = useState('');
  const [seed, setSeed] = useState('expo-001');

  const controller = useMemo<Controller | null>(() => (
    policy?.network.kind === 'fixed-graph'
      ? createFixedGraphPolicyController(policy)
      : null
  ), [policy]);

  const game = useGame({
    seed,
    mode: controller?.kind ?? 'human',
    controller,
    visibleIds: atlas?.visibleIds,
    autonomousSpeed: 1,
  });

  useEffect(() => {
    if (!atlas) return;
    const abort = new AbortController();
    void fetch(asset(BUNDLED_CONNECTOME_POLICY_PATH), { signal: abort.signal })
      .then((response) => {
        if (!response.ok) throw Error('No se pudo cargar el controlador de la mosca.');
        return response.json() as Promise<unknown>;
      })
      .then((value) => {
        if (abort.signal.aborted) return;
        setPolicy(parseBundledConnectomePolicy(value, atlas.visibleIds));
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
    if (!controller || game.state.terminal === null) return;
    const timer = window.setTimeout(() => {
      const token = globalThis.crypto.randomUUID().replaceAll('-', '').slice(0, 8);
      setSeed(nextAutoplaySeed(seed, token));
    }, 1_000);
    return () => window.clearTimeout(timer);
  }, [controller, game.state.terminal, seed]);

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
                <span className="expo-record-value">68</span>
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
                  <i><img src={asset('assets/expo/legend-anatomy.svg')} alt="" /></i>
                  Anatomía medida
                </span>
                <span className="expo-legend-activity">
                  <i><img src={asset('assets/expo/legend-activity.svg')} alt="" /></i>
                  Actividad neuronal en tiempo real
                </span>
              </div>
              <div className="expo-brain-visual">
                {atlas
                  ? <BrainScene atlas={atlas} frame={activity.frame} activityMode={activity.kind} orbitSpeed={.06} />
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

            <section className="expo-action-card" aria-live="polite">
              <div className="expo-action-metrics">
                <div className="expo-action-current">
                  <i aria-hidden="true" />
                  <span>Acción actual</span>
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
                  <strong>{game.state.step}</strong>
                </div>
                <div className="expo-motor-status">
                  <i aria-hidden="true" />
                  <span>Actividad motora</span>
                </div>
              </div>
              <div className="expo-config-card">
                <p>Aca iria la configuracion<br />de la mosca<br />(seed, policiy, etc.)</p>
              </div>
            </section>
          </div>
        </div>

        {error && <p className="expo-error" role="alert">{error}</p>}
      </section>
    </main>
  );
}
