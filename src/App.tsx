import { useEffect, useMemo, useRef, useState } from 'react';
import { Attribution } from './components/Attribution';
import { BrainScene } from './components/BrainScene';
import { FlyScene } from './components/FlyScene';
import { GameControls } from './components/GameControls';
import { GameScene } from './components/GameScene';
import { Telemetry } from './components/Telemetry';
import { createDensePolicyController, createScriptedController } from './game/controllers';
import type { Controller } from './game/controllers';
import { OBSERVATION_INPUT_SIZE, parsePolicy } from './game/model';
import type { ExportedPolicyV1 } from './game/model';
import { useGame } from './hooks/useGame';
import { asset, loadAtlas } from './lib/atlas';
import type { Atlas } from './lib/atlas';

type UiMode = 'human' | 'scripted' | 'model';
const MAX_POLICY_BYTES = 10 * 1024 * 1024;

export function App() {
  const [atlas, setAtlas] = useState<Atlas | null>(null);
  const [loadError, setLoadError] = useState('');
  const [mode, setMode] = useState<UiMode>('human');
  const [policy, setPolicy] = useState<ExportedPolicyV1 | null>(null);
  const policyFile = useRef<HTMLInputElement>(null);
  const scripted = useMemo(
    () => createScriptedController(['forward', 'forward', 'wait', 'left', 'right']),
    [],
  );
  const modelController = useMemo<Controller | null>(() => (
    policy ? createDensePolicyController(policy) : null
  ), [policy]);
  const controller = mode === 'scripted' ? scripted : mode === 'model' ? modelController : null;
  const game = useGame({
    seed: 'experiment-001',
    mode: mode === 'human' ? 'human' : controller?.kind ?? 'conventional',
    controller,
    visibleIds: atlas?.visibleIds,
  });

  useEffect(() => {
    const abort = new AbortController();
    void loadAtlas(abort.signal).then(setAtlas).catch((error: unknown) => {
      if (!abort.signal.aborted) setLoadError(String(error));
    });
    return () => abort.abort();
  }, []);

  const acceptPolicy = (value: unknown) => {
    if (!atlas) throw Error('Wait for the measured MaleCNS atlas to load.');
    const nextPolicy = parsePolicy(value, atlas.visibleIds);
    if (nextPolicy.network.kind !== 'dense') {
      throw Error('This browser runtime currently accepts dense policies; fixed-graph execution is supplied by the connectome runtime.');
    }
    if (nextPolicy.network.inputSize !== OBSERVATION_INPUT_SIZE) {
      throw Error('Policy input shape does not match ObservationV1.');
    }
    setPolicy(nextPolicy);
    setMode('model');
    setLoadError('');
  };

  const manual = mode === 'human';
  const frame = manual ? null : game.activity;

  return (
    <>
      <header>
        <h1>YOUR EXPERIMENT</h1>
        <span>Environment / measured anatomy / controller output</span>
        <a href="https://github.com/cobanov/fly-connectome-template#readme">Template guide ↗</a>
      </header>
      <main>
        <div className="toolbar">
          <span className="status">
            {game.state.terminal ? `Terminal: ${game.state.terminal}` : game.paused ? 'Paused' : game.controllerStatus}
            {' · '}step {game.state.step}
          </span>
          <div className="controls">
            <label>
              Controller{' '}
              <select value={mode} onChange={(event) => setMode(event.target.value as UiMode)}>
                <option value="human">Human / manual</option>
                <option value="scripted">Scripted control</option>
                <option value="model" disabled={!policy}>Loaded dense policy</option>
              </select>
            </label>
            <button type="button" onClick={() => game.reset()}>Reset seed</button>
            <button type="button" disabled={!atlas} onClick={() => policyFile.current?.click()}>
              Load policy JSON
            </button>
            <input
              ref={policyFile}
              hidden
              type="file"
              accept=".json,application/json"
              onChange={async (event) => {
                const selected = event.target.files?.[0];
                event.target.value = '';
                if (!selected) return;
                try {
                  if (selected.size > MAX_POLICY_BYTES) throw Error('Policy JSON must be 10 MB or smaller.');
                  acceptPolicy(JSON.parse(await selected.text()));
                } catch (error) {
                  setLoadError(error instanceof Error ? error.message : String(error));
                }
              }}
            />
          </div>
        </div>
        {loadError && <p className="error" role="alert">{loadError}</p>}
        <div className="workbench">
          <section className="panel environment-panel">
            <h2>01 / CROSSING ENVIRONMENT <span>{manual ? 'Human' : controller?.kind ?? 'No controller'}</span></h2>
            <GameScene state={game.state} events={game.events} />
            <GameControls
              onAction={game.onAction}
              paused={game.paused}
              onTogglePause={game.onTogglePause}
            />
            <div className="panel-bottom game-status-bar">
              <span>
                {game.state.terminal
                  ? `Terminal: ${game.state.terminal}`
                  : manual ? 'Manual control · no neural output' : `${game.controllerStatus} · ${controller?.id}`}
              </span>
              <span>Score {game.state.score}</span>
              <button type="button" onClick={() => game.reset()}>Repeat seed</button>
            </div>
          </section>
          <section className="panel brain-panel">
            <h2>02 / BRAIN SOMA ATLAS <span>MaleCNS v1.0</span></h2>
            {atlas
              ? <BrainScene atlas={atlas} frame={frame} activityMode={manual ? 'manual' : 'simulated'} />
              : <p className="loading" role="status">Loading measured anatomy…</p>}
            <div className="panel-bottom">
              {atlas?.visibleIds.size.toLocaleString('en-US') ?? '…'} measured somata
              {' '}<a href={asset('data/brain-atlas/NOTICE.md')}>Data notice ↗</a>
            </div>
          </section>
          <section className="panel fly-panel">
            <h2>03 / BODY <span>Flybody</span></h2>
            <FlyScene />
            <div className="panel-bottom">Anatomical mesh · no motor simulation <span>Drag to rotate</span></div>
          </section>
        </div>
        <Telemetry
          manual={manual}
          controller={controller}
          state={game.state}
          decision={game.decision}
          reward={game.reward}
          status={game.controllerStatus}
          error={game.controllerError}
        />
        <details>
          <summary>Scientific scope &amp; customization</summary>
          <p>The atlas contains curated cell-body positions, not neurite morphology or synaptic edges. Points keep native proportions. Missing soma locations are never generated. The brain filter selects optic, central and descending classes; it is not a complete brain segmentation.</p>
          <p>Policies receive the same bounded ObservationV1 input and return one action per authoritative simulation step. Activity is accepted only for atlas-visible MaleCNS body IDs. Manual mode always clears neural output.</p>
          <p>Dataset creators: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology and Google Research. <a href="https://male-cns.janelia.org/download/">MaleCNS data and publication</a>, CC BY 4.0. <a href={asset('data/brain-atlas/manifest.json')}>Exact source, filters and hashes</a>.</p>
          <p>Template code has a custom attribution-required license. Keep the linked template/author credit in your web UI and repository README. Third-party assets retain their own licenses.</p>
        </details>
      </main>
      <Attribution />
    </>
  );
}
