import { useEffect, useMemo, useRef, useState } from 'react';
import { Attribution } from './components/Attribution';
import { BrainScene } from './components/BrainScene';
import { FlyScene } from './components/FlyScene';
import { GameControls } from './components/GameControls';
import { GameScene } from './components/GameScene';
import { Telemetry } from './components/Telemetry';
import {
  BUNDLED_CONNECTOME_POLICY_PATH,
  createDensePolicyController,
  createFixedGraphPolicyController,
  createScriptedController,
  parseBundledConnectomePolicy,
} from './game/controllers';
import type { Controller } from './game/controllers';
import { OBSERVATION_INPUT_SIZE, parsePolicy } from './game/model';
import type { ExportedPolicyV1 } from './game/model';
import { activityPresentation } from './game/provenance';
import { nextAutoplaySeed, normalizeSeed, useGame } from './hooks/useGame';
import type { AutonomousSpeed } from './hooks/useGame';
import { asset, loadAtlas } from './lib/atlas';
import type { Atlas } from './lib/atlas';

type UiMode = 'human' | 'scripted' | 'model';
const MAX_POLICY_BYTES = 10 * 1024 * 1024;

export function App() {
  const [atlas, setAtlas] = useState<Atlas | null>(null);
  const [loadError, setLoadError] = useState('');
  const [mode, setMode] = useState<UiMode>('human');
  const [activeSeed, setActiveSeed] = useState('experiment-001');
  const [seedInput, setSeedInput] = useState('experiment-001');
  const [seedError, setSeedError] = useState('');
  const [autonomousSpeed, setAutonomousSpeed] = useState<AutonomousSpeed>(1);
  const [policy, setPolicy] = useState<ExportedPolicyV1 | null>(null);
  const policyFile = useRef<HTMLInputElement>(null);
  const controller = useMemo<Controller | null>(() => {
    if (mode === 'scripted') {
      return createScriptedController(['forward', 'forward', 'wait', 'left', 'right']);
    }
    if (mode === 'model' && policy) {
      return policy.network.kind === 'fixed-graph'
        ? createFixedGraphPolicyController(policy)
        : createDensePolicyController(policy);
    }
    return null;
  }, [mode, policy]);
  const game = useGame({
    seed: activeSeed,
    mode: mode === 'human' ? 'human' : controller?.kind ?? 'conventional',
    controller,
    visibleIds: atlas?.visibleIds,
    autonomousSpeed,
  });

  useEffect(() => {
    const abort = new AbortController();
    void loadAtlas(abort.signal).then(setAtlas).catch((error: unknown) => {
      if (!abort.signal.aborted) setLoadError(String(error));
    });
    return () => abort.abort();
  }, []);

  useEffect(() => {
    if (!atlas) return;
    const abort = new AbortController();
    void fetch(asset(BUNDLED_CONNECTOME_POLICY_PATH), { signal: abort.signal })
      .then((response) => {
        if (!response.ok) throw Error('Bundled connectome controller could not load.');
        return response.json() as Promise<unknown>;
      })
      .then((value) => {
        if (abort.signal.aborted) return;
        setPolicy(parseBundledConnectomePolicy(value, atlas.visibleIds));
        setMode('model');
        setLoadError('');
      })
      .catch((error: unknown) => {
        if (abort.signal.aborted) return;
        setMode('human');
        setLoadError(error instanceof Error ? error.message : String(error));
      });
    return () => abort.abort();
  }, [atlas]);

  useEffect(() => {
    if (mode !== 'model' || !controller || game.state.terminal === null) return;
    const timer = window.setTimeout(() => {
      const token = globalThis.crypto.randomUUID().replaceAll('-', '').slice(0, 8);
      const nextSeed = nextAutoplaySeed(activeSeed, token);
      setSeedInput(nextSeed);
      setSeedError('');
      setActiveSeed(nextSeed);
    }, 1_000);
    return () => window.clearTimeout(timer);
  }, [activeSeed, controller, game.state.terminal, mode]);

  const acceptPolicy = (value: unknown) => {
    if (!atlas) throw Error('Wait for the measured MaleCNS atlas to load.');
    const nextPolicy = parsePolicy(value, atlas.visibleIds);
    if (nextPolicy.network.inputSize !== OBSERVATION_INPUT_SIZE) {
      throw Error('Policy input shape does not match ObservationV1.');
    }
    setPolicy(nextPolicy);
    setMode('model');
    setLoadError('');
  };

  const manual = mode === 'human';
  const activity = activityPresentation(controller, game.activity);
  const trainWarning = game.events.some((event) => event.type === 'train-warning');

  const applySeed = () => {
    try {
      const nextSeed = normalizeSeed(seedInput);
      setSeedInput(nextSeed);
      setSeedError('');
      if (nextSeed === activeSeed) game.reset(nextSeed);
      else setActiveSeed(nextSeed);
    } catch (error) {
      setSeedError(error instanceof Error ? error.message : String(error));
    }
  };

  const newSeed = () => {
    const nextSeed = `experiment-${globalThis.crypto.randomUUID().slice(0, 8)}`;
    setSeedInput(nextSeed);
    setSeedError('');
    setActiveSeed(nextSeed);
  };

  const resetCurrent = () => {
    setSeedInput(activeSeed);
    setSeedError('');
    game.reset(activeSeed);
  };

  return (
    <>
      <header className="site-header">
        <div className="brand">
          <span className="eyebrow">Behavior laboratory / experiment 001</span>
          <h1>Fly Crossy Connectome</h1>
        </div>
        <span className="header-context">Crossing environment · measured anatomy · controller output</span>
        <a className="header-link" href="https://github.com/cobanov/fly-connectome-template#readme">Template guide ↗</a>
      </header>
      <main>
        <div className="toolbar" aria-label="Experiment controls">
          <span className="status" role="status" aria-live="polite">
            {game.state.terminal ? `Terminal: ${game.state.terminal}` : game.paused ? 'Paused' : game.controllerStatus}
            {' · '}step {game.state.step}
          </span>
          <div className="controls">
            <label>
              Controller{' '}
              <select value={mode} onChange={(event) => setMode(event.target.value as UiMode)}>
                <option value="human">Human / manual</option>
                <option value="scripted">Scripted control</option>
                <option value="model" disabled={!policy}>
                  Loaded {policy?.network.kind === 'fixed-graph' ? 'connectome' : 'dense'} policy
                </option>
              </select>
            </label>
            <label>
              Seed
              <input
                className="seed-input"
                value={seedInput}
                maxLength={64}
                aria-invalid={seedError ? 'true' : 'false'}
                aria-describedby={seedError ? 'seed-error' : undefined}
                onChange={(event) => setSeedInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    applySeed();
                  }
                }}
              />
            </label>
            <button type="button" onClick={applySeed}>Apply seed</button>
            <button type="button" onClick={newSeed}>New seed</button>
            <button type="button" onClick={resetCurrent}>Reset current</button>
            <label>
              Autonomous speed
              <select
                value={autonomousSpeed}
                disabled={manual}
                onChange={(event) => setAutonomousSpeed(Number(event.target.value) as AutonomousSpeed)}
              >
                <option value={0.5}>0.5×</option>
                <option value={1}>1×</option>
                <option value={2}>2×</option>
                <option value={4}>4×</option>
              </select>
            </label>
            <button className="primary-action" type="button" disabled={!atlas} onClick={() => policyFile.current?.click()}>
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
        {seedError && <p id="seed-error" className="error" role="alert">{seedError}</p>}
        {loadError && <p className="error" role="alert">{loadError}</p>}
        <div className="workbench">
          <section className="panel environment-panel">
            <h2>01 / CROSSING ENVIRONMENT <span>{manual ? 'Human' : controller?.kind ?? 'No controller'}</span></h2>
            <GameScene state={game.state} events={game.events} />
            {trainWarning && <p className="game-warning" role="alert">Train approaching — move off the rail.</p>}
            <GameControls
              onAction={game.onAction}
              paused={game.paused}
              terminal={game.state.terminal !== null}
              onTogglePause={game.onTogglePause}
              manual={manual}
            />
            <div className="panel-bottom game-status-bar">
              <span>
                {game.state.terminal
                  ? `Terminal: ${game.state.terminal}`
                  : `${game.controllerStatus} · ${activity.heading.toLowerCase()}`}
              </span>
              <span>Score {game.state.score}</span>
              <button type="button" onClick={resetCurrent}>Repeat current</button>
            </div>
          </section>
          <section className="panel brain-panel">
            <h2>02 / BRAIN SOMA ATLAS <span>MaleCNS v1.0</span></h2>
            {atlas
              ? <BrainScene atlas={atlas} frame={activity.frame} activityMode={activity.kind} />
              : <p className="loading" role="status">Loading measured anatomy…</p>}
            <div className="panel-bottom">
              {atlas?.visibleIds.size.toLocaleString('en-US') ?? '…'} measured somata
              {' '}<a href={asset('data/brain-atlas/NOTICE.md')}>Data notice ↗</a>
            </div>
          </section>
          <Telemetry
            controller={controller}
            activity={activity}
            state={game.state}
            decision={game.decision}
            reward={game.reward}
            status={game.controllerStatus}
            error={game.controllerError}
          />
          <section className="panel fly-panel">
            <h2>03 / BODY <span>Flybody</span></h2>
            <FlyScene />
            <div className="panel-bottom">Anatomical mesh · no motor simulation <span>Drag to rotate</span></div>
          </section>
        </div>
        <details>
          <summary>Scientific scope &amp; customization</summary>
          <p>The atlas contains curated cell-body positions, not neurite morphology or synaptic edges. Points keep native proportions. Missing soma locations are never generated. The brain filter selects optic, central and descending classes; it is not a complete brain segmentation.</p>
          <p>Policies receive the same bounded ObservationV1 input and return one action per authoritative simulation step. Activity is accepted only for atlas-visible MaleCNS body IDs. Human and scripted modes have no neural output. Dense policies show model output only when they declare an explicit mapped activity layer; the fixed graph shows simulated reduced-circuit activity.</p>
          <p>Dataset creators: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology and Google Research. <a href="https://male-cns.janelia.org/download/">MaleCNS data and publication</a>, CC BY 4.0. <a href={asset('data/brain-atlas/manifest.json')}>Exact source, filters and hashes</a>.</p>
          <p>Template code has a custom attribution-required license. Keep the linked template/author credit in your web UI and repository README. Third-party assets retain their own licenses.</p>
        </details>
      </main>
      <Attribution />
    </>
  );
}
