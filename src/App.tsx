import { useEffect, useMemo, useRef, useState } from 'react';

import { Attribution } from './components/Attribution';
import { BrainScene } from './components/BrainScene';
import { FlyScene } from './components/FlyScene';
import { GameControls } from './components/GameControls';
import { GameScene } from './components/GameScene';
import { Telemetry } from './components/Telemetry';
import { ExperimentControls } from './components/lab/ExperimentControls';
import { LabHeader } from './components/lab/LabHeader';
import { LabShell } from './components/lab/LabShell';
import { PanelFrame } from './components/lab/PanelFrame';
import {
  BUNDLED_CONNECTOME_POLICY_PATH,
  createDensePolicyController,
  createFixedGraphPolicyController,
  createScriptedController,
  parseBundledConnectomePolicy,
  type Controller,
} from './game/controllers';
import type { GameAssetStatus } from './game/createGameRenderer';
import { OBSERVATION_INPUT_SIZE, parsePolicy, type ExportedPolicyV1 } from './game/model';
import { OBSERVATION_VERSION } from './game/observation';
import { activityPresentation } from './game/provenance';
import { useAtlas } from './hooks/useAtlas';
import {
  nextAutoplaySeed,
  normalizeSeed,
  useGame,
  type AutonomousSpeed,
} from './hooks/useGame';
import { asset } from './lib/atlas';

type UiMode = 'human' | 'scripted' | 'model';
const MAX_POLICY_BYTES = 50 * 1024 * 1024;

export function App() {
  const { atlas, error: atlasError } = useAtlas();
  const [loadError, setLoadError] = useState('');
  const [mode, setMode] = useState<UiMode>('human');
  const [activeSeed, setActiveSeed] = useState('experiment-001');
  const [seedInput, setSeedInput] = useState('experiment-001');
  const [seedError, setSeedError] = useState('');
  const [autonomousSpeed, setAutonomousSpeed] = useState<AutonomousSpeed>(1);
  const [policy, setPolicy] = useState<ExportedPolicyV1 | null>(null);
  const [assetStatus, setAssetStatus] = useState<GameAssetStatus>('loading');
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
    if (!atlas || !BUNDLED_CONNECTOME_POLICY_PATH) return;
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
      throw Error('Policy input shape does not match ObservationV3.');
    }
    if (nextPolicy.observationVersion !== OBSERVATION_VERSION) {
      throw Error('Policy observation version does not match the current environment.');
    }
    setPolicy(nextPolicy);
    setMode('model');
    setLoadError('');
  };

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

  const manual = mode === 'human';
  const activity = activityPresentation(controller, game.activity);
  const trainWarning = game.events.some((event) => event.type === 'train-warning');
  const runtimeStatus = game.state.terminal
    ? `terminal · ${game.state.terminal}`
    : game.paused
      ? 'paused'
      : `${game.controllerStatus} · step ${game.state.step}`;
  const sceneStatus = assetStatus === 'ready'
    ? 'Kenney scene ready'
    : assetStatus === 'fallback'
      ? 'Using geometric fallback'
      : 'Loading Kenney scene';

  return (
    <LabShell
      header={(
        <LabHeader
          eyebrow="Behavior laboratory · experiment 001"
          title="Fly Crossy Connectome"
          context="Autonomous crossing · measured anatomy · simulated controller output"
          status={runtimeStatus}
        />
      )}
      controls={(
        <ExperimentControls
          status={`${runtimeStatus} · ${sceneStatus}`}
          controller={mode}
          controllerOptions={[
            { value: 'human', label: 'Human / manual' },
            { value: 'scripted', label: 'Scripted control' },
            {
              value: 'model',
              label: `Loaded ${policy?.network.kind === 'fixed-graph' ? 'connectome' : 'dense'} policy`,
              disabled: !policy,
            },
          ]}
          onControllerChange={(value) => setMode(value as UiMode)}
          seed={seedInput}
          seedError={seedError}
          onSeedChange={setSeedInput}
          onApplySeed={applySeed}
          onNewSeed={newSeed}
          onReset={resetCurrent}
          speed={autonomousSpeed}
          speedDisabled={manual}
          onSpeedChange={(value) => setAutonomousSpeed(value as AutonomousSpeed)}
          action={(
            <>
              <button
                className="primary-action"
                type="button"
                disabled={!atlas}
                onClick={() => policyFile.current?.click()}
              >
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
                    if (selected.size > MAX_POLICY_BYTES) {
                      throw Error('Policy JSON must be 50 MB or smaller.');
                    }
                    acceptPolicy(JSON.parse(await selected.text()));
                  } catch (error) {
                    setLoadError(error instanceof Error ? error.message : String(error));
                  }
                }}
              />
            </>
          )}
        />
      )}
      alerts={(
        <>
          {seedError && <p id="seed-error" className="error" role="alert">{seedError}</p>}
          {(loadError || atlasError) && (
            <p className="error" role="alert">{loadError || atlasError}</p>
          )}
        </>
      )}
      environment={(
        <PanelFrame
          title="01 / CROSSING ENVIRONMENT"
          badge={manual ? 'Human' : controller?.kind ?? 'No controller'}
          className="environment-panel"
          footer={(
            <div className="game-status-bar">
              <span>{game.state.terminal ? `Terminal: ${game.state.terminal}` : sceneStatus}</span>
              <span>Score {game.state.score}</span>
              <button type="button" onClick={resetCurrent}>Repeat current</button>
            </div>
          )}
        >
          <GameScene
            state={game.state}
            events={game.events}
            onAssetStatus={setAssetStatus}
          />
          {trainWarning && (
            <p className="game-warning" role="alert">Train approaching — move off the rail.</p>
          )}
          <GameControls
            onAction={game.onAction}
            paused={game.paused}
            terminal={game.state.terminal !== null}
            onTogglePause={game.onTogglePause}
            manual={manual}
          />
        </PanelFrame>
      )}
      brain={(
        <PanelFrame
          title="02 / BRAIN SOMA ATLAS"
          badge="MaleCNS v1.0"
          className="brain-panel"
          footer={(
            <>
              <span>{atlas?.visibleIds.size.toLocaleString('en-US') ?? '…'} measured somata</span>
              <a href={asset('data/brain-atlas/NOTICE.md')}>Data notice ↗</a>
            </>
          )}
        >
          {atlas
            ? <BrainScene atlas={atlas} frame={activity.frame} activityMode={activity.kind} />
            : <p className="loading" role="status">Loading measured anatomy…</p>}
        </PanelFrame>
      )}
      telemetry={(
        <Telemetry
          controller={controller}
          activity={activity}
          state={game.state}
          decision={game.decision}
          reward={game.reward}
          status={game.controllerStatus}
          error={game.controllerError}
        />
      )}
      body={(
        <PanelFrame title="03 / BODY" badge="Flybody" className="fly-panel" footer="Anatomical mesh · no motor simulation">
          <FlyScene />
        </PanelFrame>
      )}
      secondary={(
        <details>
          <summary>Scientific scope, policy upload &amp; provenance</summary>
          <p>The atlas contains measured soma positions, not neurite morphology or synaptic edges. Missing soma locations are never generated.</p>
          <p>Controller activity is simulated model output keyed by atlas-visible MaleCNS body IDs. It is not a biological recording.</p>
          <p>Kenney scenery is CC0. MaleCNS is CC BY 4.0; Flybody is Apache 2.0. Template attribution remains visible below.</p>
        </details>
      )}
      footer={<Attribution />}
    />
  );
}
