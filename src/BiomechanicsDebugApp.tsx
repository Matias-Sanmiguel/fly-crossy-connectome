import { useMemo, useState } from 'react';

import { Attribution } from './components/Attribution';
import { BrainScene } from './components/BrainScene';
import { GameScene } from './components/GameScene';
import { ExperimentControls } from './components/lab/ExperimentControls';
import { LabHeader } from './components/lab/LabHeader';
import { LabShell } from './components/lab/LabShell';
import { PanelFrame } from './components/lab/PanelFrame';
import { TelemetryStrip } from './components/lab/TelemetryStrip';
import type { GameAssetStatus } from './game/createGameRenderer';
import { normalizeSeed } from './hooks/useGame';
import { useAtlas } from './hooks/useAtlas';
import { useSimulationStation } from './hooks/useSimulationStation';
import { asset } from './lib/atlas';
import { neuralActivityFrame } from './simulation/activity';

const SERVER_URL = import.meta.env.VITE_SIMULATION_URL
  ?? 'ws://127.0.0.1:8001/api/simulation';

export function BiomechanicsDebugApp() {
  const { atlas, error: atlasError } = useAtlas();
  const [activeSeed, setActiveSeed] = useState('biomechanics-debug');
  const [seedInput, setSeedInput] = useState('biomechanics-debug');
  const [seedError, setSeedError] = useState('');
  const [speed, setSpeed] = useState(1);
  const [assetStatus, setAssetStatus] = useState<GameAssetStatus>('loading');
  const { station, transport, activity, snapshot, error } = useSimulationStation({
    seed: activeSeed,
    url: SERVER_URL,
    population: 80,
    backendPreference: 'cpu',
    speed,
    autoResetDelayMs: 1_000,
  });

  const frame = useMemo(() => neuralActivityFrame(activity), [activity]);
  const pending = station.pending;
  const runtimeStatus = `${transport.phase} · ${station.phase} · step ${station.game.step}`;
  const sceneStatus = assetStatus === 'ready'
    ? 'Kenney scene ready'
    : assetStatus === 'fallback'
      ? 'Using geometric fallback'
      : 'Loading Kenney scene';

  const applySeed = () => {
    try {
      const next = normalizeSeed(seedInput);
      setSeedInput(next);
      setSeedError('');
      setActiveSeed(next === activeSeed ? `${next}:repeat` : next);
    } catch (reason) {
      setSeedError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  const newSeed = () => {
    const next = `biomechanics-${globalThis.crypto.randomUUID().slice(0, 8)}`;
    setSeedInput(next);
    setSeedError('');
    setActiveSeed(next);
  };

  const resetCurrent = () => {
    setSeedInput(activeSeed);
    setSeedError('');
    setActiveSeed(`${activeSeed}:repeat`);
  };

  const contact = station.lastContact;
  const snapshotSummary = snapshot
    ? `body ${snapshot.body.map((value) => value.toFixed(2)).join(', ')} · ${Object.keys(snapshot.keys).length} keys`
    : 'Awaiting runtime telemetry';

  return (
    <LabShell
      header={(
        <LabHeader
          eyebrow="Biomechanical laboratory · physical gate"
          title="Fly Crossy Connectome"
          context="Controller → FlyBody legs → physical keyboard → authoritative game"
          status={runtimeStatus}
        />
      )}
      controls={(
        <ExperimentControls
          status={`${runtimeStatus} · ${sceneStatus}`}
          controller="80"
          controllerOptions={[
            { value: '80', label: '80 neurons · active' },
            { value: '1000', label: '1,000 neurons · planned', disabled: true },
            { value: '5000', label: '5,000 neurons · planned', disabled: true },
            { value: '20000', label: '20,000 neurons · planned', disabled: true },
            { value: '124289', label: '124,289 neurons · planned', disabled: true },
          ]}
          onControllerChange={() => undefined}
          seed={seedInput}
          seedError={seedError}
          onSeedChange={setSeedInput}
          onApplySeed={applySeed}
          onNewSeed={newSeed}
          onReset={resetCurrent}
          speed={speed}
          onSpeedChange={setSpeed}
        />
      )}
      alerts={(
        <>
          {seedError && <p id="seed-error" className="error" role="alert">{seedError}</p>}
          {(error || atlasError) && <p className="error" role="alert">{error || atlasError}</p>}
        </>
      )}
      environment={(
        <PanelFrame
          title="01 / CROSSING ENVIRONMENT"
          badge="Physical confirmation only"
          className="environment-panel biomechanics-environment"
          footer={(
            <>
              <span>{sceneStatus}</span>
              <span>Score {station.game.score} · Step {station.game.step}</span>
            </>
          )}
        >
          <GameScene
            state={station.game}
            events={station.events}
            onAssetStatus={setAssetStatus}
          />
          <div className="physical-gate" aria-label="Physical keyboard gate">
            <div><span>Requested action</span><strong>{pending?.requestedAction ?? '—'}</strong></div>
            <div><span>Leg target / key</span><strong>{pending?.requestedKey ?? '—'}</strong></div>
            <div><span>Applied action</span><strong>{station.lastAppliedAction ?? '—'}</strong></div>
          </div>
        </PanelFrame>
      )}
      brain={(
        <PanelFrame
          title="02 / LIVE NEURAL ACTIVITY"
          badge="MaleCNS v1.0"
          className="brain-panel"
          footer={(
            <>
              <span>{frame ? `Revision ${activity?.revision ?? 0} · t=${frame.time.toFixed(2)}s` : 'Awaiting runtime telemetry'}</span>
              <a href={asset('data/brain-atlas/NOTICE.md')}>Data notice ↗</a>
            </>
          )}
        >
          {atlas
            ? (
              <BrainScene
                atlas={atlas}
                frame={frame}
                activityMode={frame ? 'simulated-reduced-circuit' : 'none'}
              />
            )
            : <p className="loading" role="status">Loading measured anatomy…</p>}
        </PanelFrame>
      )}
      telemetry={(
        <TelemetryStrip
          heading="PHYSICAL + NEURAL TELEMETRY"
          items={[
            { label: 'Transport', value: transport.phase },
            { label: 'Station', value: station.phase },
            { label: 'Backend', value: transport.backend ?? 'Awaiting runtime telemetry' },
            { label: 'Requested', value: pending?.requestedAction ?? '—' },
            { label: 'Key', value: pending?.requestedKey ?? '—' },
            { label: 'Applied', value: station.lastAppliedAction ?? '—' },
            { label: 'Contact', value: contact ? `${contact.touchedKey ?? 'none'} · ${contact.confirmed ? 'confirmed' : 'not confirmed'}` : 'Awaiting runtime telemetry' },
            { label: 'Snapshot', value: snapshotSummary },
          ]}
          detail={(
            <p>
              The game advances only after FlyBody returns a physical action result.
              Neural values are simulated controller output, not a biological recording.
            </p>
          )}
          error={error}
        />
      )}
      secondary={(
        <details>
          <summary>Biomechanical runtime scope</summary>
          <p>The current verified controller contains 80 simulated neurons. Larger controller sizes remain visible as planned modes and are intentionally disabled until separate checkpoints exist.</p>
          <p>Missing contact, neural, or body snapshots are labeled as awaiting telemetry; the interface never derives them from the requested action.</p>
        </details>
      )}
      footer={<Attribution />}
    />
  );
}
