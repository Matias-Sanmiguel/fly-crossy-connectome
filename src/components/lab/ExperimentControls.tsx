import type { ReactNode } from 'react';

export type ControllerOption = {
  value: string;
  label: string;
  disabled?: boolean;
};

type ExperimentControlsProps = {
  status: string;
  controller: string;
  controllerOptions: readonly ControllerOption[];
  onControllerChange: (value: string) => void;
  seed: string;
  seedError?: string;
  onSeedChange: (value: string) => void;
  onApplySeed: () => void;
  onNewSeed: () => void;
  onReset: () => void;
  speed: number;
  speedDisabled?: boolean;
  onSpeedChange: (value: number) => void;
  action?: ReactNode;
};

export function ExperimentControls({
  status,
  controller,
  controllerOptions,
  onControllerChange,
  seed,
  seedError,
  onSeedChange,
  onApplySeed,
  onNewSeed,
  onReset,
  speed,
  speedDisabled = false,
  onSpeedChange,
  action,
}: ExperimentControlsProps) {
  return (
    <div className="experiment-controls" aria-label="Experiment controls">
      <span className="status" role="status" aria-live="polite">{status}</span>
      <div className="controls">
        <label>
          Controller
          <select value={controller} onChange={(event) => onControllerChange(event.target.value)}>
            {controllerOptions.map((option) => (
              <option key={option.value} value={option.value} disabled={option.disabled}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Seed
          <input
            className="seed-input"
            value={seed}
            maxLength={64}
            aria-invalid={seedError ? 'true' : 'false'}
            aria-describedby={seedError ? 'seed-error' : undefined}
            onChange={(event) => onSeedChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                onApplySeed();
              }
            }}
          />
        </label>
        <button type="button" onClick={onApplySeed}>Apply seed</button>
        <button type="button" onClick={onNewSeed}>New seed</button>
        <button type="button" onClick={onReset}>Reset current</button>
        <label>
          Autonomous speed
          <select
            value={speed}
            disabled={speedDisabled}
            onChange={(event) => onSpeedChange(Number(event.target.value))}
          >
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={4}>4×</option>
          </select>
        </label>
        {action}
      </div>
    </div>
  );
}
