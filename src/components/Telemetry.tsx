import type { Controller, ControllerDecision } from '../game/controllers.ts';
import type { GameState } from '../game/simulation.ts';
import type { ControllerStatus } from '../hooks/useGame.ts';
import type { ActivityPresentation } from '../game/provenance.ts';
import { TelemetryStrip } from './lab/TelemetryStrip.tsx';

type TelemetryProps = {
  controller: Controller | null;
  activity: ActivityPresentation;
  state: GameState;
  decision: ControllerDecision | null;
  reward: number;
  status: ControllerStatus;
  error: string | null;
};

export function Telemetry({ controller, activity, state, decision, reward, status, error }: TelemetryProps) {
  const manual = controller === null;
  const sourceDescription = activity.kind === 'simulated-reduced-circuit'
    ? 'Simulated values from the fixed reduced circuit, keyed by measured MaleCNS body IDs.'
    : activity.kind === 'model-output'
      ? 'Mapped model output keyed by measured MaleCNS body IDs; it is not measured activity.'
      : 'This controller supplied no mapped neural output; the brain view shows anatomy only.';
  return (
    <TelemetryStrip
      heading={activity.heading}
      items={[
        {
          label: 'Controller',
          value: manual
            ? 'human / manual input'
            : `${controller?.kind ?? 'unavailable'} / ${controller?.id ?? 'no controller'}`,
        },
        { label: 'Decision', value: decision?.action ?? '—' },
        { label: 'Step', value: state.step },
        { label: 'Reward', value: reward.toFixed(2) },
        { label: 'Score', value: state.score },
        { label: 'Runtime', value: status },
      ]}
      detail={(
        <>
          <p>{sourceDescription}</p>
          {!manual && activity.kind !== 'none' && (
            <p>Normalization: {controller?.source?.normalization ?? 'No normalization declared.'}</p>
          )}
          {!manual && controller?.source && (
            <p>Source: {controller.source.kind} / {controller.source.name}.</p>
          )}
          <p>Seed: {state.seed}</p>
        </>
      )}
      error={error ? `Controller error: ${error}` : null}
    />
  );
}
