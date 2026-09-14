import type { Controller, ControllerDecision } from '../game/controllers.ts';
import type { GameState } from '../game/simulation.ts';
import type { ControllerStatus } from '../hooks/useGame.ts';
import type { ActivityPresentation } from '../game/provenance.ts';

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
    <section className="model-status" aria-label="Controller telemetry">
      <strong>{activity.heading}</strong>
      <p>
        Controller: {manual ? 'human / manual input' : `${controller?.kind ?? 'unavailable'} / ${controller?.id ?? 'no controller'}`}
      </p>
      <p>
        Step {state.step} · Action {decision?.action ?? '—'} · Step reward {reward.toFixed(2)} · Score {state.score}
      </p>
      <p>Seed: {state.seed} · Status: {status}</p>
      <p>{sourceDescription}</p>
      {!manual && activity.kind !== 'none' && (
        <p>Normalization: {controller?.source?.normalization ?? 'No normalization declared.'}</p>
      )}
      {!manual && controller?.source && <p>Source: {controller.source.kind} / {controller.source.name}.</p>}
      {error && <p className="error" role="alert">Controller error: {error}</p>}
    </section>
  );
}
