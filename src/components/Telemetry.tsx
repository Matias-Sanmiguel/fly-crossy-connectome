import type { Controller, ControllerDecision } from '../game/controllers.ts';
import type { GameState } from '../game/simulation.ts';
import type { ControllerStatus } from '../hooks/useGame.ts';

type TelemetryProps = {
  manual: boolean;
  controller: Controller | null;
  state: GameState;
  decision: ControllerDecision | null;
  reward: number;
  status: ControllerStatus;
  error: string | null;
};

export function Telemetry({ manual, controller, state, decision, reward, status, error }: TelemetryProps) {
  return (
    <section className="model-status" aria-label="Controller telemetry">
      <strong>{manual ? 'MANUAL — NO NEURAL OUTPUT' : 'SIMULATED MODEL ACTIVITY'}</strong>
      <p>
        Controller: {manual ? 'human / manual input' : `${controller?.kind ?? 'unavailable'} / ${controller?.id ?? 'no controller'}`}
      </p>
      <p>
        Step {state.step} · Action {decision?.action ?? '—'} · Step reward {reward.toFixed(2)} · Score {state.score}
      </p>
      <p>Seed: {state.seed} · Status: {status}</p>
      {!manual && <p>Normalization: {controller?.source?.normalization ?? 'No neural activity values supplied.'}</p>}
      {!manual && controller?.source && (
        <p>Source: {controller.source.kind} / {controller.source.name}. Activity is simulated output keyed by measured MaleCNS body IDs.</p>
      )}
      {error && <p className="error" role="alert">Controller error: {error}</p>}
    </section>
  );
}
