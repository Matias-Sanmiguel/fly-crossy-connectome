import { encodeObservation, OBSERVATION_INPUT_SIZE, POLICY_ACTIONS } from './model.ts';
import type { DenseLayer, ExportedPolicyV1, ModelSource } from './model.ts';
import type { ObservationV1 } from './observation.ts';
import type { Action } from './types.ts';

export type ControllerDecision = {
  action: Action;
  activity: [number, number][];
  diagnostics: Record<string, number>;
};

export interface Controller {
  readonly id: string;
  readonly kind: 'scripted' | 'conventional' | 'connectome' | 'remote';
  readonly source?: ModelSource;
  decide(observation: ObservationV1, signal: AbortSignal): Promise<ControllerDecision>;
  reset(seed: string): void;
  dispose(): void;
  subscribeFailure?(listener: (error: Error) => void): () => void;
}

const isRecord = (value: unknown): value is Record<string, unknown> => (
  typeof value === 'object' && value !== null && !Array.isArray(value)
);

export function validateControllerDecision(
  value: unknown,
  visibleIds?: ReadonlySet<number>,
): ControllerDecision {
  if (!isRecord(value) || typeof value.action !== 'string'
    || !POLICY_ACTIONS.includes(value.action as (typeof POLICY_ACTIONS)[number])) {
    throw Error('Controller decision contains an invalid action.');
  }
  if (!Array.isArray(value.activity)) throw Error('Controller decision activity must be an array.');
  if (value.activity.length > 0 && !visibleIds) {
    throw Error('Atlas visible IDs are required for non-empty controller activity.');
  }
  const seen = new Set<number>();
  const activity = value.activity.map((candidate): [number, number] => {
    if (!Array.isArray(candidate) || candidate.length !== 2
      || !Number.isSafeInteger(candidate[0]) || Number(candidate[0]) <= 0
      || typeof candidate[1] !== 'number' || !Number.isFinite(candidate[1])
      || candidate[1] < 0 || candidate[1] > 1
      || (visibleIds && !visibleIds.has(Number(candidate[0])))) {
      throw Error('Controller activity must use unique visible MaleCNS IDs and values in [0, 1].');
    }
    const bodyId = Number(candidate[0]);
    if (seen.has(bodyId)) throw Error('Controller activity contains a duplicate body ID.');
    seen.add(bodyId);
    return [bodyId, candidate[1]];
  });
  if (!isRecord(value.diagnostics)) throw Error('Controller diagnostics must be an object.');
  const diagnostics: Record<string, number> = {};
  for (const [key, diagnostic] of Object.entries(value.diagnostics)) {
    if (typeof diagnostic !== 'number' || !Number.isFinite(diagnostic)) {
      throw Error('Controller diagnostics must contain only finite numbers.');
    }
    diagnostics[key] = diagnostic;
  }
  return { action: value.action as Action, activity, diagnostics };
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw new DOMException('Controller decision aborted.', 'AbortError');
}

export function createScriptedController(actions: readonly Action[]): Controller {
  if (actions.length === 0 || actions.some((action) => !POLICY_ACTIONS.includes(action))) {
    throw Error('Scripted controller requires at least one legal action.');
  }
  let index = 0;
  return {
    id: 'scripted-sequence',
    kind: 'scripted',
    async decide(_observation, signal) {
      throwIfAborted(signal);
      const action = actions[index % actions.length]!;
      index += 1;
      return { action, activity: [], diagnostics: {} };
    },
    reset() { index = 0; },
    dispose() {},
  };
}

function activate(value: number, activation: DenseLayer['activation']): number {
  if (activation === 'tanh') return Math.tanh(value);
  if (activation === 'relu') return Math.max(0, value);
  return value;
}

function runLayer(input: readonly number[], layer: DenseLayer): number[] {
  return Array.from({ length: layer.outputSize }, (_, output) => {
    let value = layer.bias[output]!;
    const offset = output * layer.inputSize;
    for (let index = 0; index < layer.inputSize; index += 1) {
      value += layer.weights[offset + index]! * input[index]!;
    }
    return activate(value, layer.activation);
  });
}

export function createDensePolicyController(policy: ExportedPolicyV1): Controller {
  if (policy.network.kind !== 'dense') throw Error('Dense controller requires a dense policy.');
  const { network } = policy;
  if (network.inputSize !== OBSERVATION_INPUT_SIZE) {
    throw Error('Dense policy input shape does not match ObservationV1.');
  }
  return {
    id: policy.source.name,
    kind: 'conventional',
    source: policy.source,
    async decide(observation, signal) {
      throwIfAborted(signal);
      let values = encodeObservation(observation);
      let hidden: number[] = [];
      network.layers.forEach((layer, index) => {
        values = runLayer(values, layer);
        if (index === network.layers.length - 2) hidden = values;
      });
      throwIfAborted(signal);
      let selected = 0;
      for (let index = 1; index < values.length; index += 1) {
        if (values[index]! > values[selected]!) selected = index;
      }
      const diagnostics = Object.fromEntries(
        POLICY_ACTIONS.map((action, index) => [`logit.${action}`, values[index]!] as const),
      );
      const activity = policy.activityBodyIds.map((bodyId, index): [number, number] => [
        bodyId,
        Math.max(0, Math.min(1, (hidden[index]! + 1) / 2)),
      ]);
      return { action: policy.actions[selected]!, activity, diagnostics };
    },
    reset() {},
    dispose() {},
  };
}
