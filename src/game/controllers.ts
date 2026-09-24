import {
  encodeObservation,
  OBSERVATION_INPUT_SIZE,
  POLICY_ACTIONS,
  parsePolicy,
  runDenseNetwork,
  runFixedGraphNetwork,
} from './model.ts';
import type { ExportedPolicyV1, ModelSource } from './model.ts';
import { OBSERVATION_VERSION, type ObservationV1 } from './observation.ts';
import type { Action } from './types.ts';
import type { ActivityProvenance } from './provenance.ts';

export type ControllerDecision = {
  action: Action;
  activity: [number, number][];
  diagnostics: Record<string, number>;
};

export interface Controller {
  readonly id: string;
  readonly kind: 'scripted' | 'conventional' | 'connectome' | 'remote';
  readonly activityProvenance: ActivityProvenance;
  readonly source?: ModelSource;
  decide(observation: ObservationV1, signal: AbortSignal): Promise<ControllerDecision>;
  reset(seed: string): void;
  dispose(): void;
  subscribeFailure?(listener: (error: Error) => void): () => void;
}

export const BUNDLED_CONNECTOME_POLICY_PATH: string | null = null;

export function parseBundledConnectomePolicy(
  value: unknown,
  visibleIds: ReadonlySet<number>,
): ExportedPolicyV1 {
  const policy = parsePolicy(value, visibleIds);
  if (policy.network.kind !== 'fixed-graph' || policy.activityBodyIds.length === 0) {
    throw Error('Bundled autoplay policy must provide mapped reduced-connectome activity.');
  }
  if (policy.network.inputSize !== OBSERVATION_INPUT_SIZE) {
    throw Error('Bundled autoplay policy input shape does not match ObservationV4.');
  }
  if (policy.observationVersion !== OBSERVATION_VERSION) {
    throw Error('Bundled autoplay policy observation version does not match the current environment.');
  }
  return policy;
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
    activityProvenance: 'none',
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

export function createDensePolicyController(policy: ExportedPolicyV1): Controller {
  if (policy.network.kind !== 'dense') throw Error('Dense controller requires a dense policy.');
  const { network } = policy;
  if (network.inputSize !== OBSERVATION_INPUT_SIZE) {
    throw Error('Dense policy input shape does not match ObservationV1.');
  }
  return {
    id: policy.source.name,
    kind: 'conventional',
    activityProvenance: policy.activityBodyIds.length > 0 ? 'model-output' : 'none',
    source: policy.source,
    async decide(observation, signal) {
      throwIfAborted(signal);
      const { output: values, hidden } = runDenseNetwork(network, encodeObservation(observation));
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

export function createFixedGraphPolicyController(policy: ExportedPolicyV1): Controller {
  if (policy.network.kind !== 'fixed-graph') {
    throw Error('Connectome controller requires a fixed graph policy.');
  }
  const { network } = policy;
  if (network.inputSize !== OBSERVATION_INPUT_SIZE) {
    throw Error('Fixed graph policy input shape does not match ObservationV1.');
  }
  let hidden = Array(network.bodyIds.length).fill(0) as number[];
  return {
    id: policy.source.name,
    kind: 'connectome',
    activityProvenance: 'simulated-reduced-circuit',
    source: policy.source,
    async decide(observation, signal) {
      throwIfAborted(signal);
      const result = runFixedGraphNetwork(network, encodeObservation(observation), hidden);
      throwIfAborted(signal);
      hidden = result.activity;
      let selected = 0;
      for (let index = 1; index < result.logits.length; index += 1) {
        if (result.logits[index]! > result.logits[selected]!) selected = index;
      }
      const diagnostics: Record<string, number> = Object.fromEntries(
        POLICY_ACTIONS.map((action, index) => [`logit.${action}`, result.logits[index]!] as const),
      );
      if (result.risk) {
        POLICY_ACTIONS.forEach((action, index) => {
          diagnostics[`risk.${action}`] = result.risk![index]!;
        });
      }
      if (result.route) {
        POLICY_ACTIONS.forEach((action, index) => {
          diagnostics[`route.${action}`] = result.route![index]!;
        });
      }
      const activity = policy.activityBodyIds.map((bodyId, index): [number, number] => [
        bodyId,
        Math.max(0, Math.min(1, (result.activity[index]! + 1) / 2)),
      ]);
      return { action: policy.actions[selected]!, activity, diagnostics };
    },
    reset() { hidden = Array(network.bodyIds.length).fill(0) as number[]; },
    dispose() { hidden = []; },
  };
}
