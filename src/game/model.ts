import type { ObservationV1 } from './observation.ts';
import type { Action } from './types.ts';

export const POLICY_ACTIONS = ['forward', 'backward', 'left', 'right', 'wait'] as const;
export const OBSERVATION_INPUT_SIZE = 370;

export type ModelSource = {
  kind: 'synthetic' | 'predicted' | 'measured';
  name: string;
  normalization: string;
};

export type Activation = 'tanh' | 'relu' | 'linear';

export type DenseLayer = {
  name: string;
  inputSize: number;
  outputSize: number;
  activation: Activation;
  /** Row-major [output, input]. */
  weights: number[];
  bias: number[];
};

export type DenseNetwork = {
  kind: 'dense';
  inputSize: number;
  layers: DenseLayer[];
};

export type FixedGraphNetwork = {
  kind: 'fixed-graph';
  inputSize: number;
  bodyIds: number[];
  activation: Activation;
  /** Row-major [node, input]. */
  sensoryWeights: number[];
  recurrentSource: number[];
  recurrentTarget: number[];
  recurrentWeights: number[];
  /** Row-major [action, node]. */
  actorWeights: number[];
  actorBias: number[];
};

export type ExportedPolicyV1 = {
  version: 1;
  observationVersion: 1;
  actions: Action[];
  source: ModelSource;
  network: DenseNetwork | FixedGraphNetwork;
  activityBodyIds: number[];
};

const isRecord = (value: unknown): value is Record<string, unknown> => (
  typeof value === 'object' && value !== null && !Array.isArray(value)
);

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (!isRecord(value)) throw Error(`${label} must be an object.`);
  return value;
}

function requirePositiveInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || Number(value) <= 0) {
    throw Error(`${label} must be a positive integer.`);
  }
  return Number(value);
}

function requireFiniteArray(value: unknown, length: number, label: string): number[] {
  if (!Array.isArray(value) || value.length !== length) {
    throw Error(`${label} must contain exactly ${length} values.`);
  }
  if (!value.every((item) => typeof item === 'number' && Number.isFinite(item))) {
    throw Error(`${label} values must be finite numbers.`);
  }
  return [...value] as number[];
}

function requireActivation(value: unknown, label: string): Activation {
  if (value !== 'tanh' && value !== 'relu' && value !== 'linear') {
    throw Error(`${label} uses an unsupported activation.`);
  }
  return value;
}

function parseSource(value: unknown): ModelSource {
  const source = requireRecord(value, 'Policy source');
  const kind = source.kind;
  if (kind !== 'synthetic' && kind !== 'predicted' && kind !== 'measured') {
    throw Error('Policy source must declare a supported kind, name, and normalization.');
  }
  if (typeof source.name !== 'string' || !source.name.trim()
    || typeof source.normalization !== 'string' || !source.normalization.trim()) {
    throw Error('Policy source must declare a supported kind, name, and normalization.');
  }
  return { kind, name: source.name.trim(), normalization: source.normalization.trim() };
}

function parseBodyIds(
  value: unknown,
  visibleIds: ReadonlySet<number>,
  label: string,
): number[] {
  if (!Array.isArray(value)) throw Error(`${label} must be an array.`);
  const seen = new Set<number>();
  return value.map((item) => {
    if (!Number.isSafeInteger(item) || Number(item) <= 0 || !visibleIds.has(Number(item))) {
      throw Error(`${label} must contain only visible MaleCNS body IDs.`);
    }
    const id = Number(item);
    if (seen.has(id)) throw Error(`${label} contains a duplicate body ID.`);
    seen.add(id);
    return id;
  });
}

function parseDenseNetwork(value: Record<string, unknown>): DenseNetwork {
  const inputSize = requirePositiveInteger(value.inputSize, 'Dense inputSize');
  if (!Array.isArray(value.layers) || value.layers.length === 0) {
    throw Error('Dense network must contain at least one layer.');
  }
  let expectedInput = inputSize;
  const names = new Set<string>();
  const layers = value.layers.map((candidate, index): DenseLayer => {
    const layer = requireRecord(candidate, `Dense layer ${index}`);
    if (typeof layer.name !== 'string' || !layer.name.trim() || names.has(layer.name)) {
      throw Error('Dense layer names must be non-empty and unique.');
    }
    names.add(layer.name);
    const layerInput = requirePositiveInteger(layer.inputSize, `Dense layer ${index} inputSize`);
    const outputSize = requirePositiveInteger(layer.outputSize, `Dense layer ${index} outputSize`);
    if (layerInput !== expectedInput) throw Error(`Dense layer ${index} input shape is not contiguous.`);
    expectedInput = outputSize;
    return {
      name: layer.name,
      inputSize: layerInput,
      outputSize,
      activation: requireActivation(layer.activation, `Dense layer ${index}`),
      weights: requireFiniteArray(layer.weights, layerInput * outputSize, `Dense layer ${index} weights`),
      bias: requireFiniteArray(layer.bias, outputSize, `Dense layer ${index} bias`),
    };
  });
  if (expectedInput !== POLICY_ACTIONS.length) {
    throw Error(`Dense actor output must contain ${POLICY_ACTIONS.length} action logits.`);
  }
  return { kind: 'dense', inputSize, layers };
}

function parseIndexArray(value: unknown, edgeCount: number, nodeCount: number, label: string): number[] {
  if (!Array.isArray(value) || value.length !== edgeCount) {
    throw Error(`Fixed graph ${label} topology must match its edge count.`);
  }
  return value.map((item) => {
    if (!Number.isSafeInteger(item) || Number(item) < 0 || Number(item) >= nodeCount) {
      throw Error(`Fixed graph ${label} topology contains an invalid node index.`);
    }
    return Number(item);
  });
}

function parseFixedGraphNetwork(
  value: Record<string, unknown>,
  visibleIds: ReadonlySet<number>,
): FixedGraphNetwork {
  const inputSize = requirePositiveInteger(value.inputSize, 'Fixed graph inputSize');
  const bodyIds = parseBodyIds(value.bodyIds, visibleIds, 'Fixed graph bodyIds');
  if (bodyIds.length === 0) throw Error('Fixed graph must contain at least one body ID.');
  const nodeCount = bodyIds.length;
  if (!Array.isArray(value.recurrentWeights)) {
    throw Error('Fixed graph recurrent weights must be an array.');
  }
  const edgeCount = value.recurrentWeights.length;
  const recurrentSource = parseIndexArray(value.recurrentSource, edgeCount, nodeCount, 'source');
  const recurrentTarget = parseIndexArray(value.recurrentTarget, edgeCount, nodeCount, 'target');
  const edges = new Set<string>();
  for (let index = 0; index < edgeCount; index += 1) {
    const edge = `${recurrentSource[index]}:${recurrentTarget[index]}`;
    if (edges.has(edge)) throw Error('Fixed graph topology contains a duplicate edge.');
    edges.add(edge);
  }
  return {
    kind: 'fixed-graph',
    inputSize,
    bodyIds,
    activation: requireActivation(value.activation, 'Fixed graph'),
    sensoryWeights: requireFiniteArray(
      value.sensoryWeights,
      nodeCount * inputSize,
      'Fixed graph sensory weights',
    ),
    recurrentSource,
    recurrentTarget,
    recurrentWeights: requireFiniteArray(value.recurrentWeights, edgeCount, 'Fixed graph recurrent weights'),
    actorWeights: requireFiniteArray(
      value.actorWeights,
      POLICY_ACTIONS.length * nodeCount,
      'Fixed graph actor weights',
    ),
    actorBias: requireFiniteArray(value.actorBias, POLICY_ACTIONS.length, 'Fixed graph actor bias'),
  };
}

/** Validate an exported policy before it is allowed to construct a controller. */
export function parsePolicy(input: unknown, visibleIds: ReadonlySet<number>): ExportedPolicyV1 {
  const policy = requireRecord(input, 'Policy');
  if (policy.version !== 1) throw Error('Expected policy version 1.');
  if (policy.observationVersion !== 1) throw Error('Expected observation version 1.');
  const actions = policy.actions;
  if (!Array.isArray(actions)
    || actions.length !== POLICY_ACTIONS.length
    || POLICY_ACTIONS.some((action, index) => actions[index] !== action)) {
    throw Error('Policy action order must be forward, backward, left, right, wait.');
  }
  const source = parseSource(policy.source);
  const activityBodyIds = parseBodyIds(policy.activityBodyIds, visibleIds, 'Activity body IDs');
  const networkRecord = requireRecord(policy.network, 'Policy network');
  let network: DenseNetwork | FixedGraphNetwork;
  if (networkRecord.kind === 'dense') network = parseDenseNetwork(networkRecord);
  else if (networkRecord.kind === 'fixed-graph') {
    const fixedNetwork = parseFixedGraphNetwork(networkRecord, visibleIds);
    network = fixedNetwork;
    if (activityBodyIds.length !== fixedNetwork.bodyIds.length
      || activityBodyIds.some((id, index) => fixedNetwork.bodyIds[index] !== id)) {
      throw Error('Fixed graph activity body IDs must match graph body IDs in order.');
    }
  } else throw Error('Policy network kind must be dense or fixed-graph.');

  if (network.kind === 'dense' && activityBodyIds.length > 0) {
    const activityLayer = network.layers.at(-2);
    if (!activityLayer || activityLayer.outputSize !== activityBodyIds.length) {
      throw Error('Dense activity body IDs must map exactly to the final hidden layer.');
    }
  }
  return {
    version: 1,
    observationVersion: 1,
    actions: [...POLICY_ACTIONS],
    source,
    network,
    activityBodyIds,
  };
}

/** Stable numeric encoding shared by exported browser policies. */
export function encodeObservation(observation: ObservationV1): number[] {
  const cells = observation.cells.flat().map((value) => value / 7);
  const motion = observation.motion.flat(2);
  const previousAction = POLICY_ACTIONS.map((action) => Number(observation.previousAction === action));
  const encoded = [...cells, ...motion, observation.support, ...previousAction, observation.edgeDistance];
  if (encoded.length !== OBSERVATION_INPUT_SIZE || !encoded.every(Number.isFinite)) {
    throw Error(`Observation must encode to ${OBSERVATION_INPUT_SIZE} finite values.`);
  }
  return encoded;
}
