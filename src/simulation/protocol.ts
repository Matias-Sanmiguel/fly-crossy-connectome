export const MAX_FRAME_BYTES = 1_048_576;
export const MAX_NEURAL_UPDATES = 20_000;
export const MAX_STRING_LENGTH = 256;
export const MAX_SEQUENCE = 2 ** 53 - 1;

export type PopulationSize = 80 | 1000 | 5000 | 20000 | 124289;
export type BackendPreference = 'auto' | 'cpu' | 'gpu' | 'gpu-strict';
export type Action = 'forward' | 'backward' | 'left' | 'right' | 'wait';
export type KeyName = 'W' | 'A' | 'S' | 'D' | 'SPACE_LEFT' | 'SPACE_RIGHT';
export type MotorPhase = 'neutral' | 'targeting' | 'reaching' | 'pressing' | 'confirmed' | 'retracting' | 'settling' | 'failed';

type Envelope = {
  version: 2;
  sessionId: string;
  episodeId: string;
  sequence: number;
  simulationTime: number;
};

export type Ready = Envelope & { type: 'ready'; backend: 'cpu' | 'gpu'; population: PopulationSize; graphHash: string; checkpointHash: string; acceptedVersions?: 2[]; maxFrameBytes?: number; maxNeuralUpdates?: number };
export type ResetComplete = Envelope & { type: 'reset_complete' };
export type Intention = Envelope & { type: 'intention'; intentionId: string; action: Action; motorPhase: MotorPhase };
export type Snapshot = Envelope & { type: 'snapshot'; body: [number, number, number]; joints: number[]; keys: Partial<Record<KeyName, number>> };
export type NeuralKeyframe = Envelope & { type: 'neural_keyframe'; updates: NeuralUpdate[] };
export type NeuralDelta = Envelope & { type: 'neural_delta'; updates: NeuralUpdate[] };
export type Contact = Envelope & { type: 'contact'; intentionId: string; requestedKey: KeyName; touchedKey: KeyName | null; travel: number; force: number; debounce: number; confirmed: boolean };
export type ActionResult = Envelope & { type: 'action_result'; intentionId: string; result: 'confirmed' | 'waited' | 'failed' };
export type Metrics = Envelope & { type: 'metrics'; physicsHz: number; motorHz: number; neuralHz: number; renderHz: number; latencyMs: number; droppedRenderFrames: number; backendUtilization?: number };
export type Paused = Envelope & { type: 'paused'; code: string; message: string };
export type ErrorMessage = Envelope & { type: 'error'; code: string; message: string };
export type NeuralUpdate = { neuronId: number; value: number };
export type ServerMessage = Ready | ResetComplete | Intention | Snapshot | NeuralKeyframe | NeuralDelta | Contact | ActionResult | Metrics | Paused | ErrorMessage;

const populations = new Set<PopulationSize>([80, 1000, 5000, 20000, 124289]);
const resolvedBackends = new Set(['cpu', 'gpu']);
const actions = new Set<Action>(['forward', 'backward', 'left', 'right', 'wait']);
const keyNames = new Set<KeyName>(['W', 'A', 'S', 'D', 'SPACE_LEFT', 'SPACE_RIGHT']);
const motorPhases = new Set<MotorPhase>(['neutral', 'targeting', 'reaching', 'pressing', 'confirmed', 'retracting', 'settling', 'failed']);
const serverTypes = new Set<ServerMessage['type']>(['ready', 'reset_complete', 'intention', 'snapshot', 'neural_keyframe', 'neural_delta', 'contact', 'action_result', 'metrics', 'paused', 'error']);

function fail(message: string): never { throw new TypeError(`Invalid simulation protocol message: ${message}`); }
function isRecord(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value); }
function finiteNumber(value: unknown, field: string, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) fail(`${field} must be a finite number between ${min} and ${max}`);
  return value;
}
function integer(value: unknown, field: string, min: number, max: number): number {
  const number = finiteNumber(value, field, min, max);
  if (!Number.isSafeInteger(number)) fail(`${field} must be a safe integer`);
  return number;
}
function string(value: unknown, field: string, pattern?: RegExp): string {
  if (typeof value !== 'string' || value.length < 1 || value.length > MAX_STRING_LENGTH || (pattern && !pattern.test(value))) fail(`${field} is invalid`);
  return value;
}
function onlyKeys(value: Record<string, unknown>, keys: readonly string[]): void {
  const allowed = new Set(keys);
  for (const key of Object.keys(value)) if (!allowed.has(key)) fail(`unexpected key ${key}`);
}
function array(value: unknown, field: string, min: number, max: number): unknown[] {
  if (!Array.isArray(value) || value.length < min || value.length > max) fail(`${field} must contain between ${min} and ${max} entries`);
  return value;
}
function frame(value: unknown): Record<string, unknown> {
  let decoded = value;
  if (typeof decoded === 'string') {
    if (new TextEncoder().encode(decoded).byteLength > MAX_FRAME_BYTES) fail('JSON frame exceeds 1 MiB');
    try { decoded = JSON.parse(decoded); } catch { fail('JSON frame is malformed'); }
  }
  if (!isRecord(decoded)) fail('frame must be a JSON object');
  let encoded: string;
  try { encoded = JSON.stringify(decoded); } catch { fail('frame is not JSON serializable'); }
  if (encoded === undefined || new TextEncoder().encode(encoded).byteLength > MAX_FRAME_BYTES) fail('JSON frame exceeds 1 MiB');
  return decoded;
}
function envelope(value: Record<string, unknown>, keys: readonly string[]): void {
  onlyKeys(value, keys);
  if (value.version !== 2) fail('version must be 2');
  string(value.sessionId, 'sessionId', /^[se]-[a-z0-9]{8,64}$/);
  string(value.episodeId, 'episodeId', /^[se]-[a-z0-9]{8,64}$/);
  integer(value.sequence, 'sequence', 0, MAX_SEQUENCE);
  finiteNumber(value.simulationTime, 'simulationTime', 0, 86_400);
}
function validateUpdates(value: unknown): void {
  for (const update of array(value, 'updates', 0, MAX_NEURAL_UPDATES)) {
    if (!isRecord(update)) fail('update must be an object');
    onlyKeys(update, ['neuronId', 'value']);
    integer(update.neuronId, 'neuronId', 0, MAX_SEQUENCE);
    finiteNumber(update.value, 'value', -1_000_000, 1_000_000);
  }
}

/** Parse one strict, bounded server control frame. */
export function parseServerMessage(value: unknown): ServerMessage {
  const message = frame(value);
  if (typeof message.type !== 'string' || !serverTypes.has(message.type as ServerMessage['type'])) fail('type is not a supported server message');
  switch (message.type) {
    case 'ready': {
      envelope(message, ['type', 'version', 'sessionId', 'episodeId', 'sequence', 'simulationTime', 'backend', 'population', 'graphHash', 'checkpointHash', 'acceptedVersions', 'maxFrameBytes', 'maxNeuralUpdates']);
      if (!resolvedBackends.has(string(message.backend, 'backend'))) fail('backend is invalid');
      if (!populations.has(message.population as PopulationSize)) fail('population is invalid');
      string(message.graphHash, 'graphHash', /^[a-f0-9]{64}$/); string(message.checkpointHash, 'checkpointHash', /^[a-f0-9]{64}$/);
      if (message.acceptedVersions !== undefined) for (const version of array(message.acceptedVersions, 'acceptedVersions', 1, 1)) if (version !== 2) fail('acceptedVersions is invalid');
      if (message.maxFrameBytes !== undefined) integer(message.maxFrameBytes, 'maxFrameBytes', 1, MAX_FRAME_BYTES);
      if (message.maxNeuralUpdates !== undefined) integer(message.maxNeuralUpdates, 'maxNeuralUpdates', 1, MAX_NEURAL_UPDATES);
      break;
    }
    case 'reset_complete': envelope(message, ['type', 'version', 'sessionId', 'episodeId', 'sequence', 'simulationTime']); break;
    case 'intention':
      envelope(message, ['type', 'version', 'sessionId', 'episodeId', 'sequence', 'simulationTime', 'intentionId', 'action', 'motorPhase']);
      string(message.intentionId, 'intentionId', /^i-[a-z0-9]{8,64}$/); if (!actions.has(message.action as Action)) fail('action is invalid'); if (!motorPhases.has(message.motorPhase as MotorPhase)) fail('motorPhase is invalid'); break;
    case 'snapshot': {
      envelope(message, ['type', 'version', 'sessionId', 'episodeId', 'sequence', 'simulationTime', 'body', 'joints', 'keys']);
      for (const coordinate of array(message.body, 'body', 3, 3)) finiteNumber(coordinate, 'body', -10_000, 10_000);
      for (const joint of array(message.joints, 'joints', 0, 256)) finiteNumber(joint, 'joints', -100, 100);
      if (!isRecord(message.keys)) fail('keys must be an object');
      for (const [key, travel] of Object.entries(message.keys)) { if (!keyNames.has(key as KeyName)) fail('keys contains an invalid key'); finiteNumber(travel, 'keys', 0, 1); }
      break;
    }
    case 'neural_keyframe': case 'neural_delta':
      envelope(message, ['type', 'version', 'sessionId', 'episodeId', 'sequence', 'simulationTime', 'updates']); validateUpdates(message.updates); break;
    case 'contact':
      envelope(message, ['type', 'version', 'sessionId', 'episodeId', 'sequence', 'simulationTime', 'intentionId', 'requestedKey', 'touchedKey', 'travel', 'force', 'debounce', 'confirmed']);
      string(message.intentionId, 'intentionId', /^i-[a-z0-9]{8,64}$/); if (!keyNames.has(message.requestedKey as KeyName)) fail('requestedKey is invalid'); if (message.touchedKey !== null && !keyNames.has(message.touchedKey as KeyName)) fail('touchedKey is invalid'); finiteNumber(message.travel, 'travel', 0, 1); finiteNumber(message.force, 'force', 0, 10_000); finiteNumber(message.debounce, 'debounce', 0, 10); if (typeof message.confirmed !== 'boolean') fail('confirmed must be boolean'); break;
    case 'action_result':
      envelope(message, ['type', 'version', 'sessionId', 'episodeId', 'sequence', 'simulationTime', 'intentionId', 'result']); string(message.intentionId, 'intentionId', /^i-[a-z0-9]{8,64}$/); if (message.result !== 'confirmed' && message.result !== 'waited' && message.result !== 'failed') fail('result is invalid'); break;
    case 'metrics':
      envelope(message, ['type', 'version', 'sessionId', 'episodeId', 'sequence', 'simulationTime', 'physicsHz', 'motorHz', 'neuralHz', 'renderHz', 'latencyMs', 'droppedRenderFrames', 'backendUtilization']);
      for (const field of ['physicsHz', 'motorHz', 'neuralHz', 'renderHz'] as const) finiteNumber(message[field], field, 0, 100_000); finiteNumber(message.latencyMs, 'latencyMs', 0, 1_000_000); integer(message.droppedRenderFrames, 'droppedRenderFrames', 0, MAX_SEQUENCE); if (message.backendUtilization !== undefined) finiteNumber(message.backendUtilization, 'backendUtilization', 0, 1); break;
    case 'paused': case 'error':
      envelope(message, ['type', 'version', 'sessionId', 'episodeId', 'sequence', 'simulationTime', 'code', 'message']); string(message.code, 'code'); string(message.message, 'message'); break;
  }
  return message as ServerMessage;
}
