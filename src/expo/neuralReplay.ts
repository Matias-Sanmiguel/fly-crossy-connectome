import type { Controller } from '../game/controllers.ts';
import type { ModelSource } from '../game/model.ts';
import type { Action } from '../game/types.ts';

export type ExpoReplay = { seed: string; actions: Action[] };
export type NeuralReplay = {
  bodyIds: number[];
  data: Uint8Array;
  source: ModelSource;
  frameCount: number;
};
const record = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const actions = new Set(['forward', 'backward', 'left', 'right', 'wait']);
const hash = /^[a-f0-9]{64}$/;

export function parseExpoReplay(value: unknown): ExpoReplay {
  if (!record(value) || value.version !== 1 || value.kind !== 'full-malecns-checkpoint-action-replay'
    || typeof value.seed !== 'string' || !value.seed || !Array.isArray(value.actions)
    || !value.actions.length || value.actions.some(v => typeof v !== 'string' || !actions.has(v))) {
    throw Error('El replay Full MaleCNS tiene un formato inválido.');
  }
  return { seed: value.seed, actions: value.actions as Action[] };
}

export async function sha256(data: Uint8Array): Promise<string> {
  const bytes = data.buffer instanceof ArrayBuffer
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), v => v.toString(16).padStart(2, '0')).join('');
}

/** Browsers already decode HTTP Content-Encoding; static hosts may serve raw gzip instead. */
export async function decodeNeuralResponse(response: Response): Promise<Uint8Array> {
  if (!response.body) throw Error('Actividad neuronal vacía.');
  const httpDecoded = response.headers.get('content-encoding')?.toLowerCase().split(',').map(v => v.trim()).includes('gzip');
  const stream = httpDecoded ? response.body : response.body.pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Fail closed: activity must belong to these exact actions and measured visible IDs. */
export async function parseNeuralReplay(value: unknown, data: Uint8Array, replay: ExpoReplay, visibleIds: ReadonlySet<number>): Promise<NeuralReplay> {
  if (!record(value) || value.version !== 1 || value.kind !== 'full-malecns-neural-state-replay'
    || value.encoding !== 'frame-major-uint8-gzip' || value.seed !== replay.seed
    || value.frameCount !== replay.actions.length || !Array.isArray(value.bodyIds) || !value.bodyIds.length
    || value.bodyIds.length > visibleIds.size
    || value.bodyIds.some(v => !Number.isSafeInteger(v) || !visibleIds.has(v as number))
    || new Set(value.bodyIds).size !== value.bodyIds.length
    || data.length !== value.bodyIds.length * replay.actions.length
    || typeof value.checkpointSha256 !== 'string' || !hash.test(value.checkpointSha256)
    || typeof value.dataSha256 !== 'string' || !hash.test(value.dataSha256)
    || typeof value.actionsSha256 !== 'string' || !hash.test(value.actionsSha256)
    || !record(value.source) || value.source.kind !== 'predicted'
    || typeof value.source.name !== 'string' || !value.source.name
    || typeof value.source.normalization !== 'string' || !value.source.normalization) {
    throw Error('Actividad Full MaleCNS inválida o no sincronizada.');
  }
  const actionHash = await sha256(new TextEncoder().encode(JSON.stringify(replay.actions)));
  if (actionHash !== value.actionsSha256 || await sha256(data) !== value.dataSha256) {
    throw Error('La actividad neuronal no corresponde al replay o está dañada.');
  }
  return {
    bodyIds: value.bodyIds as number[], data, frameCount: replay.actions.length,
    source: { kind: 'predicted', name: value.source.name, normalization: value.source.normalization, checkpointHash: value.checkpointSha256,
      ...(typeof value.source.selectionRule === 'string' ? { selectionRule: value.source.selectionRule } : {}) },
  };
}

/** One recorded state per decision. No interpolation, invented neurons or safety reflex. */
export function createNeuralReplayController(replay: ExpoReplay, neural: NeuralReplay): Controller {
  let index = 0;
  return {
    id: 'full-malecns-expo-replay', kind: 'scripted', activityProvenance: 'model-output', source: neural.source,
    async decide(_observation, signal) {
      signal.throwIfAborted();
      if (index >= replay.actions.length) throw Error('Replay terminado.');
      const activity: [number, number][] = [];
      const offset = index * neural.bodyIds.length;
      for (let i = 0; i < neural.bodyIds.length; i += 1) {
        const value = neural.data[offset + i]!;
        if (value !== 0) activity.push([neural.bodyIds[i]!, value / 255]);
      }
      const action = replay.actions[index]!;
      index += 1;
      return { action, activity, diagnostics: { replayStep: index - 1, activeNeurons: activity.length } };
    },
    reset(seed) {
      if (seed !== replay.seed) throw Error('El replay requiere su seed original.');
      index = 0;
    },
    dispose() {},
  };
}
