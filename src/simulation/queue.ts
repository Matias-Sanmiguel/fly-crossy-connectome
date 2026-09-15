import { MAX_FRAME_BYTES } from './protocol.ts';

/** A bounded transport queue that sacrifices only replaceable render work. */
export type OutboundMessage = { type: string; [field: string]: unknown };

const replaceable = new Set(['observation', 'snapshot', 'neural_delta', 'metrics']);
const latestOnly = new Set(['observation', 'snapshot', 'neural_delta', 'neural_keyframe', 'request_keyframe']);

export class OutboundQueueFullError extends Error {
  constructor() {
    super('Outbound queue capacity is saturated with critical messages.');
    this.name = 'OutboundQueueFullError';
  }
}

function validateJsonValue(value: unknown, seen: Set<object>): void {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw Error('Outbound message numbers must be finite.');
    return;
  }
  if (typeof value !== 'object') throw Error('Outbound message must contain only JSON values.');
  if (seen.has(value)) throw Error('Outbound message must not be cyclic.');
  seen.add(value);
  if (Array.isArray(value)) {
    for (const item of value) validateJsonValue(item, seen);
  } else {
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw Error('Outbound message objects must be plain JSON objects.');
    }
    for (const item of Object.values(value)) validateJsonValue(item, seen);
  }
  seen.delete(value);
}

function snapshotMessage<T extends OutboundMessage>(message: T): T {
  if (!message || typeof message.type !== 'string' || message.type.length === 0) {
    throw Error('Outbound message type is invalid.');
  }
  validateJsonValue(message, new Set());
  const serialized = JSON.stringify(message);
  if (new TextEncoder().encode(serialized).byteLength > MAX_FRAME_BYTES) {
    throw Error('Outbound message exceeds 1 MiB.');
  }
  return JSON.parse(serialized) as T;
}

export class OutboundQueue<T extends OutboundMessage = OutboundMessage> {
  private readonly messages: T[] = [];
  private readonly capacity: number;

  constructor(capacity: number) {
    if (!Number.isSafeInteger(capacity) || capacity < 1) {
      throw Error('Outbound queue capacity must be a positive integer.');
    }
    this.capacity = capacity;
  }

  get size(): number { return this.messages.length; }

  /**
   * Admit a frozen JSON snapshot. `false` means a replaceable message was
   * intentionally backpressured; critical-message saturation throws instead.
   */
  enqueue(message: T): boolean {
    const stored = snapshotMessage(message);
    if (latestOnly.has(stored.type)) this.removeType(stored.type);
    while (this.messages.length >= this.capacity) {
      const index = this.messages.findIndex((queued) => replaceable.has(queued.type));
      if (index >= 0) {
        this.messages.splice(index, 1);
        continue;
      }
      if (replaceable.has(stored.type)) return false;
      throw new OutboundQueueFullError();
    }
    this.messages.push(stored);
    return true;
  }

  /** Return queued messages in send order and leave the queue empty. */
  drain(): T[] { return this.messages.splice(0); }

  clear(): void { this.messages.length = 0; }

  removeType(type: string): void {
    for (let index = this.messages.length - 1; index >= 0; index -= 1) {
      if (this.messages[index].type === type) this.messages.splice(index, 1);
    }
  }
}
