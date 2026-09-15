/** A bounded transport queue that sacrifices only replaceable render work. */
export type OutboundMessage = { type: string; [field: string]: unknown };

const replaceable = new Set(['observation', 'snapshot', 'neural_delta', 'metrics']);
const latestOnly = new Set(['observation', 'snapshot', 'neural_delta', 'neural_keyframe', 'request_keyframe']);

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

  enqueue(message: T): void {
    if (latestOnly.has(message.type)) this.removeType(message.type);
    this.messages.push(message);
    this.trimReplaceableMessages();
  }

  /** Return queued messages in send order and leave the queue empty. */
  drain(): T[] { return this.messages.splice(0); }

  clear(): void { this.messages.length = 0; }

  removeType(type: string): void {
    for (let index = this.messages.length - 1; index >= 0; index -= 1) {
      if (this.messages[index].type === type) this.messages.splice(index, 1);
    }
  }

  private trimReplaceableMessages(): void {
    while (this.messages.length > this.capacity) {
      const index = this.messages.findIndex((message) => replaceable.has(message.type));
      if (index < 0) return;
      this.messages.splice(index, 1);
    }
  }
}
