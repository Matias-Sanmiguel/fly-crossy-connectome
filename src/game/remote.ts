import { validateControllerDecision } from './controllers.ts';
import type { Controller, ControllerDecision } from './controllers.ts';
import type { ObservationV1 } from './observation.ts';

export type RemoteResetMessage = { type: 'reset'; version: 1; seed: string };
export type RemoteObserveMessage = {
  type: 'observe';
  version: 1;
  requestId: number;
  observation: ObservationV1;
};
export type RemoteDecisionMessage = {
  type: 'decision';
  version: 1;
  requestId: number;
  decision: ControllerDecision;
};

type SocketLike = {
  readonly readyState: number;
  addEventListener(type: string, listener: EventListener): void;
  removeEventListener(type: string, listener: EventListener): void;
  send(data: string): void;
  close(): void;
};

export type RemoteControllerOptions = {
  timeoutMs?: number;
  visibleIds?: ReadonlySet<number>;
  socketFactory?: (url: string) => SocketLike;
};

type PendingDecision = {
  requestId: number;
  resolve: (decision: ControllerDecision) => void;
  reject: (error: Error) => void;
  timeout: ReturnType<typeof setTimeout>;
  signal: AbortSignal;
  onAbort: () => void;
};

const isRecord = (value: unknown): value is Record<string, unknown> => (
  typeof value === 'object' && value !== null && !Array.isArray(value)
);

export function createRemoteController(
  url: string,
  options: RemoteControllerOptions = {},
): Controller {
  if (!/^wss?:\/\//i.test(url)) throw Error('Remote controller URL must use ws:// or wss://.');
  const timeoutMs = options.timeoutMs ?? 2_000;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw Error('Remote timeout must be positive.');
  const socket = options.socketFactory
    ? options.socketFactory(url)
    : new WebSocket(url);
  let nextRequestId = 1;
  let pending: PendingDecision | null = null;
  let queuedReset: RemoteResetMessage | null = null;
  let queuedObservation: RemoteObserveMessage | null = null;
  let connectionFailure: Error | null = null;
  const failureListeners = new Set<(error: Error) => void>();
  let disposed = false;

  const cleanup = (current: PendingDecision) => {
    clearTimeout(current.timeout);
    current.signal.removeEventListener('abort', current.onAbort);
    if (pending === current) {
      pending = null;
      queuedObservation = null;
    }
  };
  const rejectPending = (error: Error) => {
    if (!pending) return;
    const current = pending;
    cleanup(current);
    current.reject(error);
  };
  const reportConnectionFailure = (error: Error) => {
    if (disposed || connectionFailure) return;
    connectionFailure = error;
    rejectPending(error);
    for (const listener of failureListeners) listener(error);
  };
  const send = (message: RemoteResetMessage | RemoteObserveMessage) => {
    if (socket.readyState !== 1) throw Error('Remote controller is not connected.');
    socket.send(JSON.stringify(message));
  };
  const onOpen: EventListener = () => {
    try {
      if (queuedReset) {
        const message = queuedReset;
        queuedReset = null;
        send(message);
      }
      if (queuedObservation) {
        const message = queuedObservation;
        queuedObservation = null;
        send(message);
      }
    } catch (error) {
      rejectPending(error instanceof Error ? error : Error(String(error)));
    }
  };
  const onMessage: EventListener = (event) => {
    try {
      const raw = JSON.parse(String((event as MessageEvent).data));
      if (!isRecord(raw) || raw.type !== 'decision') {
        throw Error('Remote controller sent an unknown message type.');
      }
      if (raw.version !== 1) throw Error('Remote controller requires protocol version 1.');
      if (!Number.isSafeInteger(raw.requestId)) throw Error('Remote decision request ID is invalid.');
      if (!pending || raw.requestId !== pending.requestId) {
        throw Error('Remote decision request ID does not match the pending request.');
      }
      const current = pending;
      const decision = validateControllerDecision(raw.decision, options.visibleIds);
      cleanup(current);
      current.resolve(decision);
    } catch (error) {
      rejectPending(error instanceof Error ? error : Error(String(error)));
    }
  };
  const onError: EventListener = () => reportConnectionFailure(Error('Remote controller connection failed.'));
  const onClose: EventListener = () => reportConnectionFailure(Error('Remote controller disconnected.'));
  socket.addEventListener('open', onOpen);
  socket.addEventListener('message', onMessage);
  socket.addEventListener('error', onError);
  socket.addEventListener('close', onClose);

  return {
    id: url,
    kind: 'remote',
    activityProvenance: 'model-output',
    decide(observation, signal) {
      if (disposed) return Promise.reject(Error('Remote controller is disposed.'));
      if (connectionFailure) return Promise.reject(connectionFailure);
      if (pending) return Promise.reject(Error('Remote controller already has a pending decision.'));
      if (signal.aborted) return Promise.reject(new DOMException('Controller decision aborted.', 'AbortError'));
      const requestId = nextRequestId;
      nextRequestId += 1;
      return new Promise<ControllerDecision>((resolve, reject) => {
        const onAbort = () => {
          if (!pending || pending.requestId !== requestId) return;
          const current = pending;
          cleanup(current);
          reject(new DOMException('Controller decision aborted.', 'AbortError'));
        };
        const timeout = setTimeout(() => {
          if (!pending || pending.requestId !== requestId) return;
          const current = pending;
          cleanup(current);
          reject(Error(`Remote controller timed out after ${timeoutMs} ms.`));
        }, timeoutMs);
        pending = { requestId, resolve, reject, timeout, signal, onAbort };
        signal.addEventListener('abort', onAbort, { once: true });
        const message: RemoteObserveMessage = { type: 'observe', version: 1, requestId, observation };
        try {
          if (socket.readyState === 1) send(message);
          else if (socket.readyState === 0) queuedObservation = message;
          else throw Error('Remote controller is not connected.');
        } catch (error) {
          const current = pending;
          if (current) cleanup(current);
          reject(error instanceof Error ? error : Error(String(error)));
        }
      });
    },
    reset(seed) {
      rejectPending(new DOMException('Controller reset.', 'AbortError'));
      const message: RemoteResetMessage = { type: 'reset', version: 1, seed };
      if (socket.readyState === 1) send(message);
      else queuedReset = message;
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      rejectPending(Error('Remote controller disposed.'));
      socket.removeEventListener('open', onOpen);
      socket.removeEventListener('message', onMessage);
      socket.removeEventListener('error', onError);
      socket.removeEventListener('close', onClose);
      failureListeners.clear();
      socket.close();
    },
    subscribeFailure(listener) {
      failureListeners.add(listener);
      if (connectionFailure) listener(connectionFailure);
      return () => failureListeners.delete(listener);
    },
  };
}
