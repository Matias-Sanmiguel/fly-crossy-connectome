import {
  parseServerMessage,
  type BackendPreference,
  type PopulationSize,
  type ServerMessage,
} from './protocol.ts';
import { OutboundQueue, type OutboundMessage } from './queue.ts';

export type SimulationClientState = {
  phase: 'connecting' | 'ready' | 'acting' | 'paused' | 'error' | 'closed';
  sessionId: string;
  episodeId: string;
  backend: 'cpu' | 'gpu' | null;
  error: string | null;
};

export type SimulationSocket = {
  readonly readyState: number;
  addEventListener(type: string, listener: (event: any) => void): void;
  removeEventListener(type: string, listener: (event: any) => void): void;
  send(data: string): void;
  close(): void;
};

export type SimulationTimerFactory = {
  setTimeout(callback: () => void, delayMs: number): unknown;
  clearTimeout(handle: unknown): void;
};

export type SimulationClientOptions = {
  socketFactory?: (url: string) => SimulationSocket;
  timerFactory?: SimulationTimerFactory;
  sessionIdFactory?: () => string;
  episodeIdFactory?: () => string;
  reconnectDelayMs?: number;
  maxQueue?: number;
  uiBuild?: string;
};

export type SimulationConfiguration = {
  population: PopulationSize;
  backend: BackendPreference;
  seed: number;
  speed: number;
  simulationTime?: number;
};

export type SimulationObservation = {
  gameStep: number;
  observation: number[];
  reward: number;
  simulationTime: number;
};

export type SimulationClient = {
  readonly state: SimulationClientState;
  configure(configuration: SimulationConfiguration): void;
  reset(options?: { seed?: number; simulationTime?: number }): void;
  observe(observation: SimulationObservation): void;
  pause(simulationTime?: number): void;
  resume(simulationTime?: number): void;
  reconnect(): void;
  close(): void;
  subscribe(listener: (message: ServerMessage) => void): () => void;
  subscribeState(listener: (state: SimulationClientState) => void): () => void;
};

type PendingObservation = { intentionId: string | null };

const OPEN = 1;
const sessionPattern = /^s-[a-z0-9]{8,64}$/;
const episodePattern = /^e-[a-z0-9]{8,64}$/;
let generatedId = 0;

function browserSocket(url: string): SimulationSocket {
  return new WebSocket(url) as unknown as SimulationSocket;
}

const browserTimers: SimulationTimerFactory = {
  setTimeout(callback, delayMs) { return setTimeout(callback, delayMs); },
  clearTimeout(handle) { clearTimeout(handle as ReturnType<typeof setTimeout>); },
};

function makeId(prefix: 's' | 'e'): string {
  const random = globalThis.crypto?.randomUUID?.().replaceAll('-', '')
    ?? `${Date.now().toString(36)}${(++generatedId).toString(36)}${Math.random().toString(36).slice(2)}`;
  return `${prefix}-${random.slice(0, 64).padEnd(8, '0')}`;
}

function requireId(value: string, pattern: RegExp, label: string): string {
  if (!pattern.test(value)) throw Error(`${label} factory returned an invalid protocol ID.`);
  return value;
}

function validUrl(url: string): void {
  if (!/^wss?:\/\//i.test(url)) throw Error('Simulation URL must use ws:// or wss://.');
}

/**
 * Connect one browser station to the v2 simulation service.
 *
 * This client deliberately does not resume a connection: a socket replacement
 * creates a fresh session and drops any ambiguous in-flight observation.
 */
export function createSimulationClient(
  url: string,
  options: SimulationClientOptions = {},
): SimulationClient {
  validUrl(url);
  const socketFactory = options.socketFactory ?? browserSocket;
  const timers = options.timerFactory ?? browserTimers;
  const reconnectDelayMs = options.reconnectDelayMs ?? 250;
  if (!Number.isFinite(reconnectDelayMs) || reconnectDelayMs < 0) {
    throw Error('Simulation reconnect delay must be non-negative.');
  }
  const uiBuild = options.uiBuild ?? 'browser';
  if (uiBuild.length < 1 || uiBuild.length > 256) {
    throw Error('Simulation UI build label must contain between 1 and 256 characters.');
  }
  const sessionIdFactory = options.sessionIdFactory ?? (() => makeId('s'));
  const episodeIdFactory = options.episodeIdFactory ?? (() => makeId('e'));
  const queue = new OutboundQueue<OutboundMessage>(options.maxQueue ?? 32);
  const messageListeners = new Set<(message: ServerMessage) => void>();
  const stateListeners = new Set<(state: SimulationClientState) => void>();
  let state: SimulationClientState = {
    phase: 'connecting',
    sessionId: '',
    episodeId: '',
    backend: null,
    error: null,
  };
  let socket: SimulationSocket | null = null;
  let reconnectTimer: unknown = null;
  let disposed = false;
  let nextOutboundSequence = 0;
  let lastInboundSequence: number | null = null;
  let pendingObservation: PendingObservation | null = null;
  let awaitingReset = false;
  let awaitingKeyframe = false;
  let configuration: SimulationConfiguration | null = null;
  const completedIntentions: string[] = [];
  const socketListeners = new Map<SimulationSocket, {
    open: () => void;
    message: (event: { data?: unknown }) => void;
    error: () => void;
    close: () => void;
  }>();

  const snapshot = (): SimulationClientState => ({ ...state });
  const publishState = () => {
    const current = snapshot();
    for (const listener of stateListeners) listener(current);
  };
  const setState = (next: Partial<SimulationClientState>) => {
    state = { ...state, ...next };
    publishState();
  };
  const enqueue = (message: OutboundMessage) => {
    queue.enqueue(message);
    flush();
  };
  const envelope = (type: string, simulationTime: number): OutboundMessage => ({
    type,
    version: 2,
    sessionId: state.sessionId,
    episodeId: state.episodeId,
    sequence: nextOutboundSequence++,
    simulationTime,
  });
  const flush = () => {
    if (!socket || socket.readyState !== OPEN) return;
    const messages = queue.drain();
    for (let index = 0; index < messages.length; index += 1) {
      try {
        socket.send(JSON.stringify(messages[index]));
      } catch (error) {
        for (const unsent of messages.slice(index)) queue.enqueue(unsent);
        setState({ phase: 'error', error: error instanceof Error ? error.message : String(error) });
        return;
      }
    }
  };
  const rememberCompletion = (intentionId: string) => {
    if (completedIntentions.includes(intentionId)) return;
    completedIntentions.push(intentionId);
    if (completedIntentions.length > 256) completedIntentions.shift();
  };
  const requestKeyframe = (simulationTime: number) => {
    if (awaitingKeyframe || disposed) return;
    awaitingKeyframe = true;
    enqueue(envelope('request_keyframe', simulationTime));
  };
  const publishMessage = (message: ServerMessage) => {
    for (const listener of messageListeners) listener(message);
  };
  const fail = (message: string) => setState({ phase: 'error', error: message });
  const receive = (raw: unknown) => {
    let message: ServerMessage;
    try {
      message = parseServerMessage(raw);
    } catch (error) {
      fail(error instanceof Error ? error.message : String(error));
      return;
    }
    if (message.sessionId !== state.sessionId || message.episodeId !== state.episodeId) return;
    if (message.type === 'ready') {
      if (lastInboundSequence !== null || message.sequence !== 0) {
        fail('Simulation ready frame must be the first server frame at sequence zero.');
        return;
      }
      lastInboundSequence = 0;
      setState({ phase: 'ready', backend: message.backend, error: null });
      publishMessage(message);
      return;
    }
    if (lastInboundSequence === null) {
      fail('Simulation server must send ready before other frames.');
      return;
    }
    if (message.sequence <= lastInboundSequence) {
      fail('Simulation server sequence must increase monotonically.');
      return;
    }
    const hasGap = message.sequence > lastInboundSequence + 1;
    lastInboundSequence = message.sequence;
    if (message.type === 'neural_delta' && hasGap) requestKeyframe(message.simulationTime);
    if (message.type === 'neural_keyframe') awaitingKeyframe = false;

    switch (message.type) {
      case 'intention':
        if (!pendingObservation) {
          fail('Simulation sent an intention without a pending observation.');
          return;
        }
        if (pendingObservation.intentionId && pendingObservation.intentionId !== message.intentionId) {
          fail('Simulation sent overlapping intentions.');
          return;
        }
        pendingObservation.intentionId = message.intentionId;
        setState({ phase: 'acting', error: null });
        break;
      case 'action_result':
        if (!pendingObservation) {
          if (completedIntentions.includes(message.intentionId)) return;
          fail('Simulation action result does not match a pending intention.');
          return;
        }
        if (pendingObservation.intentionId !== message.intentionId) {
          fail('Simulation action result does not match the pending intention.');
          return;
        }
        rememberCompletion(message.intentionId);
        pendingObservation = null;
        setState({ phase: 'ready', error: null });
        break;
      case 'reset_complete':
        awaitingReset = false;
        pendingObservation = null;
        completedIntentions.length = 0;
        setState({ phase: 'ready', error: null });
        break;
      case 'paused':
        setState({ phase: 'paused', error: null });
        break;
      case 'error':
        fail(`${message.code}: ${message.message}`);
        break;
      default:
        break;
    }
    publishMessage(message);
  };
  const disconnect = (current: SimulationSocket) => {
    const listeners = socketListeners.get(current);
    if (!listeners) return;
    current.removeEventListener('open', listeners.open);
    current.removeEventListener('message', listeners.message);
    current.removeEventListener('error', listeners.error);
    current.removeEventListener('close', listeners.close);
    socketListeners.delete(current);
  };
  const scheduleReconnect = () => {
    if (disposed || reconnectTimer !== null) return;
    reconnectTimer = timers.setTimeout(() => {
      reconnectTimer = null;
      beginConnection();
    }, reconnectDelayMs);
  };
  const attachSocket = (current: SimulationSocket) => {
    const listeners = {
      open: () => {
        if (socket === current) flush();
      },
      message: (event: { data?: unknown }) => {
        if (socket === current) receive(event.data);
      },
      error: () => {
        if (socket === current) fail('Simulation connection failed.');
      },
      close: () => {
        if (disposed || socket !== current) return;
        socket = null;
        disconnect(current);
        setState({ phase: 'connecting', backend: null, error: null });
        scheduleReconnect();
      },
    };
    socketListeners.set(current, listeners);
    current.addEventListener('open', listeners.open);
    current.addEventListener('message', listeners.message);
    current.addEventListener('error', listeners.error);
    current.addEventListener('close', listeners.close);
  };
  const beginConnection = () => {
    if (disposed) return;
    if (socket) disconnect(socket);
    queue.clear();
    pendingObservation = null;
    awaitingReset = false;
    awaitingKeyframe = false;
    completedIntentions.length = 0;
    nextOutboundSequence = 0;
    lastInboundSequence = null;
    const sessionId = requireId(sessionIdFactory(), sessionPattern, 'Session ID');
    const episodeId = requireId(episodeIdFactory(), episodePattern, 'Episode ID');
    state = { phase: 'connecting', sessionId, episodeId, backend: null, error: null };
    publishState();
    enqueue({ ...envelope('hello', 0), supportedVersions: [2], uiBuild });
    if (configuration) {
      enqueue({ ...envelope('configure', configuration.simulationTime ?? 0), ...configuration });
    }
    const nextSocket = socketFactory(url);
    socket = nextSocket;
    attachSocket(nextSocket);
    flush();
  };

  beginConnection();

  return {
    get state() { return snapshot(); },
    configure(nextConfiguration) {
      if (disposed) throw Error('Simulation client is closed.');
      if (configuration) throw Error('Simulation client is already configured for this session.');
      configuration = { ...nextConfiguration };
      enqueue({ ...envelope('configure', nextConfiguration.simulationTime ?? 0), ...nextConfiguration });
    },
    reset(resetOptions = {}) {
      if (disposed) throw Error('Simulation client is closed.');
      if (!configuration) throw Error('Simulation client must be configured before reset.');
      queue.removeType('observation');
      pendingObservation = null;
      awaitingReset = true;
      const episodeId = requireId(episodeIdFactory(), episodePattern, 'Episode ID');
      setState({ episodeId });
      enqueue({ ...envelope('reset', resetOptions.simulationTime ?? 0), seed: resetOptions.seed });
    },
    observe(nextObservation) {
      if (disposed) throw Error('Simulation client is closed.');
      if (pendingObservation) throw Error('Simulation client already has a pending observation.');
      if (state.phase !== 'ready' || awaitingReset) throw Error('Simulation client is not ready for an observation.');
      pendingObservation = { intentionId: null };
      setState({ phase: 'acting', error: null });
      enqueue({ ...envelope('observation', nextObservation.simulationTime), ...nextObservation });
    },
    pause(simulationTime = 0) {
      if (disposed) throw Error('Simulation client is closed.');
      enqueue(envelope('pause', simulationTime));
    },
    resume(simulationTime = 0) {
      if (disposed) throw Error('Simulation client is closed.');
      enqueue(envelope('resume', simulationTime));
    },
    reconnect() {
      if (disposed) throw Error('Simulation client is closed.');
      if (reconnectTimer !== null) {
        timers.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (socket) {
        const current = socket;
        disconnect(current);
        socket = null;
        current.close();
      }
      beginConnection();
    },
    close() {
      if (disposed) return;
      disposed = true;
      if (reconnectTimer !== null) timers.clearTimeout(reconnectTimer);
      reconnectTimer = null;
      if (socket) {
        const current = socket;
        socket = null;
        disconnect(current);
        current.close();
      }
      queue.clear();
      pendingObservation = null;
      setState({ phase: 'closed', backend: null, error: null });
      messageListeners.clear();
      stateListeners.clear();
    },
    subscribe(listener) {
      messageListeners.add(listener);
      return () => messageListeners.delete(listener);
    },
    subscribeState(listener) {
      stateListeners.add(listener);
      listener(snapshot());
      return () => stateListeners.delete(listener);
    },
  };
}
