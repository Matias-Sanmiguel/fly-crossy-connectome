import {
  MAX_SEQUENCE,
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
const populations = new Set<PopulationSize>([80, 1000, 5000, 20000, 124289]);
const backends = new Set<BackendPreference>(['auto', 'cpu', 'gpu', 'gpu-strict']);
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

function finite(value: unknown, label: string, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
    throw Error(`Simulation ${label} must be a finite number between ${min} and ${max}.`);
  }
  return value;
}

function integer(value: unknown, label: string, min: number, max: number): number {
  const number = finite(value, label, min, max);
  if (!Number.isSafeInteger(number)) throw Error(`Simulation ${label} must be a safe integer.`);
  return number;
}

function simulationTime(value: unknown): number {
  return finite(value, 'simulation time', 0, 86_400);
}

function validateConfiguration(value: SimulationConfiguration): SimulationConfiguration {
  if (!populations.has(value.population)) throw Error('Simulation population is invalid.');
  if (!backends.has(value.backend)) throw Error('Simulation backend is invalid.');
  const configuration: SimulationConfiguration = {
    population: value.population,
    backend: value.backend,
    seed: integer(value.seed, 'seed', 0, 2 ** 32 - 1),
    speed: finite(value.speed, 'speed', Number.MIN_VALUE, 100),
  };
  if (value.simulationTime !== undefined) configuration.simulationTime = simulationTime(value.simulationTime);
  return configuration;
}

function validateObservation(value: SimulationObservation): SimulationObservation {
  if (!Array.isArray(value.observation) || value.observation.length !== 370) {
    throw Error('Simulation observation must contain exactly 370 values.');
  }
  return {
    gameStep: integer(value.gameStep, 'game step', 0, MAX_SEQUENCE),
    observation: value.observation.map((entry) => finite(entry, 'observation value', -1_000_000, 1_000_000)),
    reward: finite(value.reward, 'reward', -1_000_000, 1_000_000),
    simulationTime: simulationTime(value.simulationTime),
  };
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
  let connectionGeneration = 0;
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
  const enqueue = (message: OutboundMessage): boolean => {
    try {
      const admitted = queue.enqueue(message);
      if (admitted) flush();
      return admitted;
    } catch (error) {
      fail(error instanceof Error ? error.message : String(error));
      throw error;
    }
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
  const requestKeyframe = (simulationTime: number): boolean => {
    if (awaitingKeyframe) return true;
    if (disposed) return false;
    awaitingKeyframe = true;
    try {
      enqueue(envelope('request_keyframe', simulationTime));
      return true;
    } catch {
      // `enqueue` has already moved the client into a controlled error state.
      return false;
    }
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
    if (hasGap && !requestKeyframe(message.simulationTime)) return;
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
        if (!awaitingReset) {
          fail('Simulation sent an unsolicited reset_complete frame.');
          return;
        }
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
  const scheduleReconnect = (generation: number) => {
    if (disposed || reconnectTimer !== null || generation !== connectionGeneration) return;
    const timer = timers.setTimeout(() => {
      if (reconnectTimer === timer) reconnectTimer = null;
      if (generation !== connectionGeneration || disposed) return;
      beginConnection();
    }, reconnectDelayMs);
    reconnectTimer = timer;
  };
  const attachSocket = (current: SimulationSocket, generation: number) => {
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
        if (disposed || socket !== current || generation !== connectionGeneration) return;
        socket = null;
        disconnect(current);
        setState({ phase: 'connecting', backend: null, error: null });
        if (generation !== connectionGeneration || socket !== null) return;
        scheduleReconnect(generation);
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
    const generation = ++connectionGeneration;
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
    if (generation !== connectionGeneration) return;
    enqueue({ ...envelope('hello', 0), supportedVersions: [2], uiBuild });
    if (configuration) {
      try {
        enqueue({ ...envelope('configure', configuration.simulationTime ?? 0), ...configuration });
      } catch {
        return;
      }
    }
    const nextSocket = socketFactory(url);
    if (generation !== connectionGeneration) return;
    socket = nextSocket;
    attachSocket(nextSocket, generation);
    flush();
  };

  beginConnection();

  return {
    get state() { return snapshot(); },
    configure(nextConfiguration) {
      if (disposed) throw Error('Simulation client is closed.');
      if (configuration) throw Error('Simulation client is already configured for this session.');
      const validated = validateConfiguration(nextConfiguration);
      if (!enqueue({ ...envelope('configure', validated.simulationTime ?? 0), ...validated })) {
        throw Error('Simulation configuration was backpressured.');
      }
      configuration = validated;
    },
    reset(resetOptions = {}) {
      if (disposed) throw Error('Simulation client is closed.');
      if (!configuration) throw Error('Simulation client must be configured before reset.');
      queue.removeType('observation');
      pendingObservation = null;
      awaitingReset = true;
      const episodeId = requireId(episodeIdFactory(), episodePattern, 'Episode ID');
      setState({ episodeId });
      const reset = envelope('reset', simulationTime(resetOptions.simulationTime ?? 0));
      if (resetOptions.seed !== undefined) reset.seed = integer(resetOptions.seed, 'seed', 0, 2 ** 32 - 1);
      enqueue(reset);
    },
    observe(nextObservation) {
      if (disposed) throw Error('Simulation client is closed.');
      if (pendingObservation) throw Error('Simulation client already has a pending observation.');
      if (state.phase !== 'ready' || awaitingReset) throw Error('Simulation client is not ready for an observation.');
      const validated = validateObservation(nextObservation);
      if (!enqueue({ ...envelope('observation', validated.simulationTime), ...validated })) {
        throw Error('Simulation observation was backpressured.');
      }
      pendingObservation = { intentionId: null };
      setState({ phase: 'acting', error: null });
    },
    pause(simulationTime = 0) {
      if (disposed) throw Error('Simulation client is closed.');
      enqueue(envelope('pause', finite(simulationTime, 'simulation time', 0, 86_400)));
    },
    resume(simulationTime = 0) {
      if (disposed) throw Error('Simulation client is closed.');
      enqueue(envelope('resume', finite(simulationTime, 'simulation time', 0, 86_400)));
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
