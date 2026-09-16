import {
  useEffect,
  useRef,
  useState,
} from 'react';

import {
  encodeObservation,
} from '../game/model.ts';

import {
  observe,
} from '../game/observation.ts';

import {
  createSimulationClient,
  type NeuralActivityState,
  type SimulationClient,
  type SimulationClientState,
  type SimulationObservation,
} from '../simulation/client.ts';

import {
  type BackendPreference,
  type PopulationSize,
  type ServerMessage,
  type Snapshot,
} from '../simulation/protocol.ts';

import {
  createStation,
  reduceStation,
  type StationState,
} from '../simulation/station.ts';


export type UseSimulationStationOptions = {
  enabled?: boolean;
  seed: string;
  url: string;

  population?: PopulationSize;
  backendPreference?: BackendPreference;
  speed?: number;
  autoResetDelayMs?: number;
};


const CLOSED_TRANSPORT: SimulationClientState = {
  phase: 'closed',
  sessionId: '',
  episodeId: '',
  backend: null,
  error: null,
};


export function seedToUint32(seed: string): number {
  let hash = 0x811c9dc5;

  for (const character of seed) {
    hash ^= character.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 0x01000193);
  }

  return hash >>> 0;
}


export function buildSimulationObservation(
  station: StationState,
): SimulationObservation {
  return {
    gameStep: station.game.step,

    observation: encodeObservation(
      observe(station.game),
    ),

    reward: station.reward,

    simulationTime: station.game.time,
  };
}


function isStationMessage(
  message: ServerMessage,
): boolean {
  return (
    message.type === 'intention'
    || message.type === 'contact'
    || message.type === 'action_result'
    || message.type === 'reset_complete'
  );
}


export function useSimulationStation({
  enabled = true,
  seed,
  url,
  population = 80,
  backendPreference = 'cpu',
  speed = 1,
    autoResetDelayMs,
}: UseSimulationStationOptions) {
  const [station, setStation] = useState<StationState>(
    () => createStation(seed),
  );

  const [transport, setTransport] =
    useState<SimulationClientState>(
      CLOSED_TRANSPORT,
    );

  const [activity, setActivity] =
    useState<NeuralActivityState | null>(null);

  const [snapshot, setSnapshot] =
    useState<Snapshot | null>(null);

  const clientRef =
    useRef<SimulationClient | null>(null);

  const lastObservedStep =
    useRef<number | null>(null);

  const episodeIndex =
    useRef(0);

  const pendingResetSeed =
    useRef<string | null>(null);


  /*
   * Own exactly one protocol client for one station configuration.
   *
   * Changing population/backend/seed creates a completely new
   * protocol session so stale physical/neural state cannot leak
   * into the next run.
   */
  useEffect(() => {
    setStation(createStation(seed));
    setActivity(null);
    setSnapshot(null);
    lastObservedStep.current = null;
    episodeIndex.current = 0;
    pendingResetSeed.current = null;

    if (!enabled) {
      clientRef.current = null;
      setTransport(CLOSED_TRANSPORT);
      return;
    }

    const client = createSimulationClient(url);

    clientRef.current = client;

    const unsubscribeState =
      client.subscribeState((next) => {
        setTransport(next);
      });

    const unsubscribeActivity =
      client.subscribeActivity((next) => {
        setActivity(next);
      });

    const unsubscribeMessages =
      client.subscribe((message) => {
        if (message.type === 'snapshot') {
          setSnapshot(message);
        }

        if (isStationMessage(message)) {
        setStation((current) => {
            if (
            message.type === 'reset_complete'
            && pendingResetSeed.current !== null
            ) {
            const nextSeed =
                pendingResetSeed.current;

            pendingResetSeed.current = null;

            return createStation(nextSeed);
            }

            return reduceStation(
            current,
            message as Parameters<
                typeof reduceStation
            >[1],
            );
        });
        }
      });

    client.configure({
      population,
      backend: backendPreference,
      seed: seedToUint32(seed),
      speed,
      simulationTime: 0,
    });

    return () => {
      unsubscribeMessages();
      unsubscribeActivity();
      unsubscribeState();

      if (clientRef.current === client) {
        clientRef.current = null;
      }

      client.close();
    };
  }, [
    backendPreference,
    enabled,
    population,
    seed,
    speed,
    url,
  ]);


  /*
   * The browser sends a new observation only after both authorities
   * agree that the previous action is finished:
   *
   *   transport = ready
   *   station   = ready
   *
   * Merely receiving an Intention never changes the game.
   */
  useEffect(() => {
    if (!enabled) return;

    const client = clientRef.current;

    if (!client) return;
    if (transport.phase !== 'ready') return;
    if (station.phase !== 'ready') return;
    if (station.game.terminal !== null) return;

    const step = station.game.step;

    if (lastObservedStep.current === step) {
      return;
    }

    const observation =
        buildSimulationObservation(station);

    try {
        client.observe(observation);

        // Only mark the step as submitted after the protocol
        // client actually accepted the observation.
        lastObservedStep.current = step;
    } catch (error) {
        console.error(
            '[biomechanics] observation submission failed',
            {
                step,
                transport: transport.phase,
                client: client.state.phase,
                error,
            },
        );
    }
  }, [
    enabled,
    station,
    transport.phase,
  ]);
    /*
   * Automatically start a fresh episode after a terminal outcome.
   *
   * The protocol session stays alive; only the episode, game world,
   * recurrent controller state, and biomechanical world are reset.
   */
  useEffect(() => {
    if (!enabled) return;

    if (autoResetDelayMs === undefined) {
      return;
    }

    if (station.game.terminal === null) {
      return;
    }

    if (station.phase !== 'paused') {
      return;
    }

    if (transport.phase !== 'ready') {
      return;
    }

    const client = clientRef.current;

    if (!client) {
      return;
    }

    const timer = globalThis.setTimeout(() => {
      if (clientRef.current !== client) {
        return;
      }

      const nextIndex =
        episodeIndex.current + 1;

      episodeIndex.current = nextIndex;

      const nextSeed =
        `${seed}:episode:${nextIndex}`;

      pendingResetSeed.current =
        nextSeed;

      lastObservedStep.current = null;

      try {
        client.reset({
          seed: seedToUint32(nextSeed),
          simulationTime: 0,
        });
      } catch (error) {
        pendingResetSeed.current = null;

        console.error(
          '[biomechanics] automatic episode reset failed',
          error,
        );
      }
    }, autoResetDelayMs);

    return () => {
      globalThis.clearTimeout(timer);
    };
  }, [
    autoResetDelayMs,
    enabled,
    seed,
    station.game.terminal,
    station.phase,
    transport.phase,
  ]);


  return {
    station,
    transport,
    activity,
    snapshot,

    connected:
      transport.phase !== 'connecting'
      && transport.phase !== 'closed'
      && transport.phase !== 'error',

    error:
      station.error
      ?? transport.error,
  };
}

