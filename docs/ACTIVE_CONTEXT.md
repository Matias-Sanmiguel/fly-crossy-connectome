# Active project context

Last updated: 2026-09-27

## Authoritative state

- Game environment: v11.
- World layout/generation: v6, intentionally preserved.
- Observation: v4, 517 float32 values.
- Reward: v5.
- Decision interval: 0.2 seconds.
- The TypeScript browser and Python environment share generated parity fixtures.

Environment v11 changed collision timing so a fly that leaves a road before a
vehicle reaches its old cell survives; waiting in that cell or entering a swept
destination remains hazardous. Any gameplay, observation, reward, or timing
change requires a new environment version and refreshed parity evidence.

## Operational controller

The only released autonomous controller is the verified 80-neuron fixed-graph
checkpoint under `release/eval-v6/`. It was trained for environment v6 and the
legacy 370-value ObservationV1. Both browser and Python runtimes now adapt the
current ObservationV4 to that exact legacy input contract.

This is a compatibility controller, not a controller retrained for v11. Its
activity is simulated model state mapped to measured MaleCNS soma body IDs; it
is not biological recording data and must not be described as such.

The 1,000-neuron graph and V5 sensory/core experiments are research artifacts,
not released playable controllers. Independent 1,000, 5,000, 20,000, and
124,289-neuron modes remain unfinished and disabled in the biomechanical UI.

## Runtime chain

Normal browser mode autoplays locally with the released 80-neuron graph and
updates the brain view on every decision.

Biomechanical mode preserves this authority chain:

`controller -> FlyBody motor -> physical key contact -> action_result -> game`

Directional intentions must never move the browser game directly. The Python
bridge emits actual controller neural activations and authoritative action
results. World snapshots, native contact frames, and runtime metrics remain the
next telemetry milestone.

The CPU Docker service packages the verified graph, checkpoint, calibration,
and manifests. FlyGym's pinned full-size meshes are downloaded once into the
named `fly-data` volume. CPU is the default; CUDA remains optional.

## Repository hygiene

- Generated datasets, training runs, pytest scratch trees, caches, and virtual
  environments are ignored and must not be committed.
- Released artifacts and their hashes are immutable evidence. Do not rewrite a
  historical release merely to match a different local Python environment.
- Do not add automated-agent coauthor trailers.

## Required verification

Before merging gameplay/runtime work:

```sh
npm test
npm run check:assets
npm run build
cd python && python -m pytest tests -q
docker compose build simulation web
docker compose up --wait
bash scripts/docker-smoke.sh
docker compose down
```

Run Python tests in the locked Python 3.12 CPU container for release-quality
evidence. The separate Python 3.14 release-environment test only applies to the
historical environment recorded in `release-environment-linux-x86_64.json`.

## Next meaningful work

1. Stream physical snapshots, contacts, motor phase, force/travel, and metrics.
2. Train and evaluate a native v11 controller instead of relying on the v6
   compatibility adapter.
3. Add population modes only when each has its own checkpoint, evaluation, and
   provenance manifest.
4. Exercise and document the CUDA image on an NVIDIA host.
