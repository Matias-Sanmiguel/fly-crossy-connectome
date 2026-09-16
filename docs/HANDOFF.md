# Engineering handoff

This repository is an **operational development snapshot**, not the finished
five-controller biomechanical product. The browser demo, unified native
FlyBody/keyboard world, and an opt-in protocol-v2 development bridge run today.
The remaining milestone is complete native telemetry plus independently
trained larger neural controllers.

Fresh snapshot verification on 2026-09-15:

- Browser tests: 123 passed.
- Python tests: 350 passed, 1 historical release-environment test skipped.
- Anatomy/connectome asset hashes: verified.
- Kenney GLB byte sizes and hashes: verified.
- TypeScript and production web build: passed (with Vite's non-blocking large
  bundle advisory).
- Default CPU Docker build, service health, HTTP/WebSocket proxy smoke, and
  clean shutdown: passed.
- Browser QA: default and biomechanical desktop entries, 360×800 stacking, and
  a forced GLB 404 with visible geometric fallback were exercised.

## What works now

### Browser application

- Deterministic Crossy Road/Frogger-style environment with roads, rails,
  rivers, hazards, rewards, terminal states, seeded replay, and manual input.
- Autonomous bundled reduced-connectome controller.
- Always-visible MaleCNS soma visualization with real-time **simulated model
  activity**. It is explicitly not a biological recording.
- Curated Kenney CC0 roads, track, vehicles, and scenery with a native geometry
  fallback that does not affect simulation.
- Shared dark laboratory shell for `/` and `/?biomechanics=1`, including
  responsive mobile stacking and explicit runtime/asset status.
- Human, scripted, dense-policy, and remote-controller boundaries.
- Versioned WebSocket protocol, reconnect/resume plumbing, payload bounds, and
  CPU/GPU Docker service definitions.

Run the currently operational demo with:

```sh
npm ci
npm run dev
```

### Native biomechanics and bridge

The following layers are implemented and individually reviewed:

- FlyGym 2.1.0 / MuJoCo 3.9 FlyBody adapter with exactly 59 actions.
- Immutable pinned FlyBody model and 87-file mesh inventory validation.
- Six physical spring-loaded keys: W, A, S, D, Space-left, and Space-right.
- Contact gate requiring intended key, intended tarsus, travel, force, 20 ms
  debounce, release-before-rearm, and exactly one terminal outcome.
- Shared 59-signal closed-loop motor controller and deterministic calibrated
  front/middle-leg targets.
- Fail-closed recovery: invalid, wrong, ambiguous, unsafe, or timed-out contact
  cannot become a directional game action.
- `BiomechanicalWorld` composes FlyBody and all six keys into one `MjModel`,
  advances the fixed-rate dynamics, caps snapshots at 30 Hz, and produces one
  terminal result per intention.
- `dev_server.py` loads the committed 80-neuron release checkpoint plus the
  verified runtime manifest, creates the world, and connects directional
  decisions to the physical gate through the protocol-v2 server.

Important physics invariant: MuJoCo stays at its native `0.0001 s` solver
timestep. One 500 Hz outer tick must execute 20 native substeps; the motor is
updated at 100 Hz, every five outer ticks.

## What is not finished

1. **Four larger neural controllers.** The requested 1,000, 5,000,
   20,000, and 124,289-neuron modes, their independent checkpoints, training,
   evaluation, manifests, and selectable UI state remain to be built. The
   current 80-cell controller is verified but is not the full five-mode system.
2. **Native telemetry emission.** The development bridge sends intentions and
   authoritative action results, but it does not yet publish world snapshots,
   contact frames, motor phase/force/travel, metrics, or non-empty native neural
   keyframes/deltas. The biomechanical UI therefore labels those fields as
   awaiting telemetry rather than deriving them.
3. **Container integration of the physical runtime.** Default Compose
   deliberately starts `fly_crossy.server:app` without an artifact registry or
   world factory. It verifies health and proxy behavior but is not the full
   physical demo. Package the verified manifests, checkpoint, calibration,
   FlyBody model, and cached assets before switching the container entrypoint.
4. **GPU release evidence.** CPU fallback is the default and passes. The CUDA
   image/runtime has not been exercised on this host.

## Recommended implementation order

1. Extend the server bridge to stream world/contact/metrics/neural data from the
   authoritative runtime and add end-to-end browser tests.
2. Package the development artifacts and physical world into the CPU container
   without weakening manifest/hash validation.
3. Complete the multiscale neural-controller plan, including real artifacts and
   independent training/evaluation for all five sizes.
4. Enable each controller option only after its artifacts and evidence exist.
5. Run the full CPU Docker path, then optional GPU smoke, browser QA, license
   audit, and release checklist.

The authoritative design and task breakdown are committed under:

- `docs/superpowers/specs/2026-09-14-multiscale-biomechanical-keyboard-design.md`
- `docs/superpowers/plans/2026-09-14-multiscale-runtime-protocol.md`
- `docs/superpowers/plans/2026-09-14-biomechanical-keyboard.md`
- `docs/superpowers/plans/2026-09-14-multiscale-neural-controllers.md`
- `docs/superpowers/plans/2026-09-14-browser-station-and-release.md`

Implementation and review evidence is under `.superpowers/sdd/`. Start with
the two `progress.md` files; they contain decisions that must survive the next
implementation pass.

## Environment notes

- Frontend: Node.js 22.18+.
- Container/native biomechanics target: Python 3.12, FlyGym 2.1.0, MuJoCo 3.9.
- The historical evaluation environment under `release/eval-v1/` is a separate
  Python 3.14 closure and must not be silently rewritten to satisfy FlyGym.
- FlyGym assets should be cached inside the persistent `FLY_DATA_ROOT` Docker
  volume via `FLYGYM_ASSET_CACHE_DIR`.
- CPU is the required default. GPU is optional and must fall back cleanly.
- Do not add automated-agent coauthor trailers to commits.

## Definition of the next meaningful milestone

The next meaningful milestone is visible end-to-end evidence: run the committed
80-neuron development controller through the physical world, stream the real
motor/contact/snapshot/neural state to `/?biomechanics=1`, and show it without
inventing values. Then package that same path into the default CPU container.
