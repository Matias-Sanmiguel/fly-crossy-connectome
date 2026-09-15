# Engineering handoff

This repository is an **operational development snapshot**, not the finished
biomechanical product. The browser demo runs today; the native biomechanics
foundation is implemented and tested in isolation; the bridge between them is
the main unfinished milestone.

Fresh snapshot verification on 2026-09-15:

- Browser tests: 102 passed.
- Python tests: 321 passed, 1 historical release-environment test skipped.
- Anatomy/connectome asset hashes: verified.
- TypeScript and production web build: passed (with Vite's non-blocking large
  bundle advisory).
- Default CPU Docker build, service health, HTTP/WebSocket proxy smoke, and
  clean shutdown: passed.

## What works now

### Browser application

- Deterministic Crossy Road/Frogger-style environment with roads, rails,
  rivers, hazards, rewards, terminal states, seeded replay, and manual input.
- Autonomous bundled reduced-connectome controller.
- Always-visible MaleCNS soma visualization with real-time **simulated model
  activity**. It is explicitly not a biological recording.
- Human, scripted, dense-policy, and remote-controller boundaries.
- Versioned WebSocket protocol, reconnect/resume plumbing, payload bounds, and
  CPU/GPU Docker service definitions.

Run the currently operational demo with:

```sh
npm ci
npm run dev
```

### Native biomechanics foundation

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

Important physics invariant: MuJoCo stays at its native `0.0001 s` solver
timestep. One 500 Hz outer tick must execute 20 native substeps; the motor is
updated at 100 Hz, every five outer ticks.

## What is not finished

1. **Unified biomechanical world (highest priority).** There is no committed
   `python/fly_crossy/biomechanics/world.py` yet. Compose FlyBody and the six
   keys into one `MjModel`, step the real dynamics, sample one complete six-key
   contact frame per outer tick, and emit a direction only through
   `is_directional_confirmation`.
2. **Server/session bridge.** Intention, physical contact, snapshots, recovery,
   and the single terminal `action_result` still need to be wired into
   `session.py` and `server.py`.
3. **Five independent neural controllers.** The requested 80, 1,000, 5,000,
   20,000, and 124,289-neuron modes, their independent checkpoints, training,
   evaluation, manifests, and UI selector remain to be built. The browser's
   current 80-cell reduced controller is not that full five-mode system.
4. **Real-time native telemetry in the browser.** The current brain panel shows
   browser-controller model values. Native controller activations, motor phase,
   leg/key contact, travel, force, debounce, and failures still need streaming.
5. **Release integration.** The current CPU web/API Docker smoke passes, but an
   end-to-end smoke of the unfinished biomechanical world with FlyGym assets,
   CPU fallback, optional NVIDIA GPU execution, accessibility/browser QA, and
   final release evidence remain.

## Known runtime issues to fix before neural streaming

The latest runtime review left three browser-side hardening items:

- Revalidate connection generation between every message/activity listener;
  one listener must not be able to make later listeners process stale events.
- Accept a revision-0 keyframe after reconnect/reset even if the previous
  generation observed a higher revision, and publish an empty/reset activity
  state while awaiting the new keyframe.
- Reject `resume()` unless the session is actually paused.

These do not invalidate the isolated physics components, but must be fixed
before relying on streamed native neural state.

## Recommended implementation order

1. Fix the three runtime issues above and add regression tests.
2. Execute Task 5 in the biomechanical plan: unified world plus server bridge.
3. Complete the multiscale neural-controller plan, including real artifacts and
   independent training/evaluation for all five sizes.
4. Complete the browser station: mode selector, native neural snapshots,
   physical keyboard/leg telemetry, and honest status labels.
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

For W, A, S, and D, a real mapped leg reaches a real key in MuJoCo, creates an
actual key-geom/tarsus contact pair, crosses travel and force thresholds for at
least 20 ms, and produces exactly one matching directional result. Disabling
contacts must produce exactly one failed `wait`. No test may inject a successful
contact outcome or manually move a key as a substitute for causal dynamics.
