# Prompt to continue the project

Copy the prompt below into the coding agent that will continue this repository.

---

You are continuing `fly-crossy-connectome`, an incomplete but operational
development snapshot. Do not restart or replace the existing architecture.

First read, in this order:

1. `README.md`
2. `docs/HANDOFF.md`
3. `docs/superpowers/specs/2026-09-14-multiscale-biomechanical-keyboard-design.md`
4. all four plans under `docs/superpowers/plans/` whose names begin with
   `2026-09-14-multiscale`, `2026-09-14-biomechanical`, or
   `2026-09-14-browser`
5. `.superpowers/sdd/2026-09-14-multiscale-runtime-protocol/progress.md`
6. `.superpowers/sdd/2026-09-14-biomechanical-keyboard/progress.md`

Goal: finish a 3D Crossy Road/Frogger browser experience played autonomously by
a simulated fly. The browser must always show the MaleCNS soma atlas with
honestly labeled real-time simulated activations. The fly must operate a
physical W/A/S/D/Space keyboard with its legs in one native MuJoCo world.

Non-negotiable requirements:

- Five selectable and independently trained MaleCNS-derived controllers:
  80, 1,000, 5,000, 20,000, and 124,289 neurons.
- Every controller outputs the exact same 59-signal motor interface.
- Front-left presses W, front-right S, middle-left A, middle-right D; hind legs
  may demonstrate the two Space zones, but Space never confirms a direction.
- The game receives a direction only after MuJoCo confirms intended key,
  intended tarsus, real key-geom/tarsus contact, key travel, force bounds, and
  20 ms debounce. Wrong, ambiguous, unsafe, cancelled, or timed-out attempts
  emit a failed `wait`, exactly once.
- No animation, commanded target, manually assigned key qpos, or injected test
  outcome may stand in for physical contact.
- Preserve the exact 59-actuator order already validated by `FlyBodyModel`.
- Preserve the MuJoCo `0.0001 s` native timestep. Execute 20 native substeps per
  500 Hz outer tick; update motor targets every five outer ticks (100 Hz); emit
  browser snapshots at no more than 30 Hz.
- CPU Docker path is the default. NVIDIA GPU support is optional and must have a
  CPU fallback. Put `FLYGYM_ASSET_CACHE_DIR` under persistent `FLY_DATA_ROOT`.
- Neural values are simulated model state over measured soma locations, never
  biological recordings. Do not invent connectome edges, neurons, performance,
  training results, or provenance.
- Preserve attribution, third-party notices, custom template license, MaleCNS
  CC BY 4.0 notice, and FlyBody Apache-2.0 notice.
- Never add an automated-agent coauthor trailer.

Work in this order:

1. Fix the three remaining runtime issues listed in `docs/HANDOFF.md` with
   regression tests.
2. Implement biomechanical Task 5. Compose a fresh FlyBody `MjSpec` with the
   keyboard into one model, re-resolve all IDs after compilation, validate the
   unchanged 59-control order, and implement `BiomechanicalWorld`.
3. Add causal integration tests for all four directions, disabled-contact
   failure, Space non-directional behavior, scheduler counts, <=30 Hz snapshots,
   exactly one terminal result, multiple actions, atomic reset, and zero MuJoCo
   warnings. If causal contact fails, fix physics/calibration; do not fake it.
4. Wire the world atomically into session/server v2. Recovery must finish before
   another observation is accepted.
5. Implement and train the five independent neural controllers exactly as the
   neural plan specifies, with deterministic manifests, validation, evaluation,
   and honest UI labels.
6. Finish native telemetry and browser UI, then run full CPU Docker and browser
   QA. Run optional GPU smoke only where NVIDIA runtime is actually available.

At every checkpoint, distinguish among: implemented and freshly verified;
implemented but not fully reviewed; designed only; and blocked by unavailable
hardware/assets. Do not claim completion from old logs. Before pushing, run the
fresh full Node tests/build, Python tests in the declared environment, asset
checks, and CPU Docker smoke. Update `docs/HANDOFF.md` as work becomes real.

---
