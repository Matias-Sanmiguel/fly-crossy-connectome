# Engineering handoff

Current handoff: 2026-10-15. The local installation has a human game with a
persistent ranking (`?play=1`) and a Full MaleCNS expo replay (`?expo=1`). Start
with `npm ci` and `npm run dev`, or build and use `npm run preview`.

The teammate keeps the full-neuron training artifacts on their own machine
because of size. They are intentionally external to Git. See
[TRAINING_ARTIFACTS.md](TRAINING_ARTIFACTS.md) for what is needed to regenerate
the expo. No checkpoint is required to display the committed exports or play
the human game. Their absence is not evidence that training was never done.

Older V2–V6 notes and the original biomechanics continuation prompt are
preserved under `docs/experiments/legacy/`. Cached-frontier scripts retain all
versions under `scripts/experiments/cached_frontier/`. Local `runs/` files are
ignored, with historical sizes, hashes and reported results retained in
`docs/experiments/run-index.json`; immutable `release/` evidence is unchanged.

## Working now

- Human game with server-created seeds, registration, server-replayed scoring
  and persistent SQLite top-10 ranking. See [VISITOR_MODE.md](VISITOR_MODE.md).
- Full MaleCNS exported action/activity replay on its original seed, checked
  against hashes and measured visible soma IDs. See
  [expo-neural-replay.md](expo-neural-replay.md).
- Deterministic environment v11 in TypeScript and Python, with shared parity
  fixtures for world generation, observations, collisions, rewards, and replay.
- Isometric browser game using curated Kenney CC0 GLB assets with procedural
  fallbacks that cannot affect simulation state.
- Continuous autoplay from the released 80-neuron environment-v6 checkpoint
  through an explicit ObservationV4-to-V1 compatibility adapter.
- Always-visible measured MaleCNS soma atlas driven by real-time simulated
  controller activations keyed by verified body IDs.
- Unified FlyGym/MuJoCo FlyBody and six-key physical keyboard.
- Fail-closed physical authority: directional game movement occurs only after
  the intended tarsus confirms the intended key contact.
- Protocol-v2 Python bridge with real controller neural keyframes, CPU/GPU
  selection, CPU fallback, artifact hash verification, and bounded messages.
- The simulation Docker recipe packages verified runtime artifacts. The current
  web image needs an integration fix before the full Compose stack can be
  presented as operational; use Vite locally for the installation.

## Scientific boundary

The 80-neuron checkpoint is the only released playable connectome controller.
It was trained on environment v6 and is explicitly labeled as compatibility
mode on v11. Neural values are simulated internal activations over measured soma
positions, not measurements from a living fly.

The expo replays Full MaleCNS exports; it does not run the full model in the
browser. The normal live game and physical runtime select the released
80-neuron policy. The separately stored full-neuron training bundle has not
been inspected by this cleanup. Do not extrapolate generalization or missing
training from either the replay or the absence of external files here.

## Separate research/runtime work

These are technical integration areas, not automatic blockers for delivering
the human game and replay installation. Confirm the requested scope before
retraining or enabling a new controller.

1. Native snapshot, physical contact, motor force/travel, and runtime metrics
   streaming to the biomechanical browser station.
2. A controller trained and evaluated natively on environment v11.
3. Independent trained/evaluated controllers and selectable UI modes for the
   four larger populations.
4. GPU runtime evidence on an NVIDIA host.

## Run and verify

Browser-only autoplay:

```sh
npm ci
npm run dev
```

The Docker stack needs `server/` included in the web build and a running API
for visitor ranking (Nginx alone does not provide it). After that integration
is repaired, the intended CPU-stack commands are:

```sh
docker compose up --build --wait
bash scripts/docker-smoke.sh
```

Open `http://127.0.0.1:8080/` for the game or
`http://127.0.0.1:8080/?biomechanics=1` for the physical station. The first
biomechanical WebSocket session may download roughly 140 MB of pinned FlyGym
meshes; subsequent sessions reuse the named volume.

Full verification commands and immutable engineering constraints are recorded
in `docs/ACTIVE_CONTEXT.md`. Do not bypass the physical action gate or describe
compatibility-mode results as native environment-v11 evaluation.
