# Engineering handoff

This repository is an operational development snapshot of a 3D Crossy
Road/Frogger experiment controlled by a reduced fly connectome.

## Working now

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
- CPU Docker Compose path containing the verified runtime artifacts. Full-size
  FlyBody meshes are integrity-checked and cached in the `fly-data` volume.

## Scientific boundary

The 80-neuron checkpoint is the only released playable connectome controller.
It was trained on environment v6 and is explicitly labeled as compatibility
mode on v11. Neural values are simulated internal activations over measured soma
positions, not measurements from a living fly.

The larger population work is incomplete. Existing 1k and V5 files are
experimental building blocks, not evidence for playable 1k, 5k, 20k, or
124,289-neuron controllers.

## Still missing

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

Complete CPU stack:

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
