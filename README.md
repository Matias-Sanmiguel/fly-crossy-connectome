# Fly Crossy Connectome

A 3D Crossy Road/Frogger-style browser experiment where a reduced fly
connectome plays autonomously while its simulated neural activity is shown over
measured MaleCNS soma positions.

The project combines a deterministic TypeScript/Python game environment,
Kenney CC0 scene assets, a real-time brain view, and an optional FlyGym/MuJoCo
body that presses physical W/A/S/D keys with its legs.

> **Development snapshot:** the browser game, 80-neuron autoplay, live brain
> activity, protocol-v2 bridge, CPU Docker stack, and unified FlyBody/keyboard
> runtime are operational. Larger independently trained neural controllers and
> complete physical telemetry are not finished. See
> [the engineering handoff](docs/HANDOFF.md).

## Quick start

Requires Node.js 22.18 or newer:

```sh
npm ci
npm run dev
```

Open the address printed by Vite. The released 80-neuron controller starts
playing automatically at 1× speed, restarts after terminal outcomes, and
updates the brain panel on every decision.

Choose **Human / manual** to take over. Use arrow keys or W/A/S/D to move,
Space to wait, and Escape to pause. Touch controls expose the same actions.

## What is real, simulated, and provisional

- Soma coordinates and body IDs come from the measured MaleCNS atlas.
- Neural colors are simulated internal values from the controller, not
  biological recordings.
- The active checkpoint was trained and evaluated on environment v6. The
  current game is environment v11, so autoplay uses an explicit, deterministic
  ObservationV4-to-V1 compatibility adapter.
- This compatibility path is labeled in the UI and must not be presented as a
  controller trained natively on v11.
- Existing 1k/V5 artifacts are experimental. The 1,000, 5,000, 20,000, and
  124,289-neuron modes still need independent training and evaluation.

## Biomechanical mode

Start the frontend and the Python runtime separately:

```sh
cd python
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements-linux-x86_64-cpu.txt
python -m pip install -e . --no-deps --no-build-isolation
uvicorn fly_crossy.dev_server:app --host 127.0.0.1 --port 8001
```

In another terminal:

```sh
VITE_SIMULATION_URL=ws://127.0.0.1:8001/api/simulation npm run dev
```

Open `/?biomechanics=1` on the Vite address. Directional game actions are
authoritative only after the intended FlyBody tarsus physically confirms the
intended keyboard key. Invalid, wrong, ambiguous, or timed-out contacts fail
closed.

The bridge currently streams controller neural activity and authoritative
action results. Native body snapshots, contact frames, force/travel, and
runtime metrics remain pending and are shown as unavailable rather than
fabricated.

The first FlyBody session downloads roughly 140 MB of pinned full-size meshes
from FlyGym's public asset host and verifies the complete inventory.

## Docker

The default Compose stack uses CPU with the web app and simulation behind one
origin:

```sh
docker compose up --build --wait
bash scripts/docker-smoke.sh
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). The game is at `/`; the
physical station is at `/?biomechanics=1`. Verified graph, checkpoint,
calibration, and model manifests are packaged in the simulation image. FlyGym
meshes are cached in the persistent `fly-data` volume.

Stop the project without deleting that cache:

```sh
docker compose down
```

The optional training and NVIDIA profiles are:

```sh
docker compose --profile training run --rm trainer
docker compose --profile gpu up --build simulation-gpu web-gpu
```

CPU is the required default and the runtime can resolve GPU requests back to
CPU when fallback is allowed. The CUDA profile still requires validation on an
NVIDIA host.

## Experimental V7 visual training

V7 trains a privileged ObservationV4 teacher and distills it into an RGB-only
measured-connectome student. It is experimental: the released 80-neuron
environment-v6 compatibility controller remains authoritative until a separate
runtime-integration review approves a V7 checkpoint.

Run the CPU contract smoke, then a meaningful 80-neuron run:

```sh
bash scripts/v7-training-smoke.sh
cd python
python -m fly_crossy.v7.curriculum --profile smoke --out ../runs/crossy-v7-smoke --device cpu
python -m fly_crossy.v7.curriculum --profile 80 --out ../runs/crossy-v7-80-local --device cpu --time-budget-seconds 3600
python -m fly_crossy.v7.curriculum --profile 80 --out ../runs/crossy-v7-80-local --device cpu --resume
```

Larger populations are locked behind the preceding report:

```sh
python -m fly_crossy.v7.curriculum --profile 1k --out ../runs/crossy-v7-1k-local --device cpu --predecessor-report ../runs/crossy-v7-80-local/report.json
python -m fly_crossy.v7.curriculum --profile full --out ../runs/crossy-v7-full-local --device cpu --flyhard-root /path/to/flyhard --predecessor-report ../runs/crossy-v7-1k-local/report.json
```

Use `--device cpu` as the universal fallback. `--device cuda` is available only
when PyTorch detects a working NVIDIA CUDA runtime. Preflight requires 2 GiB
free for smoke/80, 5 GiB for 1k, and 15 GiB for full. Each ignored run directory
keeps `best.pt`, resumable `latest.pt`, `dataset-manifest.json`, and
`report.json`. The report's `decision.nextAction` is `promote`, `continue`, or
`reject`; smoke is always `contractOnly` and can never promote.

## Current contracts

- Environment: v11.
- Preserved deterministic world layout: v6.
- Observation: v4, 517 float32 values.
- Reward: v5.
- Decision interval: 0.2 seconds.
- Released playable connectome: 80 neurons, environment-v6 compatibility.

The browser and Python implementations are checked against shared generated
fixtures. A missing visual asset falls back to native geometry and cannot
change simulation state.

## Assets and provenance

The scene uses unmodified GLB models from Kenney's City Kit (Roads), Car Kit,
Train Kit, and Mini Forest packs under CC0 1.0. Source URLs, archive hashes,
local hashes, byte counts, and roles are recorded in
[`public/assets/kenney/manifest.json`](public/assets/kenney/manifest.json) and
[`public/assets/kenney/NOTICE.md`](public/assets/kenney/NOTICE.md).

The atlas contains soma positions, not neurite morphology, synaptic edges, or
a full brain segmentation. Of 140,024 bundled positions, 124,289 classified
optic, central, and descending somata are rendered. The
[`atlas manifest`](public/data/brain-atlas/manifest.json) and
[`data notice`](public/data/brain-atlas/NOTICE.md) document filters, hashes, and
provenance.

## Verify

```sh
npm test
npm run check:assets
npm run build
cd python && python -m pytest tests -q
bash scripts/v7-training-smoke.sh
```

For release-quality Python evidence, use the hash-locked Python 3.12 CPU image.
The separately recorded Python 3.14 environment under `release/eval-v1/` is a
historical reproduction boundary and must not be silently rewritten to match a
different machine.

## Licence and attribution

This repository modifies an attribution-required, source-available template.
Its custom license is not OSI-approved; see [LICENSE](LICENSE) and
[ATTRIBUTION.md](ATTRIBUTION.md).

Built with [fly-connectome-template][repo] by [Mert Cobanov][author]. Modified
distributions must keep that linked credit readable and identify the changes.

MaleCNS data remains **CC BY 4.0**. FlyBody remains **Apache-2.0**. Kenney
assets remain **CC0 1.0**. Preserve [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

[repo]: https://github.com/cobanov/fly-connectome-template
[author]: https://github.com/cobanov
