# Fly Crossy Connectome

An interactive browser laboratory that pairs a deterministic, isometric fly-crossing environment with measured MaleCNS soma anatomy and inspectable controller output.

This project is a modified version of the fly connectome template. It adds a playable crossing task, fixed-step simulation and replay contracts, conventional and remote controller boundaries, neural telemetry, and a responsive laboratory interface.

> **Development snapshot:** the browser game, autonomous reduced controller,
> live simulated brain view, runtime protocol, and the first four native
> biomechanics layers are present. The final MuJoCo world/server bridge and the
> five independently trained neural sizes are **not finished**. Read
> [the engineering handoff](docs/HANDOFF.md) for the exact boundary and use
> [the continuation prompt](docs/CONTINUATION_PROMPT.md) to hand the repository
> to another coding agent.

## Run locally

Use Node.js **22.18 or newer**:

```sh
npm ci
npm run dev
```

Open the local address printed by Vite. Once the anatomy loads, the bundled reduced-connectome controller starts playing automatically at 1× speed and updates its 80 mapped model values in the brain panel on every decision. After a terminal outcome it waits one second, changes to a new seed, and continues. These are simulated controller values over measured soma positions, not recordings from a living fly.

Choose **Human / manual** to take over: use the arrow keys or W/A/S/D to move, Space to wait, and Escape to pause. The on-screen controls provide the same actions for touch and pointer input.

## Run with Docker

Docker Compose starts the CPU simulation and same-origin web proxy by default. The first build downloads the hash-locked Python and Node dependencies:

```sh
docker compose up --build --wait
bash scripts/docker-smoke.sh
```

Open <http://127.0.0.1:8080>. Requests under `/api/` are proxied to the simulation service, including WebSocket upgrade headers. The service health endpoint is available at `/api/healthz`. The default service deliberately has no artifact manifest: health checks work, but controller configuration fails closed until verified graph and checkpoint artifacts are supplied by a later runtime integration.

Stop only this Compose project when finished. The `fly-data` named volume is retained:

```sh
docker compose down
```

The optional trainer uses the CPU image and prints its command interface without beginning a training run:

```sh
docker compose --profile training run --rm trainer
```

On a host with the NVIDIA Container Toolkit and a supported GPU, start only the targeted GPU pair. Targeting these services prevents the default CPU pair from starting:

```sh
docker compose --profile gpu up --build simulation-gpu web-gpu
```

The simulation images use Python 3.12 and run as UID/GID 10001. CPU and CUDA dependencies are independently hash-locked in `python/requirements-linux-x86_64-cpu.txt` and `python/requirements-linux-x86_64-cu130.txt`, with Pydantic fixed at 2.11.7 for the protocol boundary.

## What is included

- A deterministic environment-v3 crossing world with roads, rails, rivers, hazards, rewards, terminal states, and seeded replay.
- Bounded group-level route checks with deterministic retries and a safe grass fallback, plus speed and hazard-density progression over distance.
- A human controller, a scripted controller, local dense-policy JSON loading, and a versioned remote-controller protocol.
- A bundled reduced-connectome policy that autoplays continuously while the brain panel visualizes its simulated activity in real time.
- A fixed-size `ObservationV1` boundary shared by every non-human controller.
- A measured MaleCNS v1.0 soma atlas whose activity values are keyed only by verified body IDs.
- A separate Flybody anatomical surface view; it is not a motor or physics simulation.
- A responsive layout that keeps the crossing environment and brain atlas available on desktop and mobile.

## Controller and anatomy scope

The atlas contains **cell-body positions**, not neurite morphology, synaptic edges, or a complete brain segmentation. Of 140,024 bundled measured positions, the viewer draws 124,289 classified optic, central, and descending somata. Missing positions are never generated, and native coordinate proportions are preserved.

Human and scripted modes deliberately show **no neural output**. Dense policies show **model output** only when a policy explicitly maps a hidden layer to atlas-visible body IDs; the released dense policy has no such mapping. The fixed graph shows **simulated reduced-circuit activity**. None of these values are measured neural activity. Model values are accepted only for atlas-visible MaleCNS body IDs with normalized values in `[0, 1]`.

The [atlas manifest](public/data/brain-atlas/manifest.json) records source filters and hashes. The [data notice](public/data/brain-atlas/NOTICE.md) documents the export and its provenance.

## Policy files

Choose **Load policy JSON** to read a dense or reduced-connectome policy locally in the browser. The validator rejects incompatible versions, observation shapes, action sets, layer dimensions, non-finite values, duplicate or unknown anatomy IDs, and invalid provenance. Policy files are limited to 10 MB.

The exact first-release smoke checkpoints, browser policies, training metadata, training curves, and evaluation outputs are tracked under [`release/eval-v1/`](release/eval-v1/). They are simulated controller artifacts, not biological firing data.

## Deterministic world and difficulty

Environment v3 generates every five-row group only from its world seed and group coordinate, so requesting one row or a large chunk produces identical row content. Eight deterministic candidate groups are checked with the same floating-point transition used by live play—including repeated 0.2-second time accumulation, swept collisions, log carry, and exact bounds—for a route from the preceding safe row to the recovery row. The search is capped at 80 steps and 1,024 explored transitions; if no witness is found within that budget, the group becomes a conservative grass fallback instead of emitting a knowingly impassable crossing. Environment v3 supersedes v2 because the older solver used integer time ticks that could accept a witness which failed under the authoritative floating-point simulation.

Difficulty advances every ten forward groups, capped at level 3. Maximum hazard speed rises from 2 to 5 cells/second; minimum speed rises from 1 to 2; non-rail hazard counts rise from 2–3 to 3–6. Every candidate is checked after those parameters are applied. These are generation guarantees, not a guarantee that every player action or arbitrary arrival phase survives.

## Reproduce the bounded controller evaluation

The Python package lives under `python/`. From the repository root:

```sh
cd python
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://pypi.org/simple pip==26.2.1
python -m pip install --index-url https://pypi.org/simple \
  -r requirements-release-linux-x86_64-cu130.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m pytest tests -q

timeout 180s python -m fly_crossy.train \
  --controller conventional \
  --seed smoke \
  --steps 2048 \
  --envs 4 \
  --learning-rate 0.0003 \
  --output runs/reproduce-conventional \
  --device cpu

timeout 180s python -m fly_crossy.train \
  --controller connectome \
  --seed smoke \
  --steps 2048 \
  --envs 4 \
  --learning-rate 0.0003 \
  --output runs/reproduce-connectome \
  --device cpu

timeout 180s python -m fly_crossy.evaluate \
  --config ../configs/eval-v1.json \
  --output runs/eval-v1

sha256sum runs/reproduce-conventional/checkpoint.pt \
  runs/reproduce-connectome/checkpoint.pt
```

Use Python 3.14.7 as recorded in [`.python-version`](.python-version). [`release-environment-linux-x86_64.json`](python/release-environment-linux-x86_64.json) declares the Linux x86_64/CPython build boundary, PyPI index, torch `2.12.1+cu130` runtime build, and CPU execution used for the release. [`requirements-release-linux-x86_64-cu130.txt`](python/requirements-release-linux-x86_64-cu130.txt) pins the complete active dependency closure for that declared platform, and an automated test compares it with installed package metadata. This historical release closure is separate from the hash-locked Python 3.12 container requirements. The installed metadata did not retain wheel hashes, so the release file is an exact-version environment manifest—not a hermetic wheel lock or a promise of cross-platform checkpoint byte identity. `timeout` is the GNU command used for the 180-second per-run time box. The expected checkpoint hashes are `03640098bf3ff5e8e2164c1cd43dff0791dea6d2f51e711a8d2d7217648c2620` and `abd1dc3f472de7687599d60cf73ecfbc31207ede197deeae353b8f4d4cb76e9f`; the evaluator verifies those hashes **before** loading the tracked checkpoints declared in [`configs/eval-v1.json`](configs/eval-v1.json).

The evaluator writes a local copy of `metrics.json`, `metrics.csv`, and `summary.md` under `python/runs/eval-v1/`. The immutable released copies and full file hashes are in [`release/eval-v1/manifest.json`](release/eval-v1/manifest.json), and `npm test` verifies the manifest from a clean checkout. See [Controller evaluation v1](docs/experiments/evaluation-v1.md) for budgets, held-out results, controls, and limitations. No human-recorded traces were available; none were fabricated.

The reduced controller learns a 370-value `ObservationV1` to 80-cell sensory projection. The bundled manifest's eight-channel injection language describes the reused FlyDino source artifact, not the input interface implemented here.

## Verify and build

```sh
npm test
npm run check:assets
npm run build
```

To inspect the production build locally:

```sh
npm run preview
```

The static output is written to `dist/` and supports deployment under a subpath.

Vite may emit a non-blocking large-chunk advisory for the Three.js renderer. The first release keeps that renderer in the main bundle; the advisory does not indicate a failed build, and code splitting remains a future loading optimization.

## Licence and attribution

This repository contains modifications to an attribution-required, source-available template. The template license is custom and is not an OSI-approved license. See [LICENSE](LICENSE) and [ATTRIBUTION.md](ATTRIBUTION.md).

Built with [fly-connectome-template][repo] by [Mert Cobanov][author].

Keep that linked credit readable in both the web interface and this README. Modified distributions must identify that changes were made.

MaleCNS data remains **CC BY 4.0**, credited to FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, and Google Research. Flybody remains **Apache-2.0**. These licenses are independent from the template license; preserve the [third-party notices](THIRD_PARTY_NOTICES.md).

[repo]: https://github.com/cobanov/fly-connectome-template
[author]: https://github.com/cobanov
