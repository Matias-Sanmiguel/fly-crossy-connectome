# Fly Crossy Connectome

An interactive browser laboratory that pairs a deterministic, isometric fly-crossing environment with measured MaleCNS soma anatomy and inspectable controller output.

This project is a modified version of the fly connectome template. It adds a playable crossing task, fixed-step simulation and replay contracts, conventional and remote controller boundaries, neural telemetry, and a responsive laboratory interface.

## Run locally

Use Node.js **22.18 or newer**:

```sh
npm ci
npm run dev
```

Open the local address printed by Vite. The default mode is keyboard-first manual control: use the arrow keys or W/A/S/D to move, Space to wait, and Escape to pause. The on-screen controls provide the same actions for touch and pointer input.

## What is included

- A deterministic crossing world with roads, rails, rivers, hazards, rewards, terminal states, and seeded replay.
- A human controller, a scripted controller, local dense-policy JSON loading, and a versioned remote-controller protocol.
- A fixed-size `ObservationV1` boundary shared by every non-human controller.
- A measured MaleCNS v1.0 soma atlas whose activity values are keyed only by verified body IDs.
- A separate Flybody anatomical surface view; it is not a motor or physics simulation.
- A responsive layout that keeps the crossing environment and brain atlas available on desktop and mobile.

## Controller and anatomy scope

The atlas contains **cell-body positions**, not neurite morphology, synaptic edges, or a complete brain segmentation. Of 140,024 bundled measured positions, the viewer draws 124,289 classified optic, central, and descending somata. Missing positions are never generated, and native coordinate proportions are preserved.

Manual mode deliberately shows no neural output. Scripted and model policies receive the same bounded observation and return one action per authoritative simulation step. Model activity is accepted only for atlas-visible MaleCNS body IDs with normalized values in `[0, 1]`.

The [atlas manifest](public/data/brain-atlas/manifest.json) records source filters and hashes. The [data notice](public/data/brain-atlas/NOTICE.md) documents the export and its provenance.

## Policy files

Choose **Load policy JSON** to read a dense policy locally in the browser. The validator rejects incompatible versions, observation shapes, action sets, layer dimensions, non-finite values, duplicate or unknown anatomy IDs, and invalid provenance. Policy files are limited to 10 MB.

Training and inference stay in your own stack. No trained policy, neural simulator, or biological firing data is included.

## Reproduce the bounded controller evaluation

The Python package lives under `python/`. From the repository root:

```sh
cd python
python -m pip install -e '.[test]'
python -m pytest tests -q

timeout 180s python -m fly_crossy.train \
  --controller conventional \
  --seed smoke \
  --steps 2048 \
  --envs 4 \
  --learning-rate 0.0003 \
  --output runs/smoke-conventional \
  --device cpu

timeout 180s python -m fly_crossy.train \
  --controller connectome \
  --seed smoke \
  --steps 2048 \
  --envs 4 \
  --learning-rate 0.0003 \
  --output runs/smoke-connectome \
  --device cpu

timeout 180s python -m fly_crossy.evaluate \
  --config ../configs/eval-v1.json \
  --output runs/eval-v1
```

`timeout` above is the GNU command used for the first-release CPU time box. The evaluator writes `metrics.json`, `metrics.csv`, and `summary.md` under `python/runs/eval-v1/`; local runs are intentionally ignored by Git. See [Controller evaluation v1](docs/experiments/evaluation-v1.md) for the exact released budgets, hashes, held-out results, controls, and limitations. No human-recorded traces were available for evaluation v1.

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
