# Fly Crossy Connectome

A 3D Crossy Road/Frogger-style installation with a human game and local ranking,
a Full MaleCNS expo replay, and separate live-controller research modes.
Built with TypeScript, React and Three.js, with a Python training environment
and an optional FlyGym/MuJoCo body that presses keyboard keys with its legs.

This project modifies [fly-connectome-template](https://github.com/cobanov/fly-connectome-template)
by [Mert Cobanov](https://github.com/cobanov), adding the crossing game,
controllers, physical keyboard integration and visitor installation.

## Run the game

Requires Node.js **22.18 or newer**:

```sh
npm ci
npm run dev
```

Open the address printed by Vite with the mode you need:

| URL suffix | Mode |
| --- | --- |
| `/?play=1` | Human game, registration and local top-10 ranking |
| `/?expo=1` | Full MaleCNS recorded actions and synchronized exported activity |
| `/` | Released 80-neuron live autoplay and manual controls |
| `/?biomechanics=1` | Physical keyboard station; requires the Python service |

Use WASD or the arrow keys to move, Space to wait, and Escape to pause.
The human mode also provides on-screen controls. At `/`, select **Human / manual**
to take over from the released controller.

To serve the built installation:

```sh
npm run build
npm run preview
```

Open `http://127.0.0.1:4173/?play=1` for the human game. Keep the local server
running while playing. Stop it with Ctrl+C.

## Visitor results

The human game asks for name, surname and email, generates a new seed per
round, and ends at death or 900 simulation decisions (three unpaused minutes).
The server replays the action trace to calculate the score, rejects incomplete
rounds and makes repeated submissions idempotent.

Results persist in **`data/visitor-ranking.sqlite`**, including its SQLite
WAL/SHM sidecars. All completed rounds are retained; only the top 10 are shown.
Name, surname and score are public on that installation's ranking. Email stays
in the database and is not returned by the public API. Earlier databases are
migrated without deleting results.

The database is ignored by Git and excluded from Docker's build context. To
back it up, stop the server and copy the database and any sidecars. Set
`VISITOR_DB_PATH` to use another location. This is a local event kiosk; the
Vite dev and preview servers provide its ranking API. A static-only host does
not provide that API.

## Expo and external training

The teammate keeps the full-neuron training bundle on their machine because
of size. Missing weights in Git do not mean that training was never performed.
The human game and already exported expo do not need those external files.

Expo uses three committed files under `public/assets/expo/`:

- `full-malecns-replay.json`: seed, action sequence, source and reported metrics.
- `full-malecns-activity.json`: neuron IDs, display rules and integrity hashes.
- `full-malecns-activity.bin.gz`: the corresponding recorded neural states.

The browser validates hashes, IDs, seed and frame counts before reproducing
the trajectory. Pause consumes no frames; restart rewinds both actions and
activity. The published trajectory has 1,000 decisions and 438 points on its
original seed. Expo performs replay rather than live Full MaleCNS inference;
its current “real time” label does not describe that distinction accurately.

Activity is simulated model state over measured soma locations, not recordings
from a living fly. Exported signed-state magnitude is quantized to uint8 with
fixed regional gains (optic=1, central=16, descending=16). At most 1,536 nonzero
signals per frame are displayed using measured-ID selection. The display does
not represent every model neuron simultaneously or directly comparable firing
rates. The separate fly-typing video loops independently.

For regeneration or further training, transfer the **original complete bundle**
through shared storage or an external drive, preserving run-directory names
under the ignored `runs/`. For the current expo this includes
`crossy-full-readout-cached-frontier-v12-10-0006-1000/best_FROZEN_1000.pt`, its
matching `run_state.pt`, and the source checkpoints, decoder and graph required
by that experiment. Include code/package versions and a manifest of relative
filenames, sizes and SHA-256 hashes. Participant databases are not training data.

The exporter still reads historical absolute Windows paths in the replay and
checkpoint metadata. Those paths must be resolved for the target machine;
copying one checkpoint alone does not make the whole bundle portable. With
the original environment and artifacts available, run from the repository root:

```sh
python scripts/export-expo-activity.py --device cuda
```

This reconstruction restores recorded recurrent-state anchors and checks every
action against the committed trace. Continuous inference without those anchors
has documented CPU/GPU divergences; the replay alone is not evidence of
generalization to new seeds. Do not replace published provenance silently.

## Live controller and physical keyboard

Normal `/` autoplay and the physical station currently select the reviewed
80-neuron checkpoint in `release/eval-v6/`, with an explicit legacy-observation
adapter. This is separate from the full-neuron training bundle and expo replay.

For the physical station, use the pinned Python 3.12 CPU environment:

```sh
cd python
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements-linux-x86_64-cpu.txt
python -m pip install -e . --no-deps --no-build-isolation
uvicorn fly_crossy.dev_server:app --host 127.0.0.1 --port 8001
```

Then, in another terminal at the repository root:

```sh
VITE_SIMULATION_URL=ws://127.0.0.1:8001/api/simulation npm run dev
```

Open `/?biomechanics=1`. Only confirmed contact by the intended tarsus on the
intended physical key may cause directional game movement. Failed or timed-out
presses become waits. The first FlyBody session downloads about 140 MB of
verified meshes. Full body/contact telemetry and NVIDIA-host validation remain
separate runtime work; CPU is the fallback.

## Docker

Compose defines CPU simulation and web services, plus optional GPU/training
profiles. The current web Docker integration needs repair: its build omits
`server/`, and the final Nginx container does not run the visitor SQLite API.
Use the Vite commands above for the local installation until that is addressed.

## Research code and repository layout

- `src/`, `server/`: browser modes and visitor-ranking API.
- `public/`: distributable assets, measured atlas, notices and expo exports.
- `python/fly_crossy/`: shared environment, controllers, physics and training.
- `scripts/`: asset tooling, export and verification commands.
- `scripts/experiments/cached_frontier/`: preserved versioned research scripts.
- `configs/`, `data/`, `protocol/v2/`: configurations, calibration and protocols.
- `release/`: reviewed checkpoints, evaluation evidence and integrity manifests.
- `tests/`, `python/tests/`, `.github/workflows/`: tests and CI verification.

Generated `runs/`, `reports/`, model downloads, local databases and backups stay
local and ignored. Historical documentation/reports remain recoverable from
Git history; this cleanup does not rewrite it. Before applying this branch on
another machine, back up local run/report files that Git previously tracked.
Agent instructions and Superpowers plans are not required to build or run the
project and are no longer included.

V7 remains an optional research path, not a prerequisite for the installation:

```sh
cd python
python -m fly_crossy.v7.curriculum --profile smoke --out ../runs/crossy-v7-smoke --device cpu
python -m fly_crossy.v7.curriculum --profile 80 --out ../runs/crossy-v7-80-local --device cpu --time-budget-seconds 3600
python -m fly_crossy.v7.curriculum --profile 80 --out ../runs/crossy-v7-80-local --device cpu --resume
```

Larger profiles require the predecessor's promotion report via
`--predecessor-report`. A smoke result cannot promote. Historical cached-frontier
tools can be run from the repository root, for example:

```sh
python scripts/experiments/cached_frontier/repair_full_malecns_cached_frontier_v12_10.py --help
```

## Reduced connectome contract

The reused 80-cell graph was selected for FlyDino, not this crossing task. It
contains 1,296 measured directed edges, with 32 input-role and 16 readout-role
cells. The upstream selection takes visual cells ranked by descending-neuron
contacts, descending targets, and two-hop bridge cells, retaining internal
measured edges and original body IDs. No synthetic edges are introduced.
Selection bias and omitted external inputs limit what can be inferred from it.

The source graph, dataset/source-table hashes, selection roles and provenance
are pinned in `public/data/connectome/manifest.json` and the adjacent notice.
For source contacts `c[j,i]` and assumed presynaptic transmitter signs `s[j]`,
the historical reduced model normalizes incoming recurrence as:

```text
A[i,j] = c[j,i] * s[j] / sum_k |c[k,i] * s[k]|
```

A zero denominator produces zero drive. Topology and recurrent weights are
fixed; engineered sensory projection and actor/critic interfaces are trainable.
The leaky tanh state is reset at episode boundaries; display normalization of
the historical reduced circuit is `clamp((h + 1) / 2, 0, 1)`. These modeling
assumptions are not physiological calibration or biological recordings. The
graph manifest's upstream “8-channel input injection” describes source
provenance, not this game's legacy 370-value observation interface.

The current environment is **v11**, world layout **v6**, observation **v4**
(517 float32 values), reward **v5**, with a **0.2 s** decision interval.
Browser/Python fixtures verify parity. Rendering must not change game state.

## Verify

```sh
npm test
npm run check:assets
npm run build
cd python && python -m pytest tests -q
```

Use the pinned Python 3.12 CPU environment for runtime evidence. Historical
release manifests retain their original environments and hashes.

## Licence and attribution

Built with [fly-connectome-template](https://github.com/cobanov/fly-connectome-template)
by [Mert Cobanov](https://github.com/cobanov). This is a modified distribution
under the custom attribution-required [LICENSE](LICENSE), not an OSI-approved
license. Keep the linked credit readable in the README and web interface;
preserve copyright and third-party notices when redistributing.

MaleCNS data is **CC BY 4.0**, FlyBody **Apache-2.0**, and Kenney assets **CC0 1.0**.
Preserve [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), asset-specific notices,
license texts, source manifests and integrity hashes. Markdown notices beside
assets and hashed release summaries are retained as distribution/evidence
artifacts. They are separate from the removed agent and process documentation.
