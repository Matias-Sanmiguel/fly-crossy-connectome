# Kenney UI and Repository Cleanup Design

**Date:** 2026-09-15  
**Status:** Approved for implementation
**Scope:** Curated Kenney visual refresh, shared laboratory UI, renderer cleanup,
reproducible local runtime, and repository hygiene.

## Context

The current browser application is functional, but its crossing scene is built
almost entirely from boxes and spheres. The main application and biomechanical
debug application also have separate presentation shells, `App.tsx` and
`GameScene.tsx` carry too many responsibilities, and the repository contains
280 committed pytest temporary files. The new biomechanical development server
references an ignored checkpoint that is absent from a clean clone, while the
committed handoff document describes an older state of the project.

This pass makes the project presentable and reproducible without changing the
authoritative game rules or MuJoCo contact rules. It does not implement the
missing 1,000, 5,000, 20,000, or 124,289-neuron controllers.

## Product Direction

The interface will become a playful low-poly field laboratory: a large Kenney
crossing diorama remains the primary surface, while the MaleCNS brain stays
visible as a first-class panel. Scientific labels remain explicit and the UI
never presents simulated model activity as a biological recording.

The desktop hierarchy is:

1. Compact header with project identity, runtime state, and the GitHub link.
2. A control rail with controller/runtime mode, seed, speed, pause, and reset.
3. A 7:5 main split: crossing environment on the left, brain atlas on the
   right.
4. A compact telemetry strip below the brain containing decision, reward,
   motor/contact state, and provenance.
5. Secondary policy upload, scientific scope, and attribution in an expandable
   drawer rather than the primary control surface.

On narrow screens the order is game, primary controls, brain, telemetry, then
secondary information. Touch targets remain at least 44 by 44 CSS pixels.
Keyboard focus is visible, status changes use polite live regions, and reduced
motion disables decorative transitions while preserving game-state animation.

## Visual System

The visual language combines Kenney's saturated low-poly assets with a dark
instrument-console frame:

- Canvas: `#08110f`
- Raised surface: `#101d1a`
- Raised highlight: `#182824`
- Primary lime: `#b7f34a`
- Neural cyan: `#62d9ff`
- Contact amber: `#ffb454`
- Error coral: `#ff6b62`
- Primary text: `#f4f7f2`
- Muted text: `#9fb0a9`

Panels use restrained borders, 14-pixel corners, and shallow shadows. Status is
expressed with text plus color, never color alone. Existing system fonts remain
to avoid another external runtime dependency.

The current procedural fly remains unchanged because it represents the project
subject, not generic scenery. The brain and anatomical FlyBody renderers also
retain their measured assets and existing provenance.

## Curated Kenney Asset Set

Only the nine GLB files below are committed. Whole source archives, FBX files,
unused models, and preview images are excluded. The selected files total less
than 750 KB before repository compression.

### City Kit (Roads)

Source: <https://kenney.nl/assets/city-kit-roads>  
License: Creative Commons CC0  
Archive SHA-256:
`22058af3d68173a7cf9bda9f0e243a8cef6bd68168c302ebc76327063849674e`

- `road-straight.glb` — road lane surface. SHA-256
  `f8f744fb6fc96dedd5ad1f763b550166f2624a3757bf36b4b39969d2e7cacac5`
- `traffic-light.glb` — deterministic roadside decoration. SHA-256
  `6c9ee253b7370e1043168493b2e662b349484a0cdd02100cff04d9d8d1bd9abb`

### Car Kit

Source: <https://kenney.nl/assets/car-kit>  
License: Creative Commons CC0  
Archive SHA-256:
`fac7dacac5c7874348cf19729af3ef205f3d366493edaf0a827d93f4fdf3d0c4`

- `sedan.glb` — car hazard. SHA-256
  `b532ea7d2c59f7f6b22b138cf1955218a2c1898f1cea932af4d3fd563c3959b7`
- `truck.glb` — truck hazard. SHA-256
  `5e1bc66c343501dd7d5734b57e672dd949bf95d7abdce680c6508cef99d374f8`

### Train Kit

Source: <https://kenney.nl/assets/train-kit>  
License: Creative Commons CC0  
Archive SHA-256:
`cf50d77e8cbacbf38dd50826d4bce5392db8e4f67373d3c4e583b0ed0e474475`

- `train-diesel-a.glb` — train hazard. SHA-256
  `20c6c7c23c005e8cda1c1e922473c0fc9c93fc7c1d1bc5a637d640424f811cc3`
- `track-detailed.glb` — rail lane surface. SHA-256
  `b56cab0c6cf54ea1e3e3dbdffc06995b7a432db41ee39bf063d54b171f937351`

### Mini Forest

Source: <https://kenney.nl/assets/mini-forest>  
License: Creative Commons CC0  
Archive SHA-256:
`8691614018075a66458e35915b8c358c2e6178648aedadafcdf313b924aa6581`

- `tree.glb` — grass-lane decoration. SHA-256
  `0075d5059c1ea855fecc1dba3c6262c3a01924c50b971ddde2eb8b0b5a62851a`
- `rocks-low.glb` — grass/river-bank decoration. SHA-256
  `f98f27210e6d4f43192290fff8af07e0e79bdd25604b78c44a8bd7b3d98ea259`
- `plant.glb` — grass-lane detail. SHA-256
  `4ce1784382a2c986b5a63cd4cf7fc41f8775148464021fc4e10d780cc7892789`

The river surface and log hazard remain lightweight project-native geometry;
none of the chosen packs contains a suitable self-contained GLB log that fits
the deterministic collision footprint. This is an explicit exception, not an
unfinished asset choice.

The selected assets live under `public/assets/kenney/<pack>/`. A committed
`public/assets/kenney/manifest.json` records source URL, upstream archive hash,
local file path, file hash, byte count, license, and semantic role. A single
`public/assets/kenney/NOTICE.md` explains that attribution is not required by
CC0 but records provenance voluntarily. `THIRD_PARTY_NOTICES.md` links to it.

## Renderer Architecture

`GameScene` will stop owning asset discovery, loading, model normalization,
scene construction, resizing, and frame updates in one effect.

- `src/game/kenneyAssets.ts` defines the typed semantic registry and validates
  that every hazard/lane role has either a Kenney model or an explicit native
  fallback.
- `src/game/kenneyLoader.ts` loads each GLB once through Three's `GLTFLoader`,
  normalizes its bounding box to a declared logical footprint, and returns
  cloneable static groups. Failed loads resolve to a typed failure instead of
  rejecting the entire scene.
- `src/game/createGameRenderer.ts` owns Three scene, camera, lighting, object
  pools, resize, disposal, and one `render(state, events, now)` boundary.
- `src/components/GameScene.tsx` becomes the React lifecycle adapter only.

Static Kenney groups are pooled up to the existing renderer capacity and are
shown, transformed, and hidden instead of created every frame. The authoritative
grid position, hazard length, movement, collision, and camera behavior remain
unchanged. Decorative placement is a pure function of world seed and row, so
replay screenshots remain deterministic.

If a GLB fails to load, the scene retains the existing procedural mesh for that
semantic role and exposes `Using geometric fallback` in a non-blocking status.
Asset failure must never stop the game or change simulation state.

## Shared Application UI

The normal and biomechanical views will use one shell rather than duplicating
headers and panels.

- `src/components/lab/LabShell.tsx` owns page regions and responsive layout.
- `src/components/lab/LabHeader.tsx` renders identity and global status.
- `src/components/lab/ExperimentControls.tsx` renders the primary control rail.
- `src/components/lab/PanelFrame.tsx` supplies consistent title, badge, content,
  and footer slots.
- `src/components/lab/TelemetryStrip.tsx` renders controller and physical state
  without deciding game behavior.
- `src/App.tsx` retains browser-controller state orchestration but delegates
  presentation.
- `src/BiomechanicsDebugApp.tsx` retains the physical station hook but delegates
  to the same shell and always includes `BrainScene`.

The `?biomechanics=1` entry remains during this pass so the incomplete physical
runtime is not mislabeled as the default finished experience. Both entry paths
must nevertheless look like the same product. Missing native contact, snapshot,
or neural streams are labeled `Awaiting runtime telemetry`; the UI does not
invent them from requested actions or animation.

## Reproducible Development Runtime

The biomechanical development server must start from a clean checkout. It will
use the committed 80-neuron checkpoint at
`release/eval-v1/training/connectome/checkpoint.pt` and its verified SHA-256,
rather than the ignored `python/runs/connectome-80-v1/checkpoint.pt` path.
`runtime-artifacts-biomechanics.json`, evaluation references, and the server
loader must agree on the same committed graph/checkpoint identity.

This pass does not claim that the requested five controller sizes exist. The UI
offers only modes backed by a verified committed artifact; unavailable sizes are
shown as planned, not selectable.

## Repository and Code Cleanup

The root `.gitignore` becomes the authoritative cross-language ignore file and
includes:

- `node_modules/`, `dist/`, `coverage/`, `*.tsbuildinfo`
- `.env*` except `.env.example`
- `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.pytest-tmp/`, `.coverage*`,
  `htmlcov/`, `.mypy_cache/`, `.ruff_cache/`, `*.egg-info/`, `.venv/`
- generated `python/runs/`, root `outputs/`, local worktrees, editor metadata,
  and operating-system metadata

The nested `python/.gitignore` is removed after its unique rules are migrated.
All committed `python/.pytest-tmp/**` files are removed from Git. Generated runs
remain ignored; release evidence and the one runtime checkpoint remain in their
explicit committed release directories.

Formatting cleanup is limited to files touched by this pass. Existing trailing
whitespace in touched biomechanical/runtime files is removed, but no unrelated
physics rewrite is included. `docs/HANDOFF.md` is updated to reflect the
unified world and server bridge now present.

## Testing and Acceptance

The change is accepted only when all of the following hold:

1. A clean asset verification command checks all nine byte sizes and SHA-256
   hashes against the manifest.
2. The game loads the curated models without changing simulation fixtures,
   hazard positions, collision outcomes, or replay determinism.
3. A forced GLB failure leaves a playable geometric fallback and visible status.
4. The default and biomechanical browser entries share the same shell; the
   brain panel is present in both.
5. Desktop, tablet, and 360-pixel mobile layouts have no horizontal overflow,
   inaccessible controls, or obscured game state.
6. `python -m fly_crossy.dev_server` imports and reaches health readiness from a
   clean checkout with no ignored checkpoint dependency.
7. The current byte-reproducible calibration test is repaired by regenerating
   the committed calibration artifact from the authoritative generator; it is
   not weakened or skipped.
8. `npm test`, asset verification, `npm run build`, and the full Python suite
   pass freshly.
9. `git ls-files 'python/.pytest-tmp/**'` prints nothing and a test run does not
   dirty the worktree.
10. The CPU Docker smoke passes and all started services are stopped afterward.

## Non-Goals

- Creating or training the four missing larger neural controllers.
- Changing MaleCNS data, brain coordinates, FlyBody mesh data, or provenance.
- Changing Crossy Road world generation, authoritative collision rules, or
  MuJoCo contact confirmation thresholds.
- Replacing the fly with a generic Kenney character.
- Adding a new UI framework or CSS dependency.
- Shipping the full Kenney archives or unused source models.
