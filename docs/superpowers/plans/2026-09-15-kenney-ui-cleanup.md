# Kenney UI and Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace generic crossing primitives with a small verified Kenney asset set, unify the normal and biomechanical visual shells, and leave a clean reproducible repository.

**Architecture:** Keep game and MuJoCo authority unchanged. Add a typed cached GLB layer beneath a refactored Three renderer, then compose both browser modes from shared laboratory UI components. Repair clean-clone reproducibility before visual work so every later task starts from a trustworthy baseline.

**Tech Stack:** React 19, TypeScript 5.9, Three.js 0.186 with `GLTFLoader`, Vite 8, Node test runner, Python 3.12, pytest, FastAPI, MuJoCo 3.9, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-15-kenney-ui-cleanup-design.md`

## Global Constraints

- Do not change deterministic world generation, collision rules, rewards, or MuJoCo contact thresholds.
- Keep the brain visible in both browser entries and label activity as simulated model output.
- Commit only the nine GLBs listed in the spec; no archives, FBX files, or unused models.
- A failed visual asset load must fall back to native geometry without affecting simulation.
- Keep `?biomechanics=1` opt-in; do not present it as the finished default product.
- Preserve template, MaleCNS, FlyBody, and Kenney provenance.
- Add no UI framework, CSS package, remote font, or runtime CDN.
- Add no automated-agent coauthor trailers.

---

### Task 1: Restore a clean and reproducible baseline

**Files:**
- Create: `tests/repository-hygiene.test.mjs`
- Create: `python/tests/test_dev_server.py`
- Modify: `.gitignore`
- Delete: `python/.gitignore`
- Delete: `python/.pytest-tmp/**`
- Delete: `configs/eval-80-v1.json`
- Delete: `configs/eval-80-v2.json`
- Modify: `runtime-artifacts-biomechanics.json`
- Modify: `python/fly_crossy/dev_server.py`
- Regenerate: `data/calibration/keyboard-reach-v1.json`

**Interfaces:**
- Consumes: committed checkpoint `release/eval-v1/training/connectome/checkpoint.pt`, SHA-256 `abd1dc3f472de7687599d60cf73ecfbc31207ede197deeae353b8f4d4cb76e9f`.
- Produces: clean-checkout runtime, one authoritative ignore file, and no tracked pytest output.

- [ ] **Step 1: Write failing hygiene tests**

Create `tests/repository-hygiene.test.mjs`:

```js
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const root = new URL('../', import.meta.url);

test('generated Python output is not tracked', () => {
  const tracked = execFileSync(
    'git', ['ls-files', 'python/.pytest-tmp/**', 'python/runs/**'],
    { cwd: root, encoding: 'utf8' },
  ).trim();
  assert.equal(tracked, '');
});

test('root ignore owns cross-language output', async () => {
  const ignore = await readFile(new URL('.gitignore', root), 'utf8');
  for (const pattern of ['.pytest-tmp/', 'python/runs/', '*.py[cod]', '.ruff_cache/', 'coverage/']) {
    assert.ok(ignore.includes(pattern), pattern);
  }
});
```

- [ ] **Step 2: Write the failing clean-checkout runtime test**

Create `python/tests/test_dev_server.py`:

```python
from pathlib import Path
from fly_crossy.dev_server import CHECKPOINT_PATH, controller

ROOT = Path(__file__).resolve().parents[2]

def test_dev_server_uses_committed_release_checkpoint() -> None:
    expected = ROOT / "release/eval-v1/training/connectome/checkpoint.pt"
    assert CHECKPOINT_PATH == expected
    assert CHECKPOINT_PATH.is_file()
    assert controller.node_count == 80
```

- [ ] **Step 3: Run the new tests and confirm RED**

Run:

```sh
node --experimental-strip-types --test tests/repository-hygiene.test.mjs
cd python
/tmp/fly-crossy-lock-py312/bin/python -m pytest tests/test_dev_server.py -q
```

Expected: tracked `.pytest-tmp` paths and the missing ignored checkpoint make the tests fail.

- [ ] **Step 4: Consolidate ignore rules and remove generated files**

Put the spec's Node, Python, test, environment, editor, OS, run, output, and worktree rules in root `.gitignore`. Migrate `runs/` as `python/runs/`, delete `python/.gitignore`, and remove all tracked `python/.pytest-tmp/**` and `python/runs/**` files. Preserve `release/eval-v1/**`, which is the immutable release source.

Delete both `configs/eval-80-*.json`; they reference absent ignored checkpoints and are not release evidence.

- [ ] **Step 5: Make runtime artifacts clean-clone reproducible**

Set `CHECKPOINT_PATH` in `dev_server.py` to the committed release checkpoint. Update `runtime-artifacts-biomechanics.json` to that same path and SHA-256. Keep the existing graph path and hash.

- [ ] **Step 6: Regenerate the calibration artifact**

Run from `python/`:

```sh
/tmp/fly-crossy-lock-py312/bin/python -m fly_crossy.biomechanics.calibration \
  --config ../configs/biomechanics-v1.json \
  --manifest ../data/manifests/flybody-v1.json \
  --output ../data/calibration/keyboard-reach-v1.json
/tmp/fly-crossy-lock-py312/bin/python -m pytest \
  tests/biomechanics/test_motor.py::test_calibration_cli_is_byte_reproducible \
  -q --basetemp=/tmp/fly-crossy-calibration-test
```

Expected: PASS without weakening byte equality.

- [ ] **Step 7: Verify and commit the baseline**

Run `npm test`, the full Python suite with `--basetemp=/tmp/fly-crossy-clean-baseline`, and `git diff --check`. Remove trailing whitespace only in files touched by this task. Confirm `git ls-files 'python/.pytest-tmp/**'` is empty.

Commit:

```sh
git add -A
git commit -m "chore: restore reproducible project baseline"
```

---

### Task 2: Import and verify the curated Kenney assets

**Files:**
- Create: `public/assets/kenney/city-roads/{road-straight,traffic-light}.glb`
- Create: `public/assets/kenney/car-kit/{sedan,truck}.glb`
- Create: `public/assets/kenney/train-kit/{train-diesel-a,track-detailed}.glb`
- Create: `public/assets/kenney/mini-forest/{tree,rocks-low,plant}.glb`
- Create: `public/assets/kenney/manifest.json`
- Create: `public/assets/kenney/NOTICE.md`
- Create: `tests/kenney-assets.test.mjs`
- Modify: `scripts/check-assets.mjs`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `public/THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Consumes: exact URLs and hashes in the approved spec.
- Produces: a nine-entry manifest as the single asset identity source.

- [ ] **Step 1: Write the failing manifest test**

Require version 1, exactly nine unique roles and paths, local `.glb` paths, official Kenney source URLs, positive byte counts, 64-hex hashes, and `CC0-1.0` licenses.

```js
assert.equal(manifest.assets.length, 9);
assert.equal(new Set(manifest.assets.map((item) => item.role)).size, 9);
for (const item of manifest.assets) {
  assert.match(item.path, /^assets\/kenney\/.+\.glb$/);
  assert.match(item.sha256, /^[a-f0-9]{64}$/);
  assert.equal(item.license, 'CC0-1.0');
}
```

- [ ] **Step 2: Run the test and confirm RED**

Run `node --experimental-strip-types --test tests/kenney-assets.test.mjs`.

Expected: missing manifest failure.

- [ ] **Step 3: Extract only the approved GLBs**

Download the four immutable archive URLs recorded in the spec into a temporary directory. Verify archive SHA-256 before extraction. Extract only the nine named `Models/GLB format/*.glb` entries to the exact public paths. Do not retain archives or unused files.

- [ ] **Step 4: Write manifest and notice**

Each entry records `role`, `path`, `bytes`, `sha256`, `source`, `archiveSha256`, `upstreamPath`, and `license`. Use exactly these roles:

```text
lane.road, lane.rail, decoration.traffic-light,
decoration.tree, decoration.rocks, decoration.plant,
hazard.car, hazard.truck, hazard.train
```

Record voluntary attribution in `NOTICE.md` and link it from both third-party notice files.

- [ ] **Step 5: Extend asset validation and tamper tests**

Refactor `scripts/check-assets.mjs` to export a Kenney manifest validator and retain its CLI behavior. Reject extra keys, path escape, duplicate path/role, wrong inventory count, incorrect bytes, and hash mismatch. Test mutations in a temporary directory.

- [ ] **Step 6: Verify and commit assets**

Run `npm run check:assets`, the Kenney test, `npm test`, and `git diff --check`.

Commit:

```sh
git add public/assets/kenney scripts/check-assets.mjs tests/kenney-assets.test.mjs THIRD_PARTY_NOTICES.md
git commit -m "feat: add curated Kenney scene assets"
```

---

### Task 3: Add the typed cached asset layer and fallback

**Files:**
- Create: `src/game/kenneyAssets.ts`
- Create: `src/game/kenneyLoader.ts`
- Create: `tests/kenney-loader.test.mjs`

**Interfaces:**
- Produces: `KENNEY_ASSETS`, `GameAssetRole`, `GameAssetLibrary`, and `loadKenneyAssets(loader?)`.
- Consumes: semantic roles and paths from Task 2.

- [ ] **Step 1: Write registry and loader tests**

Require exact hazard mappings, finite logical transforms, and explicit native log fallback. Inject a fake asynchronous loader and assert each URL loads once, templates clone independently, and one rejected role is captured without rejecting the full library.

```js
assert.equal(KENNEY_ASSETS['hazard.car'].path, 'assets/kenney/car-kit/sedan.glb');
assert.equal(NATIVE_ASSET_ROLES['hazard.log'], 'procedural-log');
const library = await loadKenneyAssets(fakeLoader);
assert.equal(library.failures.get('hazard.train'), 'offline');
assert.equal(library.templates.has('hazard.car'), true);
```

- [ ] **Step 2: Run tests and confirm RED**

Run `node --experimental-strip-types --test tests/kenney-loader.test.mjs`.

- [ ] **Step 3: Implement the semantic registry**

Define the nine-role union from Task 2. Each entry owns its public path, target logical size, rotation, and vertical offset. Keep all asset transforms in this registry.

- [ ] **Step 4: Implement cached GLB loading**

Use `GLTFLoader` from `three/addons/loaders/GLTFLoader.js` behind an injectable `(url) => Promise<THREE.Group>`. Cache promises by path, require a finite non-empty `Box3`, center/scale a template clone to its logical footprint, and return:

```ts
export type GameAssetLibrary = {
  templates: ReadonlyMap<GameAssetRole, THREE.Group>;
  failures: ReadonlyMap<GameAssetRole, string>;
  clone(role: GameAssetRole): THREE.Group | null;
};
```

Never reject the entire library because one role fails.

- [ ] **Step 5: Verify and commit**

Run the loader test, `npm test`, and `npm run build`.

Commit:

```sh
git add src/game/kenneyAssets.ts src/game/kenneyLoader.ts tests/kenney-loader.test.mjs
git commit -m "feat: load Kenney game assets with fallback"
```

---

### Task 4: Refactor the renderer and apply the Kenney scene

**Files:**
- Create: `src/game/scenery.ts`
- Create: `src/game/createGameRenderer.ts`
- Create: `tests/game-scenery.test.mjs`
- Modify: `src/components/GameScene.tsx`
- Modify: `src/game/rendering.ts`

**Interfaces:**
- Consumes: `GameAssetLibrary` and existing `selectRenderableInstances(state)`.
- Produces: `createGameRenderer(element, options)` with `render`, `resize`, `dispose`, and `assetStatus`; produces deterministic `decorationsForRow(seed, row, laneKind)`.

- [ ] **Step 1: Write deterministic scenery tests**

Require identical output for equal seed/row, bounded output for every lane, no decoration in the center five playable columns, and lane-appropriate roles.

```js
assert.deepEqual(
  decorationsForRow('visual-seed', 12, 'grass'),
  decorationsForRow('visual-seed', 12, 'grass'),
);
assert.ok(
  decorationsForRow('visual-seed', 12, 'grass')
    .every((item) => Math.abs(item.column) > 2),
);
```

- [ ] **Step 2: Run tests and confirm RED**

Run `node --experimental-strip-types --test tests/game-scenery.test.mjs`.

- [ ] **Step 3: Implement pure visual scenery selection**

Use the existing deterministic hash utilities. Return visual descriptors only; never modify lane or collision data. Limit per-row decorations and reserve the gameplay corridor.

- [ ] **Step 4: Move imperative Three ownership out of React**

Move scene/camera/lights, pooled native geometry, Kenney pools, resize, hopping interpolation, camera tracking, rendering, and disposal into `createGameRenderer.ts`.

The renderer must preserve current lane/hazard capacity, use Kenney road/track/car/truck/train/scenery when available, retain native river/log/fly geometry, reuse hidden pooled groups, and expose `loading`, `ready`, or `fallback`. It must not dispose a cached source template while clones are active.

- [ ] **Step 5: Reduce `GameScene` to the lifecycle boundary**

Keep current state/event props and add:

```ts
type GameSceneProps = {
  state: GameState;
  events: readonly GameEvent[];
  onAssetStatus?: (status: GameAssetStatus) => void;
};
```

Create one renderer per host, feed it the latest props, and dispose on unmount.

- [ ] **Step 6: Verify simulation invariants and fallback**

Run the scenery, world, and simulation tests, then `npm test` and `npm run build`. In local preview, force one GLB request to 404 and confirm the scene remains playable with `Using geometric fallback`.

- [ ] **Step 7: Commit**

```sh
git add src/game/scenery.ts src/game/createGameRenderer.ts src/game/rendering.ts src/components/GameScene.tsx tests/game-scenery.test.mjs
git commit -m "refactor: render crossing with Kenney scene assets"
```

---

### Task 5: Build the shared laboratory UI

**Files:**
- Create: `src/components/lab/LabShell.tsx`
- Create: `src/components/lab/LabHeader.tsx`
- Create: `src/components/lab/ExperimentControls.tsx`
- Create: `src/components/lab/PanelFrame.tsx`
- Create: `src/components/lab/TelemetryStrip.tsx`
- Create: `src/hooks/useAtlas.ts`
- Create: `src/simulation/activity.ts`
- Create: `tests/simulation-activity.test.mjs`
- Modify: `src/App.tsx`
- Modify: `src/BiomechanicsDebugApp.tsx`
- Modify: `src/hooks/useSimulationStation.ts`
- Modify: `src/style.css`
- Modify: `tests/presentation.test.mjs`
- Modify: `tests/simulation-station-hook.test.mjs`

**Interfaces:**
- Consumes: existing `GameScene`, `BrainScene`, browser game hook, and biomechanical station hook.
- Produces: shared lab components and `neuralActivityFrame(activity) -> ActivityFrame | null`.

- [ ] **Step 1: Write failing presentation tests**

Replace old light-layout assertions while retaining attribution, 44-pixel targets, live status, mobile stacking, and brain checks:

```js
assert.match(css, /--color-canvas:\s*#08110f/i);
assert.match(css, /--color-neural:\s*#62d9ff/i);
assert.match(css, /grid-template-columns:\s*minmax\(0,\s*7fr\)\s+minmax\(320px,\s*5fr\)/i);
assert.match(app + biomechanics, /LabShell/);
assert.match(biomechanics, /BrainScene/);
assert.match(css, /@media\s*\(prefers-reduced-motion:\s*reduce\)/i);
```

- [ ] **Step 2: Write neural activity conversion tests**

Test null, empty updates, body-ID mapping, duplicate IDs, and non-finite values:

```js
assert.deepEqual(
  neuralActivityFrame({
    revision: 3,
    simulationTime: 1.25,
    updates: [{ neuronId: 42, value: 0.75 }],
  }),
  { time: 1.25, values: [[42, 0.75]] },
);
```

Duplicate or invalid IDs return `null` rather than inventing display values.

- [ ] **Step 3: Run targeted tests and confirm RED**

Run `tests/presentation.test.mjs` and `tests/simulation-activity.test.mjs` with the Node test runner.

- [ ] **Step 4: Extract atlas loading and shared primitives**

Move the abort-safe atlas effect into:

```ts
export function useAtlas(): {
  atlas: Atlas | null;
  error: string | null;
};
```

Implement `PanelFrame` with `title`, `badge`, `children`, `footer`, and `className`. Keep header, controls, and telemetry presentational with explicit props.

- [ ] **Step 5: Compose the normal application through `LabShell`**

Keep current autonomous/manual behavior. Move policy upload and scientific scope to the secondary drawer. Surface Kenney loading/fallback status. Keep train warnings and manual controls beside the game.

- [ ] **Step 6: Compose the biomechanical entry through the same shell**

Use `useAtlas` and `neuralActivityFrame` to render `BrainScene`. Show requested action/key, transport, station phase, last contact, and snapshot. Label absent streams `Awaiting runtime telemetry`; never infer them.

Use:

```ts
const SERVER_URL = import.meta.env.VITE_SIMULATION_URL
  ?? 'ws://127.0.0.1:8001/api/simulation';
```

- [ ] **Step 7: Implement the responsive visual system**

Apply exact dark tokens, 7:5 desktop split, compact rail, telemetry strip, 14-pixel panels, 44-pixel targets, 760-pixel mobile stack, and `prefers-reduced-motion`. Prevent overflow at 360 pixels.

- [ ] **Step 8: Verify browser behavior**

Run `npm test` and `npm run build`. Inspect `/` and `/?biomechanics=1` at 1440×900, 1024×768, and 360×800. Check focus order, long statuses, fallback, brain visibility, and overlap.

- [ ] **Step 9: Commit**

```sh
git add src/App.tsx src/BiomechanicsDebugApp.tsx src/components/lab \
  src/hooks/useAtlas.ts src/hooks/useSimulationStation.ts \
  src/simulation/activity.ts src/style.css \
  tests/presentation.test.mjs tests/simulation-activity.test.mjs \
  tests/simulation-station-hook.test.mjs
git commit -m "feat: unify the low-poly laboratory interface"
```

---

### Task 6: Refresh documentation and verify the deliverable

**Files:**
- Modify: `README.md`
- Modify: `docs/HANDOFF.md`
- Modify: `docs/CONTINUATION_PROMPT.md`
- Modify: `public/THIRD_PARTY_NOTICES.md` if it differs from the root notice

**Interfaces:**
- Consumes: verified Tasks 1–5.
- Produces: accurate instructions, provenance, limitations, and fresh evidence.

- [ ] **Step 1: Update documentation**

Document default and biomechanical views, development server, `VITE_SIMULATION_URL`, Kenney sources, the now-present physical world/server bridge, four missing controller sizes, and any native stream still absent. Add test counts only after verification.

- [ ] **Step 2: Run the frontend gate**

Run `npm test`, `npm run check:assets`, and `npm run build`. Every command must exit 0.

- [ ] **Step 3: Run the Python gate outside the repository**

```sh
cd python
/tmp/fly-crossy-lock-py312/bin/python -m pytest -q \
  --basetemp=/tmp/fly-crossy-final-tests
cd ..
```

Expected: zero failures and no generated repository files.

- [ ] **Step 4: Run CPU Docker smoke and shut it down**

```sh
status_code=0
docker compose up --build --wait || status_code=$?
if [ "$status_code" -eq 0 ]; then
  bash scripts/docker-smoke.sh || status_code=$?
fi
docker compose down || status_code=$?
test "$status_code" -eq 0
```

- [ ] **Step 5: Run final hygiene checks**

Run `git diff --check`; require `git ls-files 'python/.pytest-tmp/**'` to be empty and `python/.pytest-tmp` absent. Inspect for archives, generated runs, missing notices, or automated coauthor trailers.

- [ ] **Step 6: Commit verified documentation**

```sh
git add README.md docs/HANDOFF.md docs/CONTINUATION_PROMPT.md public/THIRD_PARTY_NOTICES.md
git commit -m "docs: document the refreshed laboratory"
```

Do not push or create a pull request without explicit user instruction.
