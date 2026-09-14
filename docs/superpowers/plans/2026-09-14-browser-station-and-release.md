# Browser Hexapod Station and Public Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved three-panel browser station, connect physical key confirmations to the deterministic game, verify responsive CPU/GPU behavior, and publish a clean public repository.

**Architecture:** React coordinates one protocol-v2 simulation client and the existing deterministic game reducer. Three.js renders separate game, MaleCNS, and body/keyboard views from authoritative snapshots; a contact-gated reducer applies directions only from matching confirmed action results and exposes every requested/actual state in accessible telemetry.

**Tech Stack:** React 19, TypeScript 5.9, Three.js 0.186, Vite 8, Node test runner, Playwright-compatible browser checks, Docker Compose, Git/GitHub CLI.

**Spec:** `docs/superpowers/specs/2026-09-14-multiscale-biomechanical-keyboard-design.md`

## Global Constraints

- Game, brain, and biomechanical fly/keyboard panels remain simultaneously visible.
- The selector exposes exactly 80, 1,000, 5,000, 20,000, and 124,289.
- Requested and physically confirmed keys are distinct; activity is labeled `actividad simulada`.
- A directional game step occurs only for a matching confirmed action result.
- Failed/waited results advance as `wait`; stale, unknown, or out-of-order IDs pause visibly.
- Mode change/reset atomically resets brain, motor, body, keyboard, and game.
- Mobile order is game, keyboard/body, brain, telemetry.
- The UI reports `GPU`, `CPU`, or `CPU lento` and never claims real time when measured speed is lower.
- The public repository contains no credentials, caches, downloaded data, temporary visual-companion files, machine paths, or automated-agent coauthor trailers.

---

## File map

- `src/simulation/station.ts` — pure coordinated reducer and action-contact authority.
- `src/hooks/useSimulationStation.ts` — React lifecycle around protocol-v2 client.
- `src/components/PopulationSelector.tsx` — five-mode segmented selector and reset warning.
- `src/components/BiomechanicsScene.tsx` — interpolated body, joint, keyboard, and contact rendering.
- `src/components/StationTelemetry.tsx` — requested/actual key, motor, backend, rates, and latency.
- `src/components/BrainScene.tsx` — keyframe/delta activity application and stale clearing.
- `src/App.tsx` — approved station composition.
- `src/style.css` — desktop/mobile layout, accessibility, and reduced motion.
- `tests/station-reducer.test.mjs` — causal gating and reset tests.
- `tests/activity-stream.test.mjs` — neural keyframe/delta reconstruction tests.
- `tests/presentation.test.mjs` — semantic labels and responsive structure.
- `tests/e2e/station.spec.ts` — browser flow against deterministic simulation fixture.
- `README.md`, `docs/ARCHITECTURE.md`, `docs/PROTOCOL.md` — public setup and scientific limits.
- `scripts/release-audit.sh` — secrets, artifacts, history, test, and clean-clone checks.

### Task 1: Contact-gated station reducer

**Files:**
- Create: `src/simulation/station.ts`
- Create: `tests/station-reducer.test.mjs`
- Modify: `src/hooks/useGame.ts`

**Interfaces:**
- Consumes: protocol `intention`, `contact`, `action_result`, `reset_complete`, game `stepGame`.
- Produces: `createStation(seed)`, `reduceStation(state, event) -> StationState`, and `StationFault`.

- [ ] **Step 1: Write failing causal and idempotency tests**

```js
test('intention alone never advances the game', () => {
  const initial = createStation('gate-test');
  const requested = reduceStation(initial, intention('i-1', 'forward'));
  assert.equal(requested.game.step, 0);
});

test('matching confirmed result advances exactly once', () => {
  const requested = reduceStation(createStation('gate-test'), intention('i-1', 'forward'));
  const confirmed = reduceStation(requested, actionResult('i-1', 'confirmed', 'forward'));
  const duplicate = reduceStation(confirmed, actionResult('i-1', 'confirmed', 'forward'));
  assert.equal(confirmed.game.step, 1);
  assert.equal(duplicate.game.step, 1);
});

test('failed press advances as wait', () => {
  const requested = reduceStation(createStation('gate-test'), intention('i-2', 'left'));
  const failed = reduceStation(requested, actionResult('i-2', 'failed', 'wait'));
  assert.equal(failed.game.step, 1);
  assert.equal(failed.lastAppliedAction, 'wait');
});
```

- [ ] **Step 2: Run and verify failure**

Run: `node --experimental-strip-types --test tests/station-reducer.test.mjs`  
Expected: FAIL because `station.ts` is absent.

- [ ] **Step 3: Implement explicit pending-intention authority**

```ts
export type StationState = {
  game: GameState;
  phase: 'connecting' | 'ready' | 'acting' | 'paused' | 'error';
  population: PopulationSize;
  pending: { id: string; requestedAction: Action; requestedKey: KeyName } | null;
  completedIds: readonly string[];
  lastContact: ContactMessage | null;
  lastAppliedAction: Action | null;
  error: string | null;
};
```

Only `action_result` with matching current session/episode/intention and `confirmed` may apply its direction. `failed` and `waited` apply `wait`. Keep 256 completion IDs. Any mismatched action/key/result pauses with a stable user-visible error.

- [ ] **Step 4: Add coordinated reset tests and implementation**

```js
test('reset_complete replaces all five state domains', () => {
  const reset = reduceStation(actingStation(), resetComplete('e-next0001', 1000));
  assert.equal(reset.game.step, 0);
  assert.equal(reset.pending, null);
  assert.equal(reset.lastContact, null);
  assert.deepEqual(reset.activity.values, []);
  assert.equal(reset.biomechanics.motorPhase, 'neutral');
});
```

- [ ] **Step 5: Run browser tests and commit**

Run: `npm test`  
Expected: PASS.

```bash
git add src/simulation/station.ts src/hooks/useGame.ts tests/station-reducer.test.mjs
git commit -m "feat: gate game steps on physical key results"
```

### Task 2: React simulation station lifecycle

**Files:**
- Create: `src/hooks/useSimulationStation.ts`
- Create: `src/components/PopulationSelector.tsx`
- Create: `tests/simulation-station-hook.test.mjs`

**Interfaces:**
- Consumes: `SimulationClient`, station reducer.
- Produces: `useSimulationStation({seed, population, backendPreference})` and `PopulationSelector`.

- [ ] **Step 1: Write failing mode-switch lifecycle test**

```js
test('population switch waits for reset_complete', async () => {
  const harness = createStationHarness({population: 80});
  harness.selectPopulation(1000);
  assert.equal(harness.state.population, 80);
  assert.equal(harness.state.phase, 'connecting');
  harness.server(resetComplete('e-next0001', 1000));
  assert.equal(harness.state.population, 1000);
  assert.equal(harness.state.game.step, 0);
});
```

- [ ] **Step 2: Run and verify failure**

Run: `node --experimental-strip-types --test tests/simulation-station-hook.test.mjs`  
Expected: FAIL because the station hook does not exist.

- [ ] **Step 3: Implement socket, visibility, and reset lifecycle**

```ts
export const POPULATIONS = [80, 1_000, 5_000, 20_000, 124_289] as const;

export type UseSimulationStationOptions = {
  seed: string;
  population: PopulationSize;
  backendPreference: BackendPreference;
};
```

Open one client per mounted station, pause on hidden tab, abort old event handlers on unmount, clear stale activity/contact immediately on disconnect, and wait for `ready`/`reset_complete` before sending an observation. Confirm slow-mode warning before requesting a large CPU switch.

- [ ] **Step 4: Implement the accessible segmented selector**

Use a labeled radiogroup, buttons with `aria-checked`, visible focus, disabled transition state, thousands separators, and a text warning when effective speed is below 1.0×.

- [ ] **Step 5: Run tests/build and commit**

Run: `npm test && npm run build`  
Expected: PASS.

```bash
git add src/hooks/useSimulationStation.ts src/components/PopulationSelector.tsx tests/simulation-station-hook.test.mjs
git commit -m "feat: coordinate five neural modes in browser"
```

### Task 3: Render the fly body and physical keyboard

**Files:**
- Create: `src/components/BiomechanicsScene.tsx`
- Create: `src/simulation/biomechanics.ts`
- Create: `tests/biomechanics-render.test.mjs`

**Interfaces:**
- Consumes: protocol `snapshot` and `contact` messages.
- Produces: `BiomechanicsFrameBuffer.push(snapshot)`, `sample(time)`, and `BiomechanicsScene`.

- [ ] **Step 1: Write failing interpolation and no-fabrication tests**

```js
test('frame buffer interpolates transforms but preserves discrete contact', () => {
  const frames = new BiomechanicsFrameBuffer();
  frames.push(snapshot(1, [0, 0, 0], null));
  frames.push(snapshot(2, [2, 0, 0], null));
  assert.deepEqual(frames.sample(1.5).bodyPosition, [1, 0, 0]);
  assert.equal(frames.sample(1.5).confirmedKey, null);
});
```

- [ ] **Step 2: Run and verify failure**

Run: `node --experimental-strip-types --test tests/biomechanics-render.test.mjs`  
Expected: FAIL because the frame buffer is absent.

- [ ] **Step 3: Implement bounded snapshot interpolation**

Keep at most 90 snapshots, normalize quaternions, interpolate positions/joints, reject non-monotonic timestamps and wrong 59-joint shape, and never derive contact/key confirmation from geometry.

- [ ] **Step 4: Build the Three.js keyboard/body view**

Create recognizable W/A/S/D/Space keycaps, articulated fly body geometry bound to snapshot transforms, target-leg highlight, requested-key outline, confirmed-key fill, and force/travel indicators. Use one renderer per mount, capped pixel ratio, ResizeObserver, and complete geometry/material/renderer disposal.

- [ ] **Step 5: Run tests/build and commit**

Run: `npm test && npm run build`  
Expected: PASS.

```bash
git add src/components/BiomechanicsScene.tsx src/simulation/biomechanics.ts tests/biomechanics-render.test.mjs
git commit -m "feat: render biomechanical fly keyboard station"
```

### Task 4: Stream all five brain populations safely

**Files:**
- Create: `src/simulation/activity.ts`
- Create: `tests/activity-stream.test.mjs`
- Modify: `src/components/BrainScene.tsx`
- Modify: `src/lib/atlas.ts`

**Interfaces:**
- Consumes: chunked `neural_keyframe`, `neural_delta`, graph manifest counts.
- Produces: `ActivityStore.apply(frame)`, `clear()`, and changed-index uploads for `BrainScene`.

- [ ] **Step 1: Write failing chunk/keyframe/delta tests**

```js
test('activity store rejects delta with unknown base', () => {
  const store = new ActivityStore(atlasIds);
  assert.throws(() => store.apply(delta({baseSequence: 4, sequence: 5})), /keyframe/i);
});

test('disconnect clears all simulated values', () => {
  const store = seededActivityStore();
  store.clear();
  assert.equal(store.activeCount, 0);
  assert.equal(store.stale, true);
});
```

- [ ] **Step 2: Run and verify failure**

Run: `node --experimental-strip-types --test tests/activity-stream.test.mjs`  
Expected: FAIL because `ActivityStore` is absent.

- [ ] **Step 3: Implement bounded atlas-ID reconstruction**

Map body IDs through the atlas's existing validated index. Reject duplicates, unknown IDs, non-finite values, excessive entries, and delta gaps. Assemble chunked keyframes atomically before exposing them. Return only changed atlas indices for GPU buffer updates.

- [ ] **Step 4: Update BrainScene without reallocating per frame**

Keep soma geometry static, update the activity attribute ranges in place, reduce visual density/effects under performance pressure without hiding population membership, and show connected/isolated counts from the verified graph manifest. Display `ACTIVIDAD SIMULADA` and clear colors on stale/disconnect.

- [ ] **Step 5: Run activity and existing atlas tests**

Run: `npm test && npm run check:assets && npm run build`  
Expected: PASS.

- [ ] **Step 6: Commit multiscale brain rendering**

```bash
git add src/simulation/activity.ts tests/activity-stream.test.mjs src/components/BrainScene.tsx src/lib/atlas.ts
git commit -m "feat: visualize five streamed neural populations"
```

### Task 5: Approved layout, telemetry, and accessibility

**Files:**
- Create: `src/components/StationTelemetry.tsx`
- Modify: `src/App.tsx`
- Modify: `src/style.css`
- Modify: `tests/presentation.test.mjs`

**Interfaces:**
- Consumes: station, backend metrics, contact, motor, and activity state.
- Produces: the approved desktop station and mobile stack.

- [ ] **Step 1: Add failing semantic presentation assertions**

```js
test('station exposes required labels and all modes', () => {
  const source = readFileSync('src/App.tsx', 'utf8');
  for (const label of ['Crossy Road', 'Actividad MaleCNS', 'Mosca + teclado']) assert.match(source, new RegExp(label, 'i'));
  for (const size of ['80', '1,000', '5,000', '20,000', '124,289']) assert.match(source, new RegExp(size.replaceAll(',', ',')));
  assert.match(source, /actividad simulada/i);
});
```

- [ ] **Step 2: Run and verify presentation failure**

Run: `node --experimental-strip-types --test tests/presentation.test.mjs`  
Expected: FAIL until the new station composition exists.

- [ ] **Step 3: Compose the three simultaneous panels**

Place the selector/backend strip first, Crossy Road in the primary panel, MaleCNS and biomechanics as equal evidence panels, then telemetry with target, requested key, actual contact, force, travel, motor phase, result, latency, physics Hz, render FPS, and effective simulation speed.

- [ ] **Step 4: Implement responsive and reduced-motion rules**

At desktop widths use a 12-column station grid. At widths below 760px set DOM/CSS order to game, biomechanics, brain, telemetry. Preserve focus outlines, `aria-live="polite"` for status, `role="alert"` for faults, non-color status text, minimum 44px controls, and disable nonessential tweens under `prefers-reduced-motion`.

- [ ] **Step 5: Run semantic, build, and manual viewport QA**

Run: `npm test && npm run build`  
Expected: PASS; verify 1440×900, 1024×768, 390×844, and 360×800 with no hidden panel or horizontal overflow.

- [ ] **Step 6: Commit the approved station UI**

```bash
git add src/components/StationTelemetry.tsx src/App.tsx src/style.css tests/presentation.test.mjs
git commit -m "feat: deliver responsive hexapod station ui"
```

### Task 6: End-to-end browser and Docker verification

**Files:**
- Create: `playwright.config.ts`
- Create: `tests/e2e/station.spec.ts`
- Create: `tests/fixtures/simulation-server.ts`
- Modify: `package.json`
- Modify: `package-lock.json`
- Modify: `scripts/docker-smoke.sh`

**Interfaces:**
- Produces: `npm run test:e2e` and contact-gated deterministic browser fixtures.

- [ ] **Step 1: Add browser-test tooling and failing scenario**

```ts
test('requested forward waits for confirmed W contact', async ({page}) => {
  await page.goto('/');
  await page.getByRole('radio', {name: '1,000'}).click();
  await expect(page.getByTestId('game-step')).toHaveText('0');
  await fixture.sendIntention('i-1', 'forward', 'W');
  await expect(page.getByTestId('game-step')).toHaveText('0');
  await fixture.sendConfirmedResult('i-1', 'forward', 'W');
  await expect(page.getByTestId('game-step')).toHaveText('1');
});
```

Add failed-press/wait, duplicate result, mode switch/reset, disconnect clear/pause, slow CPU badge, and all-five-mode scenarios.

- [ ] **Step 2: Run and verify failure**

Run: `npm run test:e2e`  
Expected: FAIL before the fixture server/config is complete.

- [ ] **Step 3: Implement deterministic fixture server and browser config**

Bind fixture and preview servers to loopback, allocate fixed test ports, reset between tests, retain all critical events, and produce screenshot/trace only on failure.

- [ ] **Step 4: Run the complete local and container matrix**

Run: `npm test && npm run test:e2e && npm run check:assets && npm run build && cd python && pytest -q && cd .. && docker compose up --build --wait && bash scripts/docker-smoke.sh && docker compose down`  
Expected: every command exits 0.

- [ ] **Step 5: Commit end-to-end verification**

```bash
git add playwright.config.ts tests/e2e tests/fixtures/simulation-server.ts package.json package-lock.json scripts/docker-smoke.sh
git commit -m "test: verify physical fly game flow end to end"
```

### Task 7: Public documentation and release audit

**Files:**
- Modify: `README.md`
- Create: `docs/ARCHITECTURE.md`
- Create: `docs/PROTOCOL.md`
- Create: `docs/HARDWARE.md`
- Modify: `ATTRIBUTION.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Create: `scripts/release-audit.sh`
- Modify: `.gitignore`

**Interfaces:**
- Produces: reproducible CPU/GPU quick starts, causal architecture documentation, clean-clone audit, and publish-ready repository.

- [ ] **Step 1: Write the release audit script**

```bash
#!/usr/bin/env bash
set -euo pipefail
git diff --exit-code
git diff --cached --exit-code
test -z "$(git ls-files | grep -E '(^|/)(\.env|\.local|__pycache__|node_modules|\.superpowers)(/|$)' || true)"
test -z "$(git log --format=%B | grep -i '^Co-authored-by:.*\(codex\|openai\|chatgpt\)' || true)"
npm test
npm run check:assets
npm run build
(cd python && pytest -q)
docker compose config >/dev/null
```

- [ ] **Step 2: Run audit and record current documentation failures**

Run: `bash scripts/release-audit.sh`  
Expected: FAIL until documentation/audit exclusions are complete or PASS with a list of still-unwritten required documents checked separately.

- [ ] **Step 3: Write the public documentation**

README sections, in order: verified demo, scientific-status warning, architecture/action flow, CPU quick start, NVIDIA GPU quick start, five-mode table, hardware expectations including slow 124,289 CPU mode, controls, tests, data retrieval, training/evaluation, protocol/artifact versions, reproducibility, results generated from release manifests, limitations, licenses/credits/citation.

Use this causal diagram exactly in meaning:

```text
Game observation -> selected MaleCNS controller -> intention
    -> shared 59-joint motor -> MuJoCo key contact
    -> confirmed action (or wait on failure) -> game step
```

Document that activity is simulated and graph size does not imply biological fidelity or better performance. Link result numbers only to verified release artifacts.

- [ ] **Step 4: Audit tracked files, history, notices, and clean clone**

Run: `bash scripts/release-audit.sh && audit_dir="$(mktemp -d)" && git clone --no-local . "$audit_dir/repo" && cd "$audit_dir/repo" && npm ci && npm test && npm run build && cd python && python -m pip install -e '.[test]' && pytest -q`  
Expected: all commands exit 0 and the temporary clone needs no untracked project data for tests/build.

- [ ] **Step 5: Commit the release documentation**

```bash
git add README.md docs/ARCHITECTURE.md docs/PROTOCOL.md docs/HARDWARE.md ATTRIBUTION.md THIRD_PARTY_NOTICES.md scripts/release-audit.sh .gitignore
git commit -m "docs: prepare verified public fly release"
```

### Task 8: Create and verify the public GitHub repository

**Files:**
- No code files; this task changes the external GitHub remote after every prior checkpoint passes.

**Interfaces:**
- Consumes: clean verified branch and authenticated GitHub CLI session.
- Produces: public `fly-crossy-connectome` repository with verified default branch and README.

- [ ] **Step 1: Re-run verification immediately before publication**

Run: `bash scripts/release-audit.sh && git status --short --branch && git log -5 --format='%h %s%n%B'`  
Expected: audit exits 0, worktree is clean, and no automated-agent coauthor trailer appears.

- [ ] **Step 2: Inspect authenticated account and existing target without mutation**

Run: `gh auth status && gh repo view "$(gh api user --jq .login)/fly-crossy-connectome" --json nameWithOwner,visibility,url 2>/dev/null || true`  
Expected: authentication identifies the user's account; if the target exists, stop and inspect its ownership/history rather than overwriting it.

- [ ] **Step 3: Create the missing public repository and push the verified branch**

Run only when the target is absent:

```bash
owner="$(gh api user --jq .login)"
gh repo create "$owner/fly-crossy-connectome" --public --source=. --remote=publish
git push -u publish HEAD:main
gh repo edit "$owner/fly-crossy-connectome" --default-branch main
```

If the target already exists and is owned by the authenticated user, add a separate `publish` remote only after confirming its default branch and unrelated history are safe; never force-push.

- [ ] **Step 4: Verify the public result**

Run: `owner="$(gh api user --jq .login)" && gh repo view "$owner/fly-crossy-connectome" --json nameWithOwner,visibility,url,defaultBranchRef --jq '{nameWithOwner,visibility,url,defaultBranch:.defaultBranchRef.name}' && gh api "repos/$owner/fly-crossy-connectome/readme" --jq .html_url`  
Expected: visibility is `PUBLIC`, the default branch points to the verified commit, and README is accessible.

## Plan completion checkpoint

Run: `bash scripts/release-audit.sh`  
Expected: all native/container verification passes, the repository is clean, and the public GitHub URL resolves to the verified default branch without automated-agent coauthor attribution.
