# Fly Crossy Connectome Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a polished seeded 3D Crossy Road-style fly game with human, conventional-policy, and reduced-connectome controller modes beside a synchronized MaleCNS brain view.

**Architecture:** A pure TypeScript deterministic simulation is the authority for world state, observations, actions, collisions, score, and reward. React owns orchestration and controls while independent Three.js views render the game and measured brain anatomy; Python mirrors the controller contract for training and exports versioned browser models.

**Tech Stack:** React 19, TypeScript 5.9, Three.js 0.186, Vite 8, Node test runner, Python 3.11+, NumPy, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-fly-crossy-connectome-design.md`

## Global Constraints

- Keep the required linked `fly-connectome-template` credit in the web UI and repository README and identify this project as modified.
- Preserve MaleCNS and Flybody notices and licenses.
- Use one authoritative fixed-step simulation clock; rendering never determines outcomes.
- Keep the game and brain visualization simultaneously visible on desktop and narrow screens.
- Label atlas coordinates as measured anatomy and controller output as simulated activity.
- Human mode emits no neural activity values.
- Use actions `forward`, `backward`, `left`, `right`, and `wait` for every controller.
- A world version, seed, and action trace must reproduce the same outcome.
- Full MaleCNS execution is an extension point, not a first-release dependency.

---

## File Map

### Browser simulation

- `src/game/types.ts` — stable game, lane, hazard, action, and event types.
- `src/game/random.ts` — deterministic string-seed hashing and PRNG.
- `src/game/world.ts` — chunk/lane generation and solvability invariants.
- `src/game/simulation.ts` — authoritative fixed-step state transition and collision rules.
- `src/game/observation.ts` — bounded egocentric controller observation.
- `src/game/reward.ts` — shared reward calculation.
- `src/game/model.ts` — versioned exported-policy parser and validator.
- `src/game/controllers.ts` — controller interface, manual, scripted, and browser model adapters.
- `src/game/remote.ts` — versioned WebSocket adapter contract for future full-MaleCNS services.

### Browser presentation

- `src/components/GameScene.tsx` — Three.js low-poly renderer and camera.
- `src/components/GameControls.tsx` — keyboard/touch input and run controls.
- `src/components/Telemetry.tsx` — synchronized provenance, action, reward, and score.
- `src/components/BrainScene.tsx` — retain atlas renderer; add explicit empty/manual labeling hooks only.
- `src/hooks/useGame.ts` — simulation clock, controller scheduling, pause/reset, and trace state.
- `src/App.tsx` — laboratory composition and model loading.
- `src/style.css` — responsive two-panel game/brain layout and visual system.

### Training and evaluation

- `python/pyproject.toml` — Python package, test, NumPy, and PyTorch requirements.
- `python/fly_crossy/schema.py` — Python equivalents of observation/action/model schemas.
- `python/fly_crossy/env.py` — deterministic vectorized training environment contract.
- `python/fly_crossy/models.py` — conventional and fixed-topology recurrent policies.
- `python/fly_crossy/train.py` — PPO entry point, run metadata, and checkpoints.
- `python/fly_crossy/connectome.py` — versioned MaleCNS edge filtering and reduced graph artifact.
- `python/fly_crossy/export.py` — compact model export with hashes and provenance.
- `python/fly_crossy/evaluate.py` — held-out seed suite, controls, and metrics.

### Tests and fixtures

- `tests/game-world.test.mjs` — seeded generator and invariants.
- `tests/game-simulation.test.mjs` — movement, collision, platform, score, and replay.
- `tests/game-observation.test.mjs` — observation bounds and encoding.
- `tests/game-model.test.mjs` — browser policy validation/inference fixtures.
- `tests/game-controller.test.mjs` — controller lifecycle and failure handling.
- `tests/fixtures/episode-v1.json` — cross-language fixed episode.
- `tests/fixtures/policy-v1.json` — compact inference fixture.
- `python/tests/test_schema.py` — schema parity.
- `python/tests/test_env.py` — fixture episode parity.
- `python/tests/test_models.py` — topology and output shapes.
- `python/tests/test_export.py` — browser export format.

---

### Task 1: Deterministic world contracts

**Files:**
- Create: `src/game/types.ts`
- Create: `src/game/random.ts`
- Create: `src/game/world.ts`
- Test: `tests/game-world.test.mjs`

**Interfaces:**
- Produces: `Action`, `Lane`, `Hazard`, `WorldChunk`, `hashSeed(seed: string): number`, `createRng(seed: string): Rng`, and `generateRows(seed: string, from: number, count: number): Lane[]`.
- Invariant: lane IDs equal row coordinates and generation depends only on world version, seed, and row.

- [ ] **Step 1: Write failing seeded-generation tests**

```js
test('same seed and range produce identical lanes', () => {
  assert.deepEqual(generateRows('lab-7', -5, 40), generateRows('lab-7', -5, 40));
});
test('chunk boundaries do not change generation', () => {
  assert.deepEqual(generateRows('lab-7', 0, 40), [...generateRows('lab-7', 0, 20), ...generateRows('lab-7', 20, 20)]);
});
test('opening rows are safe and hazard groups include recovery rows', () => {
  const rows = generateRows('lab-7', 0, 100);
  assert.deepEqual(rows.slice(0, 3).map(row => row.kind), ['grass', 'grass', 'grass']);
  assert.ok(rows.every((row, index) => index < 6 || rows.slice(Math.max(0, index - 6), index + 1).some(candidate => candidate.kind === 'grass')));
});
```

- [ ] **Step 2: Run the focused test and confirm missing modules fail**

Run: `node --experimental-strip-types --test tests/game-world.test.mjs`  
Expected: FAIL with module-not-found for `src/game/world.ts`.

- [ ] **Step 3: Implement stable types, PRNG, and row-local generation**

```ts
export type Action = 'forward' | 'backward' | 'left' | 'right' | 'wait';
export type LaneKind = 'grass' | 'road' | 'rail' | 'river';
export type Direction = -1 | 1;
export type Lane = { row: number; kind: LaneKind; direction?: Direction; speed?: number; phase?: number; hazards: Hazard[] };
export type Rng = { next(): number; integer(min: number, max: number): number; pick<T>(values: readonly T[]): T };
```

Use a documented 32-bit hash plus Mulberry32. Derive each lane group from `worldVersion + seed + groupIndex`; never consume a global mutable random stream across chunk requests.

- [ ] **Step 4: Run generator tests and all existing tests**

Run: `npm test`  
Expected: all existing replay tests and new world tests PASS.

- [ ] **Step 5: Commit the deterministic world slice**

```bash
git add src/game/types.ts src/game/random.ts src/game/world.ts tests/game-world.test.mjs
git commit -m "feat: add deterministic crossy world generation"
```

### Task 2: Authoritative simulation, reward, and observation

**Files:**
- Create: `src/game/simulation.ts`
- Create: `src/game/observation.ts`
- Create: `src/game/reward.ts`
- Test: `tests/game-simulation.test.mjs`
- Test: `tests/game-observation.test.mjs`
- Create: `tests/fixtures/episode-v1.json`

**Interfaces:**
- Consumes: `Action`, `Lane`, and `generateRows` from Task 1.
- Produces: `createGame(seed: string): GameState`, `stepGame(state: GameState, action: Action): StepResult`, `observe(state: GameState): ObservationV1`, `reward(previous: GameState, next: GameState): number`.

- [ ] **Step 1: Write failing movement and terminal-event tests**

```js
const initial = createGame('road-test');
const moved = stepGame(initial, 'forward');
assert.equal(moved.state.fly.row, initial.fly.row + 1);
assert.equal(moved.state.step, 1);
assert.equal(moved.state.score, 1);
assert.equal(initial.fly.row, 0, 'stepGame must not mutate input');
```

Add explicit fixtures for a swept vehicle collision, train warning/impact, supported river movement, unsupported water death, horizontal bounds, waiting, and score not decreasing when moving backward.

- [ ] **Step 2: Confirm tests fail before implementation**

Run: `node --experimental-strip-types --test tests/game-simulation.test.mjs tests/game-observation.test.mjs`  
Expected: FAIL with missing exports.

- [ ] **Step 3: Implement immutable fixed-step transitions**

```ts
export type TerminalReason = 'vehicle' | 'train' | 'water' | 'bounds';
export type GameState = { version: 1; seed: string; step: number; time: number; fly: GridPosition; score: number; lanes: Lane[]; terminal: TerminalReason | null };
export type StepResult = { state: GameState; reward: number; events: GameEvent[] };
export const DECISION_SECONDS = 0.2;
```

Resolve moving hazards over `[time, time + DECISION_SECONDS]`, then movement/support and terminal state. Append rows ahead and prune only rows far behind the fly.

- [ ] **Step 4: Implement bounded `ObservationV1` and versioned episode fixture**

```ts
export type ObservationV1 = { version: 1; radius: 5; cells: number[][]; motion: number[][][]; support: 0 | 1; previousAction: Action; edgeDistance: number };
```

Encode only rows/columns within radius five. Write `episode-v1.json` with seed, actions, expected score, rewards, and terminal reason.

- [ ] **Step 5: Run focused and full tests**

Run: `npm test`  
Expected: all simulation, observation, world, and replay tests PASS.

- [ ] **Step 6: Commit the simulation slice**

```bash
git add src/game tests/game-simulation.test.mjs tests/game-observation.test.mjs tests/fixtures/episode-v1.json
git commit -m "feat: add deterministic game simulation"
```

### Task 3: Playable Three.js game and human input

**Files:**
- Create: `src/components/GameScene.tsx`
- Create: `src/components/GameControls.tsx`
- Create: `src/hooks/useGame.ts`
- Modify: `src/App.tsx`
- Modify: `src/style.css`

**Interfaces:**
- Consumes: `GameState`, `Action`, `StepResult`, `createGame`, and `stepGame` from Task 2.
- Produces: `GameScene({ state, events })`, `GameControls({ onAction, paused, onTogglePause })`, and `useGame(options)`.

- [ ] **Step 1: Add a temporary human-mode integration assertion**

Add an exported reducer-level helper and test it without a DOM:

```js
const run = reduceGameCommand({ game: createGame('manual') }, { type: 'action', action: 'forward' });
assert.equal(run.game.fly.row, 1);
```

- [ ] **Step 2: Run the test and confirm the reducer is absent**

Run: `node --experimental-strip-types --test tests/game-controller.test.mjs`  
Expected: FAIL for missing `reduceGameCommand`.

- [ ] **Step 3: Implement the game hook and input lifecycle**

```ts
export type ControllerMode = 'human' | 'conventional' | 'connectome';
export type GameCommand = { type: 'action'; action: Action } | { type: 'reset'; seed?: string } | { type: 'toggle-pause' };
```

Use keydown listeners with `event.repeat` ignored, prevent scrolling for game keys, remove listeners on cleanup, map arrows/WASD to directions, Space to `wait`, and Escape to pause/resume. Touch buttons call the same action function.

- [ ] **Step 4: Build the Three.js scene**

Create one scene per mounted component, an orthographic isometric camera, instanced vehicles/platforms where practical, simple geometry/materials, ResizeObserver sizing, capped device pixel ratio, and complete disposal. Center the camera smoothly on the fly row; interpolate visual hops without changing authoritative state.

- [ ] **Step 5: Replace the example environment with the playable panel**

Update `App.tsx` only enough to mount `GameScene` and `GameControls` with Human mode. Retain `BrainScene`, `Attribution`, atlas loading, and scientific scope content.

- [ ] **Step 6: Verify play and production build**

Run: `npm test && npm run check:assets && npm run build`  
Expected: all commands exit 0; keyboard and touch controls move the fly in local preview.

- [ ] **Step 7: Commit the playable slice**

```bash
git add src/components/GameScene.tsx src/components/GameControls.tsx src/hooks/useGame.ts src/App.tsx src/style.css tests/game-controller.test.mjs
git commit -m "feat: add playable isometric fly crossing game"
```

### Task 4: Controller protocol, model validation, and synchronized telemetry

**Files:**
- Create: `src/game/model.ts`
- Create: `src/game/controllers.ts`
- Create: `src/game/remote.ts`
- Create: `src/components/Telemetry.tsx`
- Modify: `src/hooks/useGame.ts`
- Modify: `src/App.tsx`
- Modify: `src/components/BrainScene.tsx`
- Test: `tests/game-model.test.mjs`
- Test: `tests/game-controller.test.mjs`
- Create: `tests/fixtures/policy-v1.json`

**Interfaces:**
- Consumes: `ObservationV1`, `Action`, `ActivityFrame`, and atlas `visibleIds`.
- Produces: `Controller`, `ControllerDecision`, `parsePolicy`, `createScriptedController`, `createDensePolicyController`, `createRemoteController`, and `Telemetry`.

- [ ] **Step 1: Write failing schema and controller tests**

```js
assert.throws(() => parsePolicy({ version: 99 }, ids));
assert.throws(() => parsePolicy({ ...validPolicy, activityBodyIds: [999] }, ids));
const decision = await createScriptedController(['forward', 'wait']).decide(observation);
assert.equal(decision.action, 'forward');
assert.deepEqual(decision.activity, []);
```

- [ ] **Step 2: Run focused tests and confirm failures**

Run: `node --experimental-strip-types --test tests/game-model.test.mjs tests/game-controller.test.mjs`  
Expected: FAIL with missing policy/controller exports.

- [ ] **Step 3: Implement controller and model contracts**

```ts
export type ControllerDecision = { action: Action; activity: [number, number][]; diagnostics: Record<string, number> };
export interface Controller { readonly id: string; readonly kind: 'scripted' | 'conventional' | 'connectome' | 'remote'; decide(observation: ObservationV1, signal: AbortSignal): Promise<ControllerDecision>; reset(seed: string): void; dispose(): void; }
export type ExportedPolicyV1 = { version: 1; observationVersion: 1; actions: Action[]; source: ModelSource; network: DenseNetwork | FixedGraphNetwork; activityBodyIds: number[] };
```

Validate all array shapes, finite numeric values, supported activations, action order, source metadata, and MaleCNS IDs before constructing a controller.

Define the remote wire messages as `{ type: 'reset', version: 1, seed }`, `{ type: 'observe', version: 1, requestId, observation }`, and `{ type: 'decision', version: 1, requestId, decision }`. `createRemoteController(url)` must match responses by request ID, reject unknown message types or schema versions, enforce one response timeout, and close its socket in `dispose()`.

- [ ] **Step 4: Schedule autonomous decisions on the authoritative clock**

Allow only one pending decision. Use AbortController on reset/mode change. A rejection, timeout, or invalid response pauses the run and records a visible error. Human mode always sends `frame={null}` to `BrainScene`.

- [ ] **Step 5: Add telemetry and mode/model controls**

Display controller kind/name, explicit `MANUAL — NO NEURAL OUTPUT` or `SIMULATED MODEL ACTIVITY`, step, action, step reward, score, seed, status, and model normalization. Add local JSON policy loading with a 10 MB limit.

- [ ] **Step 6: Verify controller failures and activity clearing**

Run: `npm test && npm run build`  
Expected: tests confirm no stale activity after reset/mode change and build exits 0.

- [ ] **Step 7: Commit the laboratory runtime slice**

```bash
git add src/game/model.ts src/game/controllers.ts src/game/remote.ts src/components/Telemetry.tsx src/components/BrainScene.tsx src/hooks/useGame.ts src/App.tsx tests/game-model.test.mjs tests/game-controller.test.mjs tests/fixtures/policy-v1.json
git commit -m "feat: add controller runtime and neural telemetry"
```

### Task 5: Responsive laboratory visual design and project identity

**Files:**
- Modify: `src/style.css`
- Modify: `src/App.tsx`
- Modify: `src/components/Attribution.tsx`
- Modify: `README.md`
- Modify: `package.json`
- Modify: `index.html`

**Interfaces:**
- Consumes: complete browser components from Tasks 3 and 4.
- Produces: final desktop/mobile layout and accurate repository documentation.

- [ ] **Step 1: Rename the app and describe modified-template provenance**

Set package name to `fly-crossy-connectome`, page title to `Fly Crossy Connectome`, and README start/run/test instructions to the commands actually present. Keep the exact linked template credit and all third-party notices.

- [ ] **Step 2: Implement the final responsive layout**

Desktop: game `minmax(0, 3fr)`, brain/telemetry `minmax(340px, 2fr)`. Narrow screens: stack game then brain then telemetry; do not use `display:none` for either major view. Give touch targets a minimum 44 px size and preserve visible keyboard focus.

- [ ] **Step 3: Add performance and reduced-motion behavior**

Cap pixel ratio at 2, disable cosmetic hop easing and brain auto-orbit under `prefers-reduced-motion`, and pause simulation scheduling when `document.hidden` without advancing elapsed game time.

- [ ] **Step 4: Run browser build checks**

Run: `npm test && npm run check:assets && npm run build`  
Expected: every command exits 0 and both panels remain visible at 390 px and 1440 px widths.

- [ ] **Step 5: Commit visual and documentation polish**

```bash
git add src/style.css src/App.tsx src/components/Attribution.tsx README.md package.json package-lock.json index.html
git commit -m "feat: polish the fly crossing laboratory"
```

### Task 6: Python schema and deterministic environment parity

**Files:**
- Create: `python/pyproject.toml`
- Create: `python/fly_crossy/__init__.py`
- Create: `python/fly_crossy/schema.py`
- Create: `python/fly_crossy/env.py`
- Test: `python/tests/test_schema.py`
- Test: `python/tests/test_env.py`
- Modify: `tests/fixtures/episode-v1.json`

**Interfaces:**
- Consumes: Task 2's `ObservationV1`, action order, world version, and fixed episode fixture.
- Produces: `Action`, `ObservationV1`, `FlyCrossyEnv.reset(seed)`, and `FlyCrossyEnv.step(action)` with Gym-style return values.

- [ ] **Step 1: Write failing parity tests**

```python
def test_fixture_episode_matches_browser_contract():
    fixture = json.loads(FIXTURE.read_text())
    env = FlyCrossyEnv()
    observation, info = env.reset(seed=fixture["seed"])
    for expected, action in zip(fixture["steps"], fixture["actions"]):
        observation, reward, terminated, truncated, info = env.step(Action(action))
        assert reward == pytest.approx(expected["reward"])
    assert info["score"] == fixture["expectedScore"]
```

- [ ] **Step 2: Run pytest and confirm missing package failure**

Run: `python -m pytest python/tests/test_schema.py python/tests/test_env.py -q`  
Expected: FAIL because `fly_crossy` does not exist.

- [ ] **Step 3: Implement exact schema and environment parity**

```python
class Action(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    WAIT = "wait"

class FlyCrossyEnv:
    def reset(self, seed: str) -> tuple[np.ndarray, dict[str, object]]:
        self.state = create_game(seed)
        return flatten_observation(observe(self.state)), {"score": 0, "seed": seed}

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        result = step_game(self.state, action)
        self.state = result.state
        return flatten_observation(observe(self.state)), result.reward, self.state.terminal is not None, False, {"score": self.state.score, "terminalReason": self.state.terminal}
```

Port the deterministic hash, row generator, transition order, observation encoding, and reward coefficients exactly; do not approximate them independently.

- [ ] **Step 4: Run cross-language fixture tests**

Run: `npm test && python -m pytest python/tests/test_schema.py python/tests/test_env.py -q`  
Expected: browser and Python consume the same fixture and PASS.

- [ ] **Step 5: Commit environment parity**

```bash
git add python tests/fixtures/episode-v1.json
git commit -m "feat: add Python training environment parity"
```

### Task 7: Conventional PPO baseline and browser export

**Files:**
- Create: `python/fly_crossy/models.py`
- Create: `python/fly_crossy/train.py`
- Create: `python/fly_crossy/export.py`
- Test: `python/tests/test_models.py`
- Test: `python/tests/test_export.py`
- Modify: `src/game/model.ts`
- Modify: `tests/game-model.test.mjs`

**Interfaces:**
- Consumes: flattened `ObservationV1` and five-action order.
- Produces: `DensePolicy`, PPO CLI, `export_policy(checkpoint, output)`, and a model accepted by `parsePolicy`.

- [ ] **Step 1: Write failing model/export tests**

```python
def test_dense_policy_shapes():
    policy = DensePolicy(observation_size=OBS_SIZE, hidden_size=64, actions=5)
    logits, value = policy(torch.zeros(3, OBS_SIZE))
    assert logits.shape == (3, 5)
    assert value.shape == (3,)

def test_export_contains_provenance_and_finite_weights(tmp_path):
    payload = export_policy(checkpoint, tmp_path / "policy.json")
    assert payload["version"] == 1
    assert payload["source"]["kind"] == "predicted"
```

- [ ] **Step 2: Confirm tests fail before implementation**

Run: `python -m pytest python/tests/test_models.py python/tests/test_export.py -q`  
Expected: FAIL with missing exports.

- [ ] **Step 3: Implement compact actor-critic and PPO**

Use two `tanh` hidden layers of width 64, a categorical actor head, and scalar critic head. Expose CLI options for seed, environment count, steps, learning rate, output directory, and device. Save configuration, software versions, training curve, checkpoint hash, and episode metrics.

- [ ] **Step 4: Implement export and browser inference parity**

Export layer names, shapes, row-major float arrays, `tanh` activation, normalization, action order, checkpoint hash, and empty `activityBodyIds`. Add a fixed-input Python/browser logits fixture with tolerance `1e-5`.

- [ ] **Step 5: Run a short deterministic smoke train**

Run: `python -m fly_crossy.train --controller conventional --seed smoke --steps 2048 --envs 4 --output runs/smoke-conventional`  
Expected: checkpoint, metadata, metrics, and exported policy are created with finite values.

- [ ] **Step 6: Run all model and browser tests**

Run: `python -m pytest python/tests -q && npm test && npm run build`  
Expected: all tests and build PASS.

- [ ] **Step 7: Commit conventional training**

```bash
git add python src/game/model.ts tests/game-model.test.mjs tests/fixtures/policy-v1.json
git commit -m "feat: train and export conventional PPO controller"
```

### Task 8: Reduced MaleCNS graph and connectome policy

**Files:**
- Create: `python/fly_crossy/connectome.py`
- Modify: `python/fly_crossy/models.py`
- Modify: `python/fly_crossy/train.py`
- Modify: `python/fly_crossy/export.py`
- Test: `python/tests/test_connectome.py`
- Modify: `python/tests/test_models.py`
- Create: `docs/experiments/reduced-connectome-v1.md`
- Modify: `THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Consumes: a versioned MaleCNS edge table plus atlas-visible body IDs.
- Produces: `ReducedGraphArtifact`, `build_reduced_graph`, `FixedGraphPolicy`, and browser `FixedGraphNetwork` export.

- [ ] **Step 1: Write failing graph integrity tests**

```python
def test_reduced_graph_preserves_declared_edges_and_ids(source_fixture):
    graph = build_reduced_graph(source_fixture, selected_ids=SELECTED_FIXTURE_IDS)
    assert graph.body_ids.tolist() == sorted(SELECTED_FIXTURE_IDS)
    assert graph.edge_index.shape[0] == 2
    assert np.isfinite(graph.edge_weight).all()
    assert graph.source_sha256 == SOURCE_FIXTURE_SHA256
```

- [ ] **Step 2: Confirm tests fail before implementation**

Run: `python -m pytest python/tests/test_connectome.py python/tests/test_models.py -q`  
Expected: FAIL because reduced graph and policy are absent.

- [ ] **Step 3: Implement reproducible graph construction**

Require explicit dataset version, source URL, license, file SHA-256, selection rule, minimum edge threshold, included body IDs, and output artifact SHA-256. Reject unknown atlas IDs, duplicate edges after aggregation, non-finite weights, and empty sensory or readout populations.

- [ ] **Step 4: Implement the fixed-topology recurrent policy**

```python
class FixedGraphPolicy(nn.Module):
    def __init__(self, graph: ReducedGraphArtifact, observation_size: int, actions: int = 5):
        super().__init__()
        self.register_buffer("adjacency", graph.sparse_adjacency())
        self.sensory = nn.Linear(observation_size, graph.node_count, bias=False)
        self.actor = nn.Linear(graph.node_count, actions)
        self.critic = nn.Linear(graph.node_count, 1)

    def forward(self, observation: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        recurrent = torch.sparse.mm(self.adjacency, hidden.T).T
        activity = torch.tanh(self.sensory(observation) + recurrent)
        return self.actor(activity), self.critic(activity).squeeze(-1), activity
```

Store the recurrent adjacency as a fixed sparse buffer. Train sensory projection, declared gains/time constants, actor readout, and critic readout only. Return normalized per-node activity for mapped body IDs.

- [ ] **Step 5: Export fixed graph and validate browser inference**

Extend `parsePolicy` with strict `FixedGraphNetwork` shape, finite-value, ID, and topology checks. Add a deterministic one-step Python/browser fixture with logits and activity tolerance `1e-5`.

- [ ] **Step 6: Document scientific scope and notices**

Record source version/hash, selection rule, retained nodes/edges, model equations, trainable/fixed parameters, normalization, known limitations, and exact attribution in `reduced-connectome-v1.md` and `THIRD_PARTY_NOTICES.md`.

- [ ] **Step 7: Run the complete reduced-controller suite**

Run: `python -m pytest python/tests -q && npm test && npm run check:assets && npm run build`  
Expected: all tests and build PASS.

- [ ] **Step 8: Commit the connectome controller**

```bash
git add python src/game/model.ts tests docs/experiments/reduced-connectome-v1.md THIRD_PARTY_NOTICES.md
git commit -m "feat: add reduced MaleCNS controller"
```

### Task 9: Evaluation controls, final QA, and release evidence

**Files:**
- Create: `python/fly_crossy/evaluate.py`
- Create: `python/tests/test_evaluate.py`
- Create: `configs/eval-v1.json`
- Create: `docs/experiments/evaluation-v1.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: conventional and connectome checkpoints, environment version, held-out seeds, rewiring and silencing configuration.
- Produces: versioned JSON/CSV metrics and a concise Markdown report.

- [ ] **Step 1: Write failing metric and control tests**

```python
def test_summary_uses_distributions_and_terminal_breakdown():
    summary = summarize(episodes)
    assert set(summary["score"]) == {"mean", "median", "max"}
    assert sum(summary["terminalReasons"].values()) == len(episodes)

def test_eval_seeds_do_not_overlap_training_seeds():
    config = load_eval_config(CONFIG)
    assert set(config.training_seeds).isdisjoint(config.evaluation_seeds)
```

- [ ] **Step 2: Run tests and confirm evaluator is absent**

Run: `python -m pytest python/tests/test_evaluate.py -q`  
Expected: FAIL with missing evaluator exports.

- [ ] **Step 3: Implement deterministic evaluation and controls**

Evaluate human-recorded traces when present, conventional policy, reduced-connectome policy, degree-preserving rewired control, selected-population silencing, and untrained readout. Record mean/median/max score, survival steps, terminal reasons, action distribution, wait frequency, model parameters, environment steps, wall-clock inference performance, and hashes.

- [ ] **Step 4: Run bounded training/evaluation suitable for the local machine**

Run conventional and connectome smoke experiments first. Then use the longest training budget that completes within the first-release time box, recording the actual budget rather than implying convergence.

Run: `python -m fly_crossy.evaluate --config configs/eval-v1.json --output runs/eval-v1`  
Expected: JSON, CSV, and Markdown-ready summary with all required metrics and no train/evaluation seed overlap.

- [ ] **Step 5: Perform final verification**

Run: `npm test && npm run check:assets && npm run build && python -m pytest python/tests -q`  
Expected: all commands exit 0.

Open the production preview and verify at 1440×900 and 390×844: game and brain visible, keyboard/touch controls work, all three modes select correctly, manual activity is empty, reset clears activity, same seed repeats, and template attribution is one click away.

- [ ] **Step 6: Document actual results and remaining full-MaleCNS extension**

Populate `evaluation-v1.md` only from emitted metrics, add exact reproduction commands to README, and identify the full MaleCNS live adapter as future work.

- [ ] **Step 7: Commit release evidence**

```bash
git add python configs docs/experiments README.md
git commit -m "test: add controller evaluation and release evidence"
```
