# Multiscale MaleCNS Controllers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate five nested MaleCNS-derived sparse graphs and train, evaluate, and stream a distinct autonomous controller for 80, 1,000, 5,000, 20,000, and 124,289 neurons.

**Architecture:** Immutable source manifests feed a deterministic graph builder that preserves atlas somata and only sourced edges. A sparse recurrent-rate policy injects observations into declared sensory nodes and reads actions from declared descending nodes; behavior cloning establishes useful play before bounded independent PPO fine-tuning and common held-out evaluation.

**Tech Stack:** Python 3.12, PyTorch sparse/indexed operations, NumPy, MaleCNS artifacts, pytest, JSON/NPZ/Safetensors-style immutable artifacts.

**Spec:** `docs/superpowers/specs/2026-09-14-multiscale-biomechanical-keyboard-design.md`

## Global Constraints

- Population sizes are exactly 80, 1,000, 5,000, 20,000, and 124,289 and strictly nested.
- The full mode includes all visible atlas somata; source-missing edges produce explicitly counted isolated nodes.
- No synapse, identifier, anatomy, activity, result, or performance number may be invented.
- A dense 124,289 by 124,289 matrix is forbidden.
- Each size has its own graph, configuration, checkpoint, hashes, training record, and evaluation report.
- All modes share observation/action schemas, held-out seeds, and the biomechanical motor layer.
- CPU is exact within declared tolerances but may be slower than real time for 124,289 neurons.
- Activity is labeled simulated and keyed by validated MaleCNS body ID.
- Do not add automated-agent coauthor trailers to commits.

---

## File map

- `data/manifests/malecns-v1.json` — verified source URLs, licenses, sizes, and hashes.
- `configs/populations-v1.json` — selection roots, groups, expansion priorities, and exact targets.
- `configs/controllers/{80,1000,5000,20000,124289}.json` — independent model/training configs.
- `python/fly_crossy/data.py` — verified downloader and manifest resolver.
- `python/fly_crossy/populations.py` — nested soma selection and graph artifact builder.
- `python/fly_crossy/sparse_policy.py` — sparse recurrent controller and activity extraction.
- `python/fly_crossy/imitation.py` — deterministic planner dataset and behavior cloning.
- `python/fly_crossy/ppo.py` — bounded independent PPO fine-tuning.
- `python/fly_crossy/activity.py` — keyframe/delta encoding.
- `python/fly_crossy/evaluate_multiscale.py` — shared held-out evaluation and reports.
- `python/fly_crossy/artifacts.py` — immutable artifact manifests and hash checks.
- `python/tests/*` — data, graph, sparse-allocation, training, activity, and evaluation tests.
- `tests/fixtures/malecns-mini/*` — small sourced-shape fixtures for deterministic tests.

### Task 1: Verified MaleCNS data manifest and downloader

**Files:**
- Create: `data/manifests/malecns-v1.json`
- Create: `python/fly_crossy/data.py`
- Create: `python/tests/test_data.py`
- Create: `tests/fixtures/malecns-mini/atlas.csv`
- Create: `tests/fixtures/malecns-mini/edges.csv`

**Interfaces:**
- Produces: `DataManifest.load(path)`, `VerifiedDownloader.fetch(artifact_id) -> Path`, and `sha256_file(path) -> str`.
- Enforces: HTTPS source, 64-hex checksum, expected byte size, atomic `.partial` download, and manifest-rooted artifact IDs.

- [ ] **Step 1: Add minimal graph fixtures and failing hash tests**

```csv
body_id,x,y,z,group
10,0,0,0,sensory
20,1,0,0,interneuron
30,2,0,0,descending
40,3,0,0,atlas_only
```

```csv
source,target,weight
10,20,4
20,30,2
```

```python
def test_checksum_mismatch_never_replaces_cached_artifact(tmp_path, manifest):
    target = tmp_path / "atlas.csv"
    target.write_text("known-good")
    with pytest.raises(ChecksumMismatch):
        VerifiedDownloader(manifest, tmp_path).install_bytes("atlas", b"wrong")
    assert target.read_text() == "known-good"
```

- [ ] **Step 2: Run and verify missing downloader failure**

Run: `cd python && pytest tests/test_data.py -q`  
Expected: FAIL because `fly_crossy.data` is absent.

- [ ] **Step 3: Implement verified, atomic, manifest-rooted data access**

```python
@dataclass(frozen=True, slots=True)
class DataArtifact:
    id: str
    url: str
    sha256: str
    bytes: int
    license: str

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

Download only enumerated IDs to a sibling `.partial`, check byte size and checksum, then atomically replace. Never accept a client path or follow a manifest filename outside the configured data root.

- [ ] **Step 4: Resolve and record real source metadata**

Use the MaleCNS download page named in the spec. Record exact retrieval date, release label, URLs, licenses/notices, byte sizes, and checksums in `malecns-v1.json`; run the downloader in `--verify-only` mode against the Docker volume.

Run: `cd python && python -m fly_crossy.data --manifest ../data/manifests/malecns-v1.json --root ../.local/fly-data --verify-only`  
Expected: every declared artifact reports `verified`; a partial or mismatch exits nonzero.

- [ ] **Step 5: Run tests and commit manifest tooling**

Run: `cd python && pytest tests/test_data.py -q`  
Expected: PASS.

```bash
git add data/manifests/malecns-v1.json tests/fixtures/malecns-mini python/fly_crossy/data.py python/tests/test_data.py
git commit -m "feat: verify MaleCNS source artifacts"
```

### Task 2: Deterministic nested graph builder

**Files:**
- Create: `configs/populations-v1.json`
- Create: `python/fly_crossy/populations.py`
- Create: `python/tests/test_populations.py`
- Create: `python/fly_crossy/artifacts.py`
- Create: `python/tests/test_artifacts.py`

**Interfaces:**
- Consumes: verified atlas/edge tables and current 80-node graph.
- Produces: `build_populations(inputs, config) -> dict[int, PopulationGraph]`.
- Produces: `PopulationGraph(node_ids, edge_index, edge_weight, sensory_indices, descending_indices, isolated_indices)` and immutable `graph-manifest.json`.

- [ ] **Step 1: Write failing exact-size, nesting, and no-invention tests**

```python
def test_populations_are_exact_and_nested(real_or_synthetic_inputs, config):
    graphs = build_populations(real_or_synthetic_inputs, config)
    expected = (80, 1000, 5000, 20000, 124289)
    assert tuple(graphs) == expected
    for smaller, larger in itertools.pairwise(expected):
        assert len(graphs[smaller].node_ids) == smaller
        assert set(graphs[smaller].node_ids) < set(graphs[larger].node_ids)

def test_every_output_edge_exists_in_source(mini_inputs, mini_config):
    graph = build_populations(mini_inputs, mini_config)[4]
    assert set(graph.edges) <= set(mini_inputs.edges)
    assert graph.isolated_ids == (40,)
```

- [ ] **Step 2: Run graph tests and verify failure**

Run: `cd python && pytest tests/test_populations.py tests/test_artifacts.py -q`  
Expected: FAIL because graph/artifact modules are absent.

- [ ] **Step 3: Implement stable expansion and tie-breaking**

```python
TARGET_SIZES = (80, 1_000, 5_000, 20_000, 124_289)

def expansion_key(candidate: Candidate) -> tuple[int, int, int]:
    return (-candidate.priority, -candidate.source_weight, candidate.body_id)
```

Start with the committed 80-node ordered core. Expand through configured functional groups and synaptic neighbors; stable-sort ties by source body ID. For the last population append every remaining atlas soma in body-ID order. Filter edges only by endpoint membership and configured source rules. Fail on duplicate IDs, unknown edge endpoints, insufficient somata, or non-exact target size.

- [ ] **Step 4: Write compressed immutable graph artifacts**

```python
manifest = {
    "schemaVersion": 1,
    "requestedSize": len(graph.node_ids),
    "connectedCount": len(graph.node_ids) - len(graph.isolated_indices),
    "isolatedCount": len(graph.isolated_indices),
    "edgeCount": graph.edge_index.shape[1],
    "nodeOrderSha256": sha256_array(graph.node_ids),
    "edgeSha256": sha256_arrays(graph.edge_index, graph.edge_weight),
}
```

Store integer arrays and weights without JSON expansion. Loaders verify shape, sorted order, range, manifest hashes, and no dense adjacency member.

- [ ] **Step 5: Generate twice and compare hashes**

Run: `cd python && python -m fly_crossy.populations --manifest ../data/manifests/malecns-v1.json --config ../configs/populations-v1.json --output ../.local/graphs-a && python -m fly_crossy.populations --manifest ../data/manifests/malecns-v1.json --config ../configs/populations-v1.json --output ../.local/graphs-b && diff -rq ../.local/graphs-a ../.local/graphs-b`  
Expected: no differences.

- [ ] **Step 6: Run tests and commit graph pipeline**

Run: `cd python && pytest tests/test_populations.py tests/test_artifacts.py -q`  
Expected: PASS.

```bash
git add configs/populations-v1.json python/fly_crossy/populations.py python/fly_crossy/artifacts.py python/tests/test_populations.py python/tests/test_artifacts.py
git commit -m "feat: build five nested MaleCNS graphs"
```

### Task 3: Sparse recurrent policy and independent configs

**Files:**
- Create: `configs/controllers/80.json`
- Create: `configs/controllers/1000.json`
- Create: `configs/controllers/5000.json`
- Create: `configs/controllers/20000.json`
- Create: `configs/controllers/124289.json`
- Create: `python/fly_crossy/sparse_policy.py`
- Create: `python/tests/test_sparse_policy.py`

**Interfaces:**
- Consumes: `PopulationGraph`, `ObservationV1` tensor, previous recurrent state.
- Produces: `SparseRatePolicy.forward(observation, state) -> PolicyOutput(logits, value, state, activity)`.
- Enforces: sensory injection indices and descending readout indices are subsets of the graph manifest.

- [ ] **Step 1: Add explicit independent controller configs**

Each file names its exact population size and distinct output directory while sharing schema version, five actions, observation version, recurrent update family, and seed-suite reference:

```json
{
  "schemaVersion": 1,
  "population": 1000,
  "graphManifest": "graphs/1000/graph-manifest.json",
  "observationVersion": 1,
  "actions": ["forward","backward","left","right","wait"],
  "dynamics": {"kind":"sparse-rate-v1","stepsPerDecision":4,"activation":"tanh"},
  "training": {"seed":1000,"behaviorCloneEpochs":20,"ppoEnvironmentSteps":100000}
}
```

- [ ] **Step 2: Write failing sparse-shape and allocation tests**

```python
@pytest.mark.parametrize("size", [80, 1000, 5000, 20000, 124289])
def test_policy_output_shapes(size, graph_factory):
    graph = graph_factory(size)
    policy = SparseRatePolicy(graph, observation_size=370, action_size=5)
    output = policy(torch.zeros(2, 370), policy.initial_state(2))
    assert output.logits.shape == (2, 5)
    assert output.state.shape == (2, size)

def test_policy_has_no_square_population_parameter(graph_1000):
    policy = SparseRatePolicy(graph_1000, 370, 5)
    assert all(parameter.numel() < 1_000 * 1_000 for parameter in policy.parameters())
```

- [ ] **Step 3: Run and verify failure**

Run: `cd python && pytest tests/test_sparse_policy.py -q`  
Expected: FAIL because `SparseRatePolicy` is absent.

- [ ] **Step 4: Implement indexed recurrent propagation**

```python
messages = state[:, self.edge_source] * self.edge_scale
recurrent = torch.zeros_like(state).scatter_add_(1, self.edge_target.expand(batch, -1), messages)
drive = torch.zeros_like(state).index_add_(1, self.sensory_index, self.sensory_projection(observation))
next_state = torch.tanh(self.leak * state + recurrent + drive + self.bias)
logits = self.readout(next_state[:, self.descending_index])
```

Represent isolated nodes explicitly with leak, bias, and permitted sensory drive but no recurrent messages. Validate all indices and finite values at load. Chunk edge propagation when memory benchmarks require it without changing numeric semantics beyond tolerance.

- [ ] **Step 5: Compare CPU determinism and optional GPU tolerance**

Run: `cd python && pytest tests/test_sparse_policy.py -q`  
Expected: CPU repeated inference is exact for fixed inputs; CUDA comparison is skipped when absent and passes configured `rtol`/`atol` when present.

- [ ] **Step 6: Commit sparse controllers**

```bash
git add configs/controllers python/fly_crossy/sparse_policy.py python/tests/test_sparse_policy.py
git commit -m "feat: add sparse multiscale neural policy"
```

### Task 4: Behavior cloning from the deterministic planner

**Files:**
- Create: `python/fly_crossy/planner.py`
- Create: `python/fly_crossy/imitation.py`
- Create: `python/tests/test_planner.py`
- Create: `python/tests/test_imitation.py`
- Create: `configs/seed-suites-v1.json`

**Interfaces:**
- Produces: `Planner.act(state) -> Action`, `generate_demonstrations(seeds)`, and `train_behavior_clone(policy, dataset, config)`.
- Produces: disjoint `train`, `validation`, and `heldOut` deterministic seed lists.

- [ ] **Step 1: Write failing planner legality and split tests**

```python
def test_planner_uses_only_legal_actions():
    actions = [Planner().act(state) for state in seeded_states("planner-test", 200)]
    assert set(actions) <= {"forward", "backward", "left", "right", "wait"}

def test_seed_suites_are_disjoint(seed_suites):
    train, validation, held_out = map(set, (seed_suites.train, seed_suites.validation, seed_suites.held_out))
    assert not (train & validation or train & held_out or validation & held_out)
```

- [ ] **Step 2: Run and verify failure**

Run: `cd python && pytest tests/test_planner.py tests/test_imitation.py -q`  
Expected: FAIL because planner/imitation modules are absent.

- [ ] **Step 3: Implement deterministic finite-horizon planning**

Score legal actions over the existing bounded observation using terminal avoidance, forward progress, safe landing, and wait cost already defined by the game reward. Break equal scores by the fixed order `forward,left,right,wait,backward`; record planner version with demonstrations.

- [ ] **Step 4: Implement independently seeded behavior-cloning runs**

```python
def clone_loss(output: PolicyOutput, target_action: Tensor) -> Tensor:
    return torch.nn.functional.cross_entropy(output.logits, target_action)
```

Train every size from its own initialization and configuration. Save best validation checkpoint atomically with graph/config/seed hashes and never overwrite an existing released run directory.

- [ ] **Step 5: Run a five-size smoke clone**

Run: `cd python && python -m fly_crossy.imitation --configs ../configs/controllers --seed-suites ../configs/seed-suites-v1.json --smoke --output ../.local/runs/bc-smoke && pytest tests/test_planner.py tests/test_imitation.py -q`  
Expected: five distinct run manifests/checkpoints are created and tests PASS.

- [ ] **Step 6: Commit imitation pipeline**

```bash
git add configs/seed-suites-v1.json python/fly_crossy/planner.py python/fly_crossy/imitation.py python/tests/test_planner.py python/tests/test_imitation.py
git commit -m "feat: bootstrap controllers from planner behavior"
```

### Task 5: Independent bounded PPO fine-tuning and checkpoints

**Files:**
- Create: `python/fly_crossy/ppo.py`
- Create: `python/fly_crossy/checkpoints_v2.py`
- Create: `python/tests/test_ppo.py`
- Create: `python/tests/test_checkpoints_v2.py`

**Interfaces:**
- Consumes: behavior-cloned policy, deterministic Python environment, controller config.
- Produces: `fine_tune_ppo(run, config) -> TrainingRecord` and immutable v2 checkpoint bundle.

- [ ] **Step 1: Write failing rollout and compatibility tests**

```python
def test_ppo_rollout_contains_required_fields(smoke_trainer):
    batch = smoke_trainer.collect(steps=8)
    assert batch.observations.shape[0] == 8
    assert batch.actions.shape == batch.rewards.shape == batch.dones.shape

def test_checkpoint_rejects_wrong_graph_hash(checkpoint, graph):
    with pytest.raises(ArtifactMismatch, match="graph"):
        load_checkpoint(checkpoint, dataclasses.replace(graph, sha256="0" * 64))
```

- [ ] **Step 2: Run and verify failure**

Run: `cd python && pytest tests/test_ppo.py tests/test_checkpoints_v2.py -q`  
Expected: FAIL because PPO/v2 checkpoint modules are absent.

- [ ] **Step 3: Implement clipped PPO with explicit budgets**

```python
ratio = (new_log_prob - old_log_prob).exp()
policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
value_loss = 0.5 * (value - returns).square().mean()
loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy.mean()
```

Keep environment steps, optimizer updates, seeds, device, wall time, losses, and checkpoint hashes in `training-record.json`. Stop exactly at configured environment steps; smoke mode uses 64 steps and is labeled `smoke`, never `release`.

- [ ] **Step 4: Run smoke PPO for all sizes**

Run: `cd python && python -m fly_crossy.ppo --input ../.local/runs/bc-smoke --configs ../configs/controllers --smoke --output ../.local/runs/ppo-smoke && pytest tests/test_ppo.py tests/test_checkpoints_v2.py -q`  
Expected: five compatible distinct bundles and passing tests.

- [ ] **Step 5: Commit PPO and checkpoint code**

```bash
git add python/fly_crossy/ppo.py python/fly_crossy/checkpoints_v2.py python/tests/test_ppo.py python/tests/test_checkpoints_v2.py
git commit -m "feat: fine-tune five independent sparse controllers"
```

### Task 6: Sparse activity keyframes and deltas

**Files:**
- Create: `python/fly_crossy/activity.py`
- Create: `python/tests/test_activity.py`
- Modify: `python/fly_crossy/server.py`
- Modify: `python/tests/test_server.py`

**Interfaces:**
- Produces: `ActivityEncoder.keyframe(ids, values)` and `delta(previous, current, epsilon)` compatible with protocol v2.

- [ ] **Step 1: Write failing reconstruction and bound tests**

```python
def test_delta_reconstructs_within_epsilon():
    encoder = ActivityEncoder(max_delta=20_000, epsilon=1e-4)
    keyframe = encoder.keyframe(IDS, BEFORE)
    delta = encoder.delta(BEFORE, AFTER)
    restored = apply_delta(keyframe, delta)
    np.testing.assert_allclose(restored, AFTER, atol=1e-4)

def test_oversize_change_uses_chunked_keyframe():
    frames = ActivityEncoder(max_delta=20_000).encode(np.zeros(124_289), np.ones(124_289))
    assert all(len(frame.entries) <= 20_000 for frame in frames)
```

- [ ] **Step 2: Run and verify failure**

Run: `cd python && pytest tests/test_activity.py -q`  
Expected: FAIL because the activity encoder is absent.

- [ ] **Step 3: Implement changed-value deltas and periodic keyframes**

Send body IDs, not array positions, and include base/keyframe sequence. Start/reconnect/population changes require a complete chunked keyframe. Emit a periodic keyframe at a configured interval and on browser request.

- [ ] **Step 4: Run activity/server tests and commit**

Run: `cd python && pytest tests/test_activity.py tests/test_server.py -q`  
Expected: PASS.

```bash
git add python/fly_crossy/activity.py python/tests/test_activity.py python/fly_crossy/server.py python/tests/test_server.py
git commit -m "feat: stream sparse simulated neural activity"
```

### Task 7: Common held-out evaluation and release artifacts

**Files:**
- Create: `python/fly_crossy/evaluate_multiscale.py`
- Create: `python/tests/test_evaluate_multiscale.py`
- Create: `docs/experiments/multiscale-method.md`
- Create: `release/multiscale-v1/README.md`

**Interfaces:**
- Consumes: five release checkpoint bundles, held-out seeds, shared motor metrics.
- Produces: per-episode JSONL, aggregate JSON/CSV/Markdown, artifact manifest, and model cards.

- [ ] **Step 1: Write failing metric-separation tests**

```python
def test_summary_separates_decision_and_motor_failures(records):
    summary = summarize(records)
    assert summary[80]["neuralActionCount"] == 3
    assert summary[80]["confirmedPressCount"] == 2
    assert summary[80]["motorFailureCount"] == 1
    assert "meanScore" in summary[80]
```

- [ ] **Step 2: Run and verify failure**

Run: `cd python && pytest tests/test_evaluate_multiscale.py -q`  
Expected: FAIL because the evaluator is absent.

- [ ] **Step 3: Implement reproducible evaluation outputs**

Report mean/median/max score, survival, terminal causes, action distribution, wait rate, press success, motor failure, action latency, environment steps, parameters, wall time, hardware, and effective simulation speed. Never substitute smoke results for release results.

- [ ] **Step 4: Run the deterministic smoke evaluation**

Run: `cd python && python -m fly_crossy.evaluate_multiscale --runs ../.local/runs/ppo-smoke --seed-suites ../configs/seed-suites-v1.json --smoke --output ../.local/eval-smoke && pytest tests/test_evaluate_multiscale.py -q`  
Expected: five rows, hashes/provenance present, and tests PASS.

- [ ] **Step 5: Run release training/evaluation on available hardware**

Run: `cd python && python -m fly_crossy.imitation --configs ../configs/controllers --seed-suites ../configs/seed-suites-v1.json --output ../.local/runs/bc-release && python -m fly_crossy.ppo --input ../.local/runs/bc-release --configs ../configs/controllers --output ../.local/runs/ppo-release && python -m fly_crossy.evaluate_multiscale --runs ../.local/runs/ppo-release --seed-suites ../configs/seed-suites-v1.json --output ../release/multiscale-v1`  
Expected: commands complete, each size has a distinct checkpoint/report, and the manifest verifies every committed release file. If available hardware cannot finish an approved release budget, publish no fabricated checkpoint; keep the mode marked unreleased until a verified run exists.

- [ ] **Step 6: Run complete Python tests and commit generated evidence**

Run: `cd python && pytest -q && python -m fly_crossy.artifacts verify ../release/multiscale-v1/manifest.json`  
Expected: PASS.

```bash
git add python/fly_crossy/evaluate_multiscale.py python/tests/test_evaluate_multiscale.py docs/experiments/multiscale-method.md release/multiscale-v1
git commit -m "feat: release multiscale controller evidence"
```

## Plan completion checkpoint

Run: `cd python && pytest -q && python -m fly_crossy.artifacts verify ../release/multiscale-v1/manifest.json`  
Expected: exact nested graphs, distinct compatible checkpoints, reconstructed activity, and common held-out reports all verify without dense graph allocation.
