# V7 Privileged-Teacher Visual Connectome Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-first V7 curriculum that trains a privileged ObservationV4 teacher, distills it into an RGB-only measured-connectome student, and promotes 80 -> 1k -> full populations only through held-out closed-loop evidence.

**Architecture:** Add a focused `fly_crossy.v7` package for contracts, immutable datasets, teacher/student models, training, evaluation, safe artifacts, and orchestration. Reuse the canonical environment, camera, planner, reduced graphs, full MaleCNS core, and action order; keep the released controller and V5/V6 experiments unchanged.

**Tech Stack:** Python 3.12, PyTorch 2.12+, NumPy 2+, PyArrow, pytest, existing CPU/CUDA Docker images.

**Spec:** `docs/superpowers/specs/2026-09-27-v7-privileged-teacher-visual-connectome-design.md`

## Global Constraints

- Environment v11, world v6, ObservationV4 (517 float32 values), Reward v5, and canonical `ACTION_ORDER` remain unchanged.
- The teacher receives ObservationV4; every student forward path receives only a 24 x 48 RGB frame and recurrent state.
- Student actions and visualization activity must come from the measured connectome state, never directly from the teacher or planner.
- Measured adjacency/signs remain fixed in V7; core plasticity is outside this plan.
- Profiles and defaults are exactly `smoke` (2/3/3 seeds, 1 DAgger round, 32 steps), `80` (64/21/100, 4, 200), `1k` (128/42/100, 6, 200), and `full` (256/42/100, 8, 200).
- Minimum free disk is 2 GiB for `smoke`/`80`, 5 GiB for `1k`, and 15 GiB for `full`.
- Generated datasets, checkpoints, reports, and logs remain ignored; historical release evidence is immutable.
- `smoke` is contract-only evidence: it exercises teacher and student updates but can never authorize population promotion.
- Do not add coauthor trailers.

## Review Focus

- A dataset with correct shapes but NaN/Inf, invalid action indices, or duplicate seed partitions must fail before an optimizer is created (Task 2 tests).
- A noncanonical or reordered five-action contract must fail at model/checkpoint boundaries instead of silently training wrong actions (Tasks 1, 3, and 7 tests).
- A full/1k run with a missing, stale, or non-promoting predecessor report must stop before loading graph data (Task 8 tests).
- Disk exhaustion or interruption during checkpoint replacement must leave the previous verified `best.pt` loadable (Task 7 tests).
- A time-budget interruption between DAgger rounds must write resumable `latest.pt`, report `continue`, and never run final test on an unselected model (Task 8 tests).

---

### Task 1: V7 profile and seed contracts

**Files:**
- Create: `python/fly_crossy/v7/__init__.py`
- Create: `python/fly_crossy/v7/contracts.py`
- Create: `python/tests/test_v7_contracts.py`

**Interfaces:**
- Consumes: `fly_crossy.schema.ACTION_ORDER`, `OBSERVATION_INPUT_SIZE`, and image constants from `fly_crossy.v4.policy`.
- Produces: `ProfileName`, `ProfileBudget`, `PROFILE_BUDGETS`, `SeedManifest`, `profile_budget(name)`, and `build_seed_manifest(profile, namespace)`.

- [ ] **Step 1: Write failing profile-budget and action-contract tests**

Add tests asserting all four exact budgets from the spec, 2/5/15 GiB disk thresholds, 517 observation values, 24 x 48 x 3 frames, and exactly the canonical five actions. Invalid profile names and any caller-supplied action order differing from `ACTION_ORDER` must raise `ValueError`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `cd python && python -m pytest tests/test_v7_contracts.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: FAIL because `fly_crossy.v7.contracts` does not exist.

- [ ] **Step 3: Implement immutable profile and seed contracts**

Implement:

```python
ProfileName = Literal["smoke", "80", "1k", "full"]

@dataclass(frozen=True, slots=True)
class ProfileBudget:
    name: ProfileName
    training_seeds: int
    validation_seeds: int
    final_test_seeds: int
    dagger_rounds: int
    max_steps: int
    minimum_free_bytes: int

@dataclass(frozen=True, slots=True)
class SeedManifest:
    training: tuple[str, ...]
    validation: tuple[str, ...]
    final_test: tuple[str, ...]
    def validate(self) -> None: ...
    def sha256(self) -> str: ...

def profile_budget(name: str) -> ProfileBudget: ...
def build_seed_manifest(profile: ProfileName, namespace: str = "crossy-v7") -> SeedManifest: ...
def validate_action_order(actions: Sequence[Action]) -> None: ...
```

Seed names must encode namespace, profile, partition, and zero-padded index. `validate()` rejects empty required partitions, duplicates within a partition, and overlap across partitions.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit the contract slice**

```bash
git add python/fly_crossy/v7/__init__.py python/fly_crossy/v7/contracts.py python/tests/test_v7_contracts.py
git commit -m "feat: add V7 profile and seed contracts"
```

### Task 2: Immutable labelled dataset shards

**Files:**
- Create: `python/fly_crossy/v7/dataset.py`
- Create: `python/tests/test_v7_dataset.py`

**Interfaces:**
- Consumes: `SeedManifest`; `create_game`/`step_game`; `flatten_observation`; `render_crossy_neural_frame`; `resize_rgb`; `planner_action_preferences`, `primary_acceptable_mask`, and `pairwise_targets`.
- Produces: `TransitionDataset`, `DatasetShard`, `BehaviorPolicy`, `collect_labeled_episode(...)`, `write_dataset_shard(...)`, and `load_dataset_shard(...)`.

- [ ] **Step 1: Write failing dataset validation and collection tests**

Tests must assert:

- exact dtypes/shapes: frames `uint8[N,24,48,3]`, observations `float32[N,517]`, planner values `float32[N,5,6]`, acceptable/safe `bool[N,5]`, pair signs `int8[N,10]`, pair weights `float32[N,10]`, best/behavior actions `int64[N]`;
- NaN/Inf, out-of-range actions, empty datasets, inconsistent row counts, and noncanonical action metadata are rejected;
- a fixed seed collected twice produces identical arrays;
- the dataset records source, episode seed, and step index without mixing validation/test seeds into a training shard;
- changing one byte changes the content hash; and
- corrupt or hash-mismatched shards fail to load.

- [ ] **Step 2: Run dataset tests and verify RED**

Run: `cd python && python -m pytest tests/test_v7_dataset.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: FAIL because the dataset module is missing.

- [ ] **Step 3: Implement dataset types, deterministic collection, and content-addressed storage**

Implement:

```python
class BehaviorPolicy(Protocol):
    def reset(self) -> None: ...
    def act(self, state: GameState, frame: np.ndarray, observation: np.ndarray) -> int: ...

@dataclass(slots=True)
class TransitionDataset:
    frames: np.ndarray
    observations: np.ndarray
    planner_values: np.ndarray
    acceptable: np.ndarray
    immediate_safe: np.ndarray
    pair_sign: np.ndarray
    pair_weight: np.ndarray
    best_action: np.ndarray
    behavior_action: np.ndarray
    episode_seed: np.ndarray
    step_index: np.ndarray
    source: np.ndarray
    action_order: tuple[str, ...]
    def validate(self) -> None: ...
    def concatenate(self, other: "TransitionDataset") -> "TransitionDataset": ...

@dataclass(frozen=True, slots=True)
class DatasetShard:
    path: Path
    sha256: str
    rows: int
    partition: Literal["training", "validation", "final-test"]

def collect_labeled_episode(
    seed: str,
    *,
    max_steps: int,
    planner_depth: int,
    source: Literal["planner", "teacher", "student"],
    behavior: BehaviorPolicy | None,
) -> TransitionDataset: ...

def write_dataset_shard(dataset: TransitionDataset, root: Path, *, partition: str) -> DatasetShard: ...
def load_dataset_shard(shard: DatasetShard) -> TransitionDataset: ...
```

Use compressed NPZ plus canonical JSON metadata. Write to a temporary sibling, fsync, hash, atomically rename to `<sha256>.npz`, and load-validate before returning.

- [ ] **Step 4: Run dataset tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit immutable dataset support**

```bash
git add python/fly_crossy/v7/dataset.py python/tests/test_v7_dataset.py
git commit -m "feat: add V7 labelled dataset shards"
```

### Task 3: Privileged teacher and corrective updates

**Files:**
- Create: `python/fly_crossy/v7/teacher.py`
- Create: `python/tests/test_v7_teacher.py`

**Interfaces:**
- Consumes: validated `TransitionDataset`, canonical planner preference helpers, and `ACTION_ORDER`.
- Produces: `PrivilegedTeacher`, `teacher_loss(...)`, `fit_teacher_epoch(...)`, `teacher_behavior(...)`, and `teacher_agreement(...)`.

- [ ] **Step 1: Write failing teacher architecture, loss, and update tests**

Tests assert a `517 -> 128 -> 128 -> 5` stateless MLP, canonical output order, finite weighted loss, parameter change after one real optimizer update, deterministic inference, and >=90% agreement calculation based on the planner acceptable mask. Passing RGB frames or reordered actions must raise rather than be ignored.

- [ ] **Step 2: Run teacher tests and verify RED**

Run: `cd python && python -m pytest tests/test_v7_teacher.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: FAIL because the teacher module is missing.

- [ ] **Step 3: Implement the teacher API**

Implement:

```python
class PrivilegedTeacher(nn.Module):
    def __init__(self, observation_size: int = 517, hidden_size: int = 128, actions: int = 5) -> None: ...
    def forward(self, observations: Tensor) -> Tensor: ...

@dataclass(frozen=True, slots=True)
class TeacherLoss:
    total: Tensor
    preference: Tensor
    acceptable: Tensor
    safety: Tensor

def teacher_loss(logits: Tensor, dataset: TransitionDataset, rows: np.ndarray) -> TeacherLoss: ...
def fit_teacher_epoch(model: PrivilegedTeacher, dataset: TransitionDataset, optimizer: Optimizer, *, batch_size: int, seed: int) -> dict[str, float]: ...
def teacher_behavior(model: PrivilegedTeacher, device: torch.device) -> BehaviorPolicy: ...
def teacher_agreement(model: PrivilegedTeacher, dataset: TransitionDataset, device: torch.device) -> float: ...
```

Use tanh hidden activations, existing pairwise preference loss, an acceptable-set negative log-likelihood term, and immediate-fatal-action penalty. Reject all non-finite tensors before backward.

- [ ] **Step 4: Run teacher and existing preference tests**

Run: `cd python && python -m pytest tests/test_v7_teacher.py tests/test_v2_preference_distill.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: PASS.

- [ ] **Step 5: Commit the privileged teacher**

```bash
git add python/fly_crossy/v7/teacher.py python/tests/test_v7_teacher.py
git commit -m "feat: add V7 privileged teacher"
```

### Task 4: Sensory-drive graph API and RGB-only students

**Files:**
- Modify: `python/fly_crossy/models.py`
- Modify: `python/tests/test_nested_gated_connectome.py`
- Create: `python/fly_crossy/v7/student.py`
- Create: `python/tests/test_v7_student.py`

**Interfaces:**
- Consumes: `ReducedGraphArtifact`, `GatedNestedPopulationFixedGraphPolicy`, `FullMaleCNSRGBPolicy`, V6 structured retina slot/kernel helpers, and canonical image/action constants.
- Produces: `NestedPopulationFixedGraphPolicy.forward_sensory_drive(...)`, `StructuredLocalRetina`, `StudentOutput`, `ReducedVisualConnectomeStudent`, `FullVisualConnectomeStudent`, and `build_visual_student(...)`.

- [ ] **Step 1: Write a failing parity test for direct sensory drive**

Add a nested-gated test proving `model(observation, hidden)` exactly matches `model.forward_sensory_drive(model.sensory(observation), hidden)` for both 80 and 1k graphs. Invalid sensory-drive and hidden shapes must raise.

- [ ] **Step 2: Run the nested-gated test and verify RED**

Run: `cd python && python -m pytest tests/test_nested_gated_connectome.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: FAIL because `forward_sensory_drive` is missing.

- [ ] **Step 3: Refactor nested graph forward paths without changing outputs**

Add:

```python
def forward_sensory_drive(
    self, sensory_drive: Tensor, hidden: Tensor
) -> tuple[Tensor, Tensor, Tensor]: ...
```

to nested and gated-nested policies. Existing `forward()` validates observations, computes `self.sensory(observation)`, and delegates. Preserve parameter/state-dict names and old outputs exactly.

- [ ] **Step 4: Run nested model suites and verify GREEN**

Run: `cd python && python -m pytest tests/test_nested_connectome.py tests/test_nested_gated_connectome.py tests/test_nested_feedback_connectome.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: PASS.

- [ ] **Step 5: Write failing RGB student contract tests**

Tests assert:

- deterministic local 3 x 3/eight-channel retina routing with no global matrix;
- reduced `80` and `1k` students accept only `[B,24,48,3]` plus `[B,N]` state;
- output logits are `[B,5]`, activity/next state are `[B,N]`, and all are finite;
- neuron activity changes with the frame;
- adjacency values/indices remain buffers with `requires_grad=False`;
- retina, sensory gain/bias, recurrent scalars, expansion gate, and full motor head are trainable;
- a fake full core produces the same output contract; and
- no student signature or state dict contains ObservationV4/teacher features.

- [ ] **Step 6: Run student tests and verify RED**

Run: `cd python && python -m pytest tests/test_v7_student.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: FAIL because the V7 student module is missing.

- [ ] **Step 7: Implement local retina and reduced/full student adapters**

Implement:

```python
@dataclass(frozen=True, slots=True)
class StudentOutput:
    logits: Tensor
    neuron_activity: Tensor
    next_recurrent_state: Tensor

class StructuredLocalRetina(nn.Module):
    def __init__(self, sensory_count: int, channels: int = 8) -> None: ...
    def forward(self, frames: Tensor) -> Tensor: ...

class ReducedVisualConnectomeStudent(nn.Module):
    def forward(self, frames: Tensor, recurrent_state: Tensor) -> StudentOutput: ...

class FullVisualConnectomeStudent(nn.Module):
    def forward(self, frames: Tensor, recurrent_state: Tensor) -> StudentOutput: ...

def build_visual_student(
    profile: Literal["80", "1k", "full"],
    *,
    device: torch.device,
    flyhard_root: Path | None = None,
) -> nn.Module: ...
```

Reduced students inject retina output directly through `forward_sensory_drive`; the full adapter drives declared `ol_sensory` IDs through the frozen measured core and reads declared `vnc_motor` activity. Use a trainable `readout_count -> 64 -> 5` tanh motor head for every profile. Freeze adjacency/sign tensors; leave artificial recurrent scalars and 1k expansion gate trainable.

- [ ] **Step 8: Run student and nested suites and verify GREEN**

Run: `cd python && python -m pytest tests/test_v7_student.py tests/test_nested_connectome.py tests/test_nested_gated_connectome.py tests/test_nested_feedback_connectome.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: PASS.

- [ ] **Step 9: Commit the student model slice**

```bash
git add python/fly_crossy/models.py python/tests/test_nested_gated_connectome.py python/fly_crossy/v7/student.py python/tests/test_v7_student.py
git commit -m "feat: add RGB visual connectome students"
```

### Task 5: Joint student distillation and DAgger update

**Files:**
- Create: `python/fly_crossy/v7/training.py`
- Create: `python/tests/test_v7_training.py`

**Interfaces:**
- Consumes: `TransitionDataset`, accepted `PrivilegedTeacher`, visual student models, and existing pairwise/progress preference losses.
- Produces: `StudentLoss`, `attach_teacher_logits(...)`, `student_loss(...)`, `fit_student_epoch(...)`, and `student_behavior(...)`.

- [ ] **Step 1: Write failing joint-gradient and fixed-topology tests**

With a real tiny reduced graph batch, assert one update gives finite non-zero gradients and changes retina kernels, sensory calibration, recurrent scalars, and both motor-head layers. Snapshot adjacency indices/values before the update and assert byte equality afterward. Test temperature-2 KL, planner preference, safety, and trust components independently; non-finite teacher logits/loss/gradients must abort before optimizer step.

- [ ] **Step 2: Run training tests and verify RED**

Run: `cd python && python -m pytest tests/test_v7_training.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: FAIL because V7 training functions are missing.

- [ ] **Step 3: Implement teacher-logit attachment and truncated recurrent updates**

Implement:

```python
@dataclass(frozen=True, slots=True)
class StudentLoss:
    total: Tensor
    distillation: Tensor
    preference: Tensor
    safety: Tensor
    trust: Tensor

@dataclass(frozen=True, slots=True)
class StudentBatch:
    frames: Tensor
    acceptable: Tensor
    immediate_safe: Tensor
    pair_sign: Tensor
    pair_weight: Tensor
    teacher_logits: Tensor
    mask: Tensor

def attach_teacher_logits(dataset: TransitionDataset, teacher: PrivilegedTeacher, device: torch.device) -> np.ndarray: ...
def student_loss(output: StudentOutput, batch: StudentBatch, *, trust_penalty: Tensor, temperature: float = 2.0) -> StudentLoss: ...
def fit_student_epoch(model: nn.Module, dataset: TransitionDataset, teacher_logits: np.ndarray, optimizer: Optimizer, *, batch_size: int, window: int, seed: int) -> dict[str, float]: ...
def student_behavior(model: nn.Module, device: torch.device) -> BehaviorPolicy: ...
```

Use weighted sum `1.0*KL + 1.0*preference + 2.0*safety + 0.01*trust`. Carry recurrent state within an episode/window, detach only at window boundaries, and never pass structured observations to the student.

- [ ] **Step 4: Run training, student, and teacher tests and verify GREEN**

Run: `cd python && python -m pytest tests/test_v7_training.py tests/test_v7_student.py tests/test_v7_teacher.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: PASS.

- [ ] **Step 5: Commit joint distillation**

```bash
git add python/fly_crossy/v7/training.py python/tests/test_v7_training.py
git commit -m "feat: jointly distill V7 visual students"
```

### Task 6: Closed-loop metrics and promotion authority

**Files:**
- Create: `python/fly_crossy/v7/evaluation.py`
- Create: `python/tests/test_v7_evaluation.py`

**Interfaces:**
- Consumes: seed manifests, behavior callbacks, planner acceptable masks, and canonical environment transitions.
- Produces: `EpisodeMetrics`, `AggregateMetrics`, `PairedComparison`, `PromotionDecision`, `evaluate_policy(...)`, `robust_key(...)`, and `decide_promotion(...)`.

- [ ] **Step 1: Write failing aggregation and gate tests**

Tests pin the exact robust tuple order: successes, progress-qualified count, mean score, median score, mean survival, negative fatal-action rate. Include cases for one-seed improvement rejected, 59% wins rejected, 60% accepted, mean improvement below 10% rejected, median regression rejected, success-count regression rejected, agreement below 90% teacher/80% student rejected, >90% single-action collapse rejected, and `smoke` never returning `promote`.

- [ ] **Step 2: Run evaluation tests and verify RED**

Run: `cd python && python -m pytest tests/test_v7_evaluation.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: FAIL because the evaluation module is missing.

- [ ] **Step 3: Implement closed-loop evaluation and pure promotion decisions**

Implement:

```python
@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    seed: str
    steps: int
    score: float
    success: bool
    progress_qualified: bool
    terminal_reason: str | None
    fatal_actions: int
    labelled_states: int
    actions: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    episodes: int
    successes: int
    progress_qualified: int
    mean_score: float
    median_score: float
    mean_survival_steps: float
    fatal_action_rate: float
    action_distribution: tuple[float, ...]

@dataclass(frozen=True, slots=True)
class PairedComparison:
    seeds: tuple[str, ...]
    wins: int
    ties: int
    losses: int
    win_fraction: float
@dataclass(frozen=True, slots=True)
class PromotionDecision:
    next_action: Literal["promote", "continue", "reject"]
    predicates: Mapping[str, bool]
    reasons: tuple[str, ...]

def evaluate_policy(policy: BehaviorPolicy, seeds: Sequence[str], *, max_steps: int, planner_depth: int) -> tuple[EpisodeMetrics, ...]: ...
def robust_key(metrics: AggregateMetrics) -> tuple[float, ...]: ...
def decide_promotion(*, profile: ProfileName, predecessor: Sequence[EpisodeMetrics], candidate: Sequence[EpisodeMetrics], planner_agreement: float, agreement_threshold: float, final_test_passed: bool) -> PromotionDecision: ...
```

`decide_promotion` must be pure, serialize every predicate, compare paired seeds by seed ID, and fail on missing/duplicate/mismatched seeds.

- [ ] **Step 4: Run evaluation and legacy summary tests and verify GREEN**

Run: `cd python && python -m pytest tests/test_v7_evaluation.py tests/test_evaluate.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: PASS.

- [ ] **Step 5: Commit promotion authority**

```bash
git add python/fly_crossy/v7/evaluation.py python/tests/test_v7_evaluation.py
git commit -m "feat: gate V7 promotion on robust evaluation"
```

### Task 7: Resource preflight and durable resumable artifacts

**Files:**
- Create: `python/fly_crossy/v7/artifacts.py`
- Create: `python/tests/test_v7_artifacts.py`

**Interfaces:**
- Consumes: profile contracts, model/optimizer states, seed and dataset hashes.
- Produces: `ResourceSnapshot`, `CheckpointProvenance`, `preflight_resources(...)`, `save_verified_checkpoint(...)`, `load_verified_checkpoint(...)`, `sha256_file(...)`, and `retain_run_artifacts(...)`.

- [ ] **Step 1: Write failing preflight, atomicity, and resume tests**

Tests must cover exact disk thresholds, resource snapshot fields, safe tensor-only loading, canonical action-order validation, graph/teacher/dataset/seed hash mismatch, RNG round-trip reproducing the next optimizer update, corrupt temporary writes preserving old `best.pt`, verified replacement ordering, and retention limited to `best.pt`, `latest.pt`, `report.json`, one log, and a dataset manifest.

- [ ] **Step 2: Run artifact tests and verify RED**

Run: `cd python && python -m pytest tests/test_v7_artifacts.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: FAIL because the artifacts module is missing.

- [ ] **Step 3: Implement resource and artifact contracts**

Implement:

```python
@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    free_disk_bytes: int
    available_ram_bytes: int
    device: str
    torch_version: str
    thread_count: int

@dataclass(frozen=True, slots=True)
class CheckpointProvenance:
    version: str
    profile: ProfileName
    stage: str
    action_order: tuple[str, ...]
    graph_sha256: str | None
    teacher_sha256: str | None
    dataset_manifest_sha256: str
    seed_manifest_sha256: str
    completed_round: int
    completed_epoch: int

def preflight_resources(profile: ProfileName, output: Path, device: torch.device) -> ResourceSnapshot: ...
def save_verified_checkpoint(path: Path, *, payload: Mapping[str, Any], verify: Callable[[Mapping[str, Any]], None]) -> str: ...
def load_verified_checkpoint(path: Path, *, expected: CheckpointProvenance) -> dict[str, Any]: ...
def sha256_file(path: Path) -> str: ...
def retain_run_artifacts(run_dir: Path) -> tuple[Path, ...]: ...
```

Checkpoint payloads contain only tensor-safe primitives accepted by `torch.load(..., weights_only=True)`. Save to a sibling temporary file, fsync, safe-load, validate, hash, then `os.replace`; never unlink the old target first.

- [ ] **Step 4: Run artifact and existing checkpoint tests and verify GREEN**

Run: `cd python && python -m pytest tests/test_v7_artifacts.py tests/test_checkpoint.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: PASS.

- [ ] **Step 5: Commit safe artifacts**

```bash
git add python/fly_crossy/v7/artifacts.py python/tests/test_v7_artifacts.py
git commit -m "feat: add durable V7 training artifacts"
```

### Task 8: Curriculum orchestration and CPU smoke path

**Files:**
- Create: `python/fly_crossy/v7/curriculum.py`
- Create: `python/tests/test_v7_curriculum.py`
- Create: `scripts/v7-training-smoke.sh`

**Interfaces:**
- Consumes: all Tasks 1-7 interfaces; current released 80-neuron controller adapter for v11 baseline; reduced/full graph loaders.
- Produces: `CurriculumConfig`, `run_curriculum(...)`, module CLI, schema-versioned `report.json`, and Docker smoke script.

- [ ] **Step 1: Write failing orchestration state-machine tests**

Using tiny real models and controlled policy callbacks, test:

- order `preflight -> baseline -> dataset -> teacher -> student -> validation -> selected final test -> report`;
- validation selects `best.pt`, rejected rounds restore best, and final test runs once only after selection;
- `smoke` executes one teacher and one student update but cannot promote;
- `1k` requires a hash-valid promoting 80 report before graph loading;
- `full` requires a hash-valid promoting 1k report before graph loading;
- a non-promoting predecessor, expired time budget, interrupt, OOM, or non-finite update yields durable `latest.pt` plus `continue`/`reject` with reason;
- resume begins at the next incomplete round with identical seed manifests; and
- report includes provenance, resources, trainable names, per-round losses, per-seed metrics, paired predicates, elapsed time, checkpoint hash, decision, and explicit simulated/teacher/planner labels.

- [ ] **Step 2: Run curriculum tests and verify RED**

Run: `cd python && python -m pytest tests/test_v7_curriculum.py -q --maxfail=1 --basetemp=.pytest-tmp`

Expected: FAIL because the curriculum module is missing.

- [ ] **Step 3: Implement the curriculum state machine and CLI**

Implement:

```python
@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    profile: ProfileName
    output: Path
    device: str = "cpu"
    resume: bool = False
    namespace: str = "crossy-v7"
    time_budget_seconds: float | None = None
    flyhard_root: Path | None = None
    predecessor_report: Path | None = None

def run_curriculum(config: CurriculumConfig) -> dict[str, Any]: ...
def main() -> None: ...
```

The CLI is `python -m fly_crossy.v7.curriculum --profile smoke|80|1k|full --out <run-dir>`. Default the 80 profile time budget to 3,600 seconds on CPU. Baseline evaluation adapts the immutable released controller to v11 exactly as the current runtime does. `smoke` labels `contractOnly=true` and never changes release artifacts.

- [ ] **Step 4: Run curriculum tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Add and run the Docker CPU smoke script**

The script bind-mounts the repository read-only except for a temporary run directory, invokes the locked CPU simulation image with `--profile smoke`, validates `report.json`, reloads both checkpoints, and asserts the report is contract-only and contains finite teacher/student evidence.

Run: `bash scripts/v7-training-smoke.sh`

Expected: PASS and print the temporary report path plus elapsed time; it must not modify `release/`.

- [ ] **Step 6: Commit the orchestration slice**

```bash
git add python/fly_crossy/v7/curriculum.py python/tests/test_v7_curriculum.py scripts/v7-training-smoke.sh
git commit -m "feat: orchestrate V7 CPU training curriculum"
```

### Task 9: Documentation, full verification, and first local evidence

**Files:**
- Modify: `README.md`
- Modify: `docs/ACTIVE_CONTEXT.md`
- Modify: `.gitignore`
- Modify: `tests/repository-hygiene.test.mjs`
- Modify: `tests/release-evidence.test.mjs`
- Create (generated, ignored): `runs/crossy-v7-80-local/`

**Interfaces:**
- Consumes: V7 CLI/report schema and repository validation commands.
- Produces: operator instructions and the first local 80-neuron evidence report; no runtime controller integration.

- [ ] **Step 1: Write failing documentation/repository-hygiene assertions**

Extend `tests/repository-hygiene.test.mjs` and `tests/release-evidence.test.mjs` to assert V7 generated paths remain ignored, release artifacts remain unchanged, and documented commands reference the real module/profile names.

- [ ] **Step 2: Run documentation-focused tests and verify RED**

Run: `node --test tests/repository-hygiene.test.mjs tests/release-evidence.test.mjs`

Expected: FAIL until the V7 ignore/documentation contract is added.

- [ ] **Step 3: Document V7 as experimental and non-authoritative**

Add concise commands for smoke, 80, resume, 1k with predecessor report, full with predecessor report, CPU fallback, CUDA selection, report interpretation, and disk thresholds. State that the released 80-neuron compatibility controller remains authoritative until separate integration approval.

- [ ] **Step 4: Run focused documentation tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Run the complete required verification**

Run:

```bash
npm test
npm run check:assets
npm run build
docker run --rm -v "$PWD/python:/workspace" -w /workspace stabilize-origin-main-simulation python -m pytest tests -q --maxfail=1 --basetemp=.pytest-tmp
docker compose build simulation web
docker compose up --wait
bash scripts/docker-smoke.sh
bash scripts/v7-training-smoke.sh
docker compose down
```

Expected: every command exits 0. Record any unrelated pre-existing failure by exact test/command; do not claim completion while required verification is red.

- [ ] **Step 6: Commit documentation and hygiene updates**

```bash
git add README.md docs/ACTIVE_CONTEXT.md .gitignore tests/repository-hygiene.test.mjs tests/release-evidence.test.mjs
git commit -m "docs: document V7 training workflow"
```

- [ ] **Step 7: Run the first local 80-neuron curriculum without integrating it**

Run:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace/python \
  stabilize-origin-main-simulation \
  python -m fly_crossy.v7.curriculum \
    --profile 80 \
    --out /workspace/runs/crossy-v7-80-local \
    --device cpu \
    --time-budget-seconds 3600
```

Expected: a load-verified `best.pt`, resumable `latest.pt`, and `report.json` with `promote`, `continue`, or `reject`. Report the measured outcome; do not change browser/backend controller selection in this plan.
