# V7 Privileged-Teacher Visual Connectome Design

**Date:** 2026-09-27

**Status:** Proposed for implementation

## Purpose

Build a CPU-first training curriculum that can prove a visual connectome
controller improves before the project spends time or GPU capacity on the full
165,122-neuron MaleCNS graph.

The curriculum uses a small privileged teacher to turn planner knowledge into a
dense learning signal. The teacher receives the canonical structured game
observation. The connectome student receives only the 24 x 48 RGB neural
camera frame. At runtime, student actions must be produced from connectome
activity; the teacher is a training-only component.

## Context and Problem

V5 retained the random/global RGB-to-sensory interface and plateaued at an
exact score of 41 with zero robust successes. V6 replaced that interface with a
deterministic local retina but started the new representation against a frozen
MLP64 motor decoder. Its full run selected a score of 7, a robust mean of 3.67,
and zero successes. Training loss declined while closed-loop gameplay did not.

Two separate issues must be corrected:

1. A newly initialized visual representation needs a stable, informative target
   before it is asked to drive a previously trained motor decoder.
2. Model selection must use held-out closed-loop episodes rather than one exact
   seed or training loss.

The released 80-neuron controller is an environment-v6 compatibility controller.
V7 targets the current environment v11 and establishes a new v11 baseline before
comparing any candidate.

## Goals

- Train and evaluate the teacher and an 80-neuron visual student on this laptop
  using the CPU Docker environment.
- Keep privileged state out of the student model and its checkpoints.
- Jointly train the artificial visual and motor interfaces instead of freezing
  the decoder against a new retina.
- Preserve a per-decision activity vector for every student neuron so accepted
  checkpoints remain compatible with real-time brain visualization.
- Preserve measured connectome topology during the first student curriculum.
- Promote from 80 neurons to 1,000 neurons, then to the full MaleCNS, only after
  deterministic held-out gates pass.
- Produce resumable checkpoints and machine-readable reports that explain why a
  run was promoted, continued, or rejected.
- Reuse datasets and retain only bounded training artifacts so local runs do not
  fill the disk.

## Non-goals

- V7 does not replace the released runtime controller until a candidate passes
  final held-out evaluation and receives a separate integration decision.
- V7 does not change gameplay, world generation, collision timing, observation,
  rewards, or the FlyBody physical keyboard authority chain.
- V7 does not describe simulated activity as biological recording data.
- The first implementation does not train measured graph edges. Limited core
  plasticity is a later, explicitly gated experiment.
- The first implementation does not promise that a larger population will beat
  a smaller one; promotion is evidence-based.

## Architecture

### Privileged teacher

The teacher is a compact two-hidden-layer MLP operating on the canonical
517-value ObservationV4 vector. It produces five action logits in canonical
`ACTION_ORDER` order. The teacher is trained from the existing depth-4 planner's
full action preferences, not only its argmax action.

The teacher is deliberately small and stateless. ObservationV4 already includes
the position, local hazards, traffic, support, and motion data needed for a
Markov decision. If held-out evaluation disproves that assumption, the run is
rejected rather than silently adding recurrence.

Teacher training has two phases:

1. Offline preference distillation on planner trajectories and perturbed starts.
2. DAgger-style collection on states visited by the current teacher, labelled by
   the planner, followed by short corrective updates.

The teacher checkpoint is never accepted on classification loss alone. It must
pass the closed-loop teacher gate described below.

### Visual connectome student

The student receives only the RGB neural camera frame used by V5/V6. Its data
path is:

`RGB frame -> shared local retina -> declared sensory population -> fixed measured graph -> declared readout population -> trainable motor head -> action logits`

The retina uses a shared local 3 x 3 kernel bank with eight channels and
deterministic position-major routing. It must not use random or global pixel
mixing. The graph adjacency and signs remain fixed during the initial curriculum.

The following artificial parameters train jointly from the first update:

- retina kernels and biases;
- sensory gains and biases;
- recurrent integration scalars already exposed by the reduced policy, where
  applicable; and
- the complete action readout/motor head.

The student loss combines:

- temperature-scaled KL distillation from teacher logits;
- the existing planner pairwise preference loss;
- an immediate-safety penalty for fatal actions; and
- a small trust penalty between DAgger rounds to prevent catastrophic drift.

No loss term may consume ObservationV4 inside the student forward path. The
structured observation exists only on the teacher/label side of the batch.
The student forward contract consumes `(rgb_frame, recurrent_state)` and returns
`(action_logits, neuron_activity, next_recurrent_state)`. `neuron_activity` has
exactly one value per graph neuron and is the only activity source eligible for
future runtime visualization.

### Population progression

V7 exposes four profiles:

- `smoke`: minimal data and one short update for contract, resource, checkpoint,
  and report validation;
- `80`: the first meaningful local candidate;
- `1k`: the measured 1,000-neuron expansion, started only after the 80-neuron
  candidate passes;
- `full`: the 165,122-neuron MaleCNS run, disabled unless a 1,000-neuron report
  contains a passing promotion decision.

Default CPU evidence budgets are:

| Profile | Training seeds | Validation seeds | Final-test seeds | DAgger rounds | Maximum steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| `smoke` | 2 | 3 | 3 | 1 | 32 |
| `80` | 64 | 21 | 100 | 4 | 200 |
| `1k` | 128 | 42 | 100 | 6 | 200 |
| `full` | 256 | 42 | 100 | 8 | 200 |

Training seed manifests may grow through DAgger, but validation and final-test
counts and seeds remain immutable for the run. Explicit research overrides are
recorded in provenance and create a distinct run identity.

The 80-to-1,000 progression uses the existing nested/gated measured-graph
relationship so the verified 80-cell core remains identifiable. Shape-compatible
retina parameters may be transferred. Population-specific sensory routing and
readout layers are trained through the same teacher targets rather than padded or
silently copied.

The full graph uses its declared `ol_sensory` and `vnc_motor` populations. It
reuses the dataset, teacher, retina architecture, loss definitions, and gates,
but it does not pretend reduced and full graph indices are interchangeable.

## Data and Information Boundaries

Each recorded transition contains:

- deterministic episode seed and step index;
- RGB neural frame;
- canonical ObservationV4 vector;
- planner preference values, acceptable-action mask, immediate-safety mask, and
  best action;
- teacher logits once a teacher checkpoint exists; and
- collection source (`planner`, `teacher`, or `student`) plus behavior action.

Training, validation, and test seeds are disjoint and recorded verbatim in the
report. Dataset generation is deterministic for a given configuration and seed
manifest. Validation and test transitions are never appended to training after
evaluation.

The batch API presents privileged observations to the teacher trainer and RGB
frames to the student trainer through separate typed fields. Student code must
not accept an observation tensor argument, making accidental leakage a contract
violation rather than a convention.

## Curriculum

### Stage 0: v11 baseline

Evaluate the current released 80-neuron compatibility controller on the V7
validation and test seed manifests. Record score, survival steps, terminal
reasons, action distribution, and step-limit episodes. This baseline is the only
runtime comparison used for promotion; historical environment-v6 metrics remain
context only.

### Stage 1: teacher

Collect planner trajectories, train the teacher offline, evaluate it closed-loop,
then run bounded DAgger correction rounds. Keep `best.pt` selected by the robust
validation tuple and `latest.pt` for resumption.

### Stage 2: 80-neuron student

Distill the accepted teacher into the RGB-only student with the measured 80-cell
graph fixed. Alternate bounded offline updates with student-state collection.
Evaluate after every round and roll back to `best.pt` when a round fails to
improve the robust validation tuple.

### Stage 3: 1,000-neuron student

Run only when the 80-neuron report says `promote`. Reuse the immutable dataset
manifest and teacher checkpoint. Add student-state data for the 1,000-neuron
candidate without changing validation or test seeds.

### Stage 4: full MaleCNS

Run only when the 1,000-neuron report says `promote`. The local laptop may run
`full --smoke`, but a production full curriculum should run on a CUDA host with
sufficient memory. CPU remains a supported fallback, not the expected production
path.

### Optional core-plasticity experiment

Core plasticity is out of the initial implementation. A later experiment may
unfreeze only measured edges outgoing from the sensory population after a fixed-
core student has passed its gate. It must use a new version and separate report;
it must not mutate an accepted fixed-core checkpoint in place.

## Evaluation and Promotion Gates

Every profile evaluates the same candidate on a deterministic suite of unseen
seeds. Candidate ordering is lexicographic:

1. step-limit/success episode count;
2. progress-qualified episode count;
3. mean score;
4. median score;
5. mean survival steps; and
6. lower fatal-action rate.

A candidate may replace the current best within a training stage only when it
improves this robust tuple. One exact seed is retained for diagnostics but has no
selection authority.

Promotion to the next population requires all of the following on held-out
validation seeds:

- mean score at least 10% above the predecessor/baseline;
- median score no lower than the predecessor/baseline;
- a strictly higher robust tuple;
- wins on at least 60% of paired seeds; and
- no decrease in step-limit/success episode count.

The teacher additionally must select a planner-acceptable action on at least 90%
of held-out states. A student additionally must select a planner-acceptable
action on at least 80% of held-out states. These agreement metrics cannot
override failed closed-loop criteria.

The final test suite is run once after configuration and checkpoint selection.
It never influences checkpoint selection. A failed final test produces `reject`
and cannot be relabelled as a validation success.

Reports expose exactly one next action:

- `promote`: every gate passed;
- `continue`: validation improved but the promotion margin was not reached; or
- `reject`: the candidate regressed, leaked privileged input, violated a
  resource/contract check, or failed final test.

## Resource and Artifact Policy

Reduced CPU profiles run inside the locked CPU Docker environment. Before a run,
the orchestrator records available disk, RAM, device, PyTorch version, and thread
count. It refuses to start when available disk is below:

- 2 GiB for `smoke` or `80`;
- 5 GiB for `1k`; or
- 15 GiB for `full`.

Each run directory may retain:

- `best.pt`;
- `latest.pt`;
- `report.json`;
- a small log; and
- a dataset manifest referencing shared immutable shards.

Superseded checkpoints are deleted only after the replacement is durably written
and load-verified. Shared dataset shards are content-addressed and reused across
profiles. Generated checkpoints, shards, and logs remain ignored by Git.

The `80` profile is designed to complete teacher plus first student evidence in
under one hour on the current i5-1335U laptop. This is a budget and reporting
target, not a reason to weaken evaluation gates. If the budget is exceeded, the
run checkpoints cleanly and reports `continue` with timing evidence.

## Checkpoint and Resume Contract

Teacher and student checkpoints include:

- V7 format/version;
- profile and stage;
- model and optimizer state;
- graph/artifact identifiers and hashes;
- teacher checkpoint hash for student runs;
- dataset manifest hash;
- train/validation/test seed-manifest hashes;
- RNG states;
- completed round/epoch counters; and
- current best validation metrics.

Resume validates every immutable identifier before restoring state. A mismatch
fails with a clear error and never silently starts a different experiment in the
same run directory. Loading untrusted checkpoints uses safe tensor-only loading
where the PyTorch format permits it.

## Reporting

The final `report.json` contains:

- provenance and resource snapshot;
- model parameter counts and trainable parameter names;
- per-round losses as diagnostics;
- per-seed closed-loop metrics;
- aggregate validation and test metrics;
- paired comparison against the predecessor;
- memory and elapsed-time measurements;
- selected checkpoint hash;
- every promotion predicate and its boolean result; and
- final decision and next action.

The report must explicitly distinguish simulated neural activity, teacher
behavior, student behavior, and planner labels.

## Failure Handling

- Non-finite loss, logits, gradients, or activity abort the current update,
  preserve the previous verified best checkpoint, and mark the run `reject`.
- Out-of-memory and low-disk conditions produce a resource failure with the last
  durable checkpoint path; they do not produce a partial `best.pt`.
- Empty/corrupt datasets, graph/hash mismatches, privileged-input leakage, and
  incompatible resume attempts fail before training.
- Action collapse is reported when one action exceeds 90% of held-out decisions;
  a collapsed candidate cannot be promoted.
- Interrupted runs remain resumable from `latest.pt` and never rewrite release
  artifacts.

## Components and Boundaries

V7 is a new package under `python/fly_crossy/v7/` with focused modules:

- dataset collection and immutable manifests;
- privileged teacher model/training;
- RGB-only visual connectome student;
- closed-loop evaluation and promotion rules;
- checkpoint/resource handling; and
- curriculum CLI/report orchestration.

The modules reuse canonical environment, camera, planner, graph, and action-order
implementations. They do not duplicate game rules. Existing V5/V6 modules remain
unchanged historical experiments.

The primary command shape is:

```text
python -m fly_crossy.v7.curriculum --profile smoke|80|1k|full --out <run-dir>
```

Explicit flags may override budgets for research, but no flag may bypass seed
separation, privileged-input boundaries, checkpoint provenance, or final-test
isolation.

## Testing Strategy

Unit tests must prove:

- retina routing is deterministic and local;
- student forward accepts RGB and has no privileged observation input;
- student forward returns one finite activity value per graph neuron;
- retina and motor-head parameters receive finite non-zero gradients;
- fixed measured adjacency and signs do not change during student training;
- teacher/student action order matches canonical `ACTION_ORDER`;
- paired promotion rejects a one-seed improvement and enforces every predicate;
- train/validation/test seed sets are disjoint;
- checkpoint resume reproduces the next update and rejects provenance mismatch;
- artifact retention never deletes the verified best before replacement; and
- resource preflight rejects insufficient disk without starting training.

The CPU Docker smoke test must generate a tiny dataset, train a teacher update,
train an 80-neuron student update, reload both checkpoints, run closed-loop
evaluation, and emit a schema-valid report without exceeding its resource
budget.

Before any runtime integration, the repository's full Python suite, frontend
tests, asset checks, build, Docker build, and Docker smoke test must pass as
specified in `docs/ACTIVE_CONTEXT.md`.

## Rollout

1. Implement contracts, promotion rules, and CPU smoke path with test-first
   development.
2. Run the 80-neuron local curriculum and inspect its held-out report.
3. If it says `promote`, run 1,000 neurons locally; otherwise iterate only on
   the failed stage identified by the report.
4. If 1,000 neurons says `promote`, prepare a separate full-run command and CUDA
   execution handoff.
5. Integrate a new controller into the browser/backend only after a separate
   review of final test evidence and provenance.

## Acceptance Criteria

The initial V7 implementation is complete when:

- the CPU smoke path is reproducible and fully tested;
- the teacher and student information boundary is enforced by their APIs;
- the 80-neuron profile can checkpoint, resume, and report within the local
  resource budget;
- promotion decisions are derived only from held-out robust metrics;
- failed candidates leave the released controller untouched; and
- the generated report is sufficient to decide whether to continue locally,
  redesign, or scale to 1,000 neurons without inspecting training code.
