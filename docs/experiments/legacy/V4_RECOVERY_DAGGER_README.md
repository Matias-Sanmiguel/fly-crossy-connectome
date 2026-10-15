# Crossy V4 — Recovery DAgger

This stage follows the capacity audit. It does **not** retrain the measured
MaleCNS. The V4 `best.pt` remains frozen in its entirety.

The capacity audit showed that the frozen motor state is linearly informative
about lane/support semantics and that a better 708→5 readout improves
teacher-forced action prediction, while closed-loop play still collapses. The
working hypothesis is therefore covariate shift: after a small mistake the fly
visits states that are not on the teacher trajectory and errors compound.

## What this experiment trains

Only a tiny normalized linear motor decoder:

```text
48x24 RGB -> frozen MaleCNS (165,122 neurons / 25,563,197 edges)
          -> 708 vnc_motor states
          -> TRAINABLE Linear(708, 5)
          -> Crossy action
```

The connectome edge gains, leaks, sensory mapping and recurrent dynamics are
all frozen.

## Recovery data

Three replay sources are mixed:

1. **Teacher anchors** from the expo seed with phase/column variation. The exact
   `(phase=0, column=0)` expo trajectory is withheld from these anchors.
2. **Recoverable perturbations**. The teacher occasionally executes an action
   that is immediately safe but preferably outside the planner's acceptable
   set. The following states are marked as recovery states and receive more
   training weight.
3. **On-policy DAgger**. The current decoder plays; every visited state is
   labelled by the depth-4 planner and added to replay. Unacceptable and unsafe
   student choices receive strong weight.

Training uses the existing preference objective (acceptable-set mass,
pairwise planner preference and fatal-action suppression), not reward shaping.

## Install and test

Extract the patch at the repository root, then:

```powershell
cd C:\Users\Nico\Documents\GitHub\fly-crossy-connectome
.\.venv\Scripts\Activate.ps1
cd python

pytest -q tests\test_v4_recovery_dagger.py --basetemp .pytest-tmp
python -m fly_crossy.v4.recovery_dagger --device cuda --smoke
```

The smoke run uses only 40 steps, one recovery round and a tiny training budget.
It is a plumbing check only.

## Full run

```powershell
python -m fly_crossy.v4.recovery_dagger --device cuda
```

Defaults:

- 12 teacher-anchor episodes;
- 16 deliberate perturbation episodes;
- 4 on-policy recovery rounds;
- 12 on-policy episodes per round;
- 100 decoder epochs per round;
- 200 game decisions maximum;
- same expo seed: `crossy-v4-expo:0006`;
- full 21-variant robustness evaluation after each round.

This should be much cheaper than full V4 training because backpropagation never
passes through the 25.5M measured connectome edges.

## Outputs

```text
runs/crossy-v4-recovery-dagger/best_decoder.pt
runs/crossy-v4-recovery-dagger/round-1.pt
...
runs/crossy-v4-recovery-dagger/report.json
```

`best_decoder.pt` is selected by closed-loop performance, not by training loss.

The report tracks three signals separately:

- **teacher-forced action quality** on the exact expo trajectory;
- **semantic probe quality on states actually visited by the student**;
- **closed-loop survival/score** on the exact seed and 21 phase/column variants.

Interpretation:

- semantic quality remains high on student states + closed-loop improves:
  recovery/readout was the main bottleneck;
- semantic quality remains high + action metrics improve but closed-loop does
  not: recovery data/objective still needs work;
- semantic quality collapses specifically on student states: the frozen
  sensory/core representation is out-of-distribution and the next experiment
  should target the interface/core rather than more decoder training.
