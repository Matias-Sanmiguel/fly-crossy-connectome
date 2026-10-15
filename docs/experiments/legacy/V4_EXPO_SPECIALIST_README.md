# Crossy V4 — Expo Specialist

V4 deliberately targets the single expo geometry selected by the teacher gate:

```text
crossy-v4-expo:0006
```

The teacher already demonstrated 200/200 decisions on this seed at traffic phase
0.0, 0.4, 0.8 and 1.2 seconds. V4 therefore focuses on the student-side failures
found in V3 rather than changing reward shaping.

## What changes from V3

### RGB is preserved

The neural camera remains the existing 160x120 perspective renderer, but V4
resizes it to **48x24 RGB** instead of 64x32 grayscale:

```text
48 x 24 x 3 = 3456 visual features
4114 ol_sensory neurons
```

The fixed sensory map guarantees that every RGB feature is mapped to at least
one `ol_sensory` neuron; the remaining sensory neurons repeat random features.
No CNN, object detector or symbolic GameState input exists in the production
path.

### Recurrent state is really carried during training

V3 sampled arbitrary windows and initialized the 165,122-neuron state to zero
for every sampled window. V4 processes each training episode in chronological
chunks. At a TBPTT boundary the state value is preserved and only the gradient
graph is detached.

```text
episode start -> state=0
frames 0..15  -> detach -> frames 16..31 -> detach -> ...
```

### The target is no longer one-hot imitation

For every training state the offline teacher evaluates all five root actions.
The objective combines:

- acceptable-action set loss;
- pairwise planner preference ranking;
- immediate-fatal probability suppression;
- extra weight on rare/critical decisions.

`reward` is not used by this trainer.

### Training-only semantic supervision

A small auxiliary linear head reads the 708 VNC motor states during training and
must predict information already available from the authoritative environment:

- current lane type;
- next lane type;
- river support;
- the 25-value traffic/TTC radar.

This head exists only to push semantic/dynamic structure into MaleCNS activity.
It is not part of the expo inference path.

Production remains:

```text
48x24 RGB camera -> measured MaleCNS -> fixed VNC motor readout -> action
```

### Same seed, varied situations

Training always uses `crossy-v4-expo:0006`, but varies traffic phase and initial
lateral position and injects safe teacher exploration. DAgger then adds states
reached by the fly itself. This makes pure step-number memorization insufficient
while keeping the task specialized for the expo.

## Run

From the repository root after extracting the V4 patch:

```powershell
.\.venv\Scripts\Activate.ps1
cd python
pytest -q tests\test_v4_expo_specialist.py --basetemp .pytest-tmp
```

Run a tiny GPU smoke test first:

```powershell
python -m fly_crossy.v4.train_expo_specialist --device cuda --smoke
```

If the smoke finishes and produces `V4 COMPLETE`, run the full specialist:

```powershell
python -m fly_crossy.v4.train_expo_specialist --device cuda
```

Default full budget:

- 48 same-seed teacher/recovery episodes;
- 4 state-carried clone epochs;
- up to 2 DAgger rounds;
- 24 same-seed on-policy episodes per DAgger round;
- 2 state-carried epochs per DAgger round;
- 16-decision TBPTT windows, batch 4;
- Adam LR 0.02;
- semantic auxiliary weight 0.20.

DAgger stops early if the exact expo run reaches 200/200 and at least 18 of the
21 phase/lateral robustness variants also reach the step limit.

## Outputs

Full run:

```text
runs/crossy-v4-expo-specialist/best.pt
runs/crossy-v4-expo-specialist/latest.pt   # only when a later candidate is rejected
runs/crossy-v4-expo-specialist/config.json
runs/crossy-v4-expo-specialist/report.json
```

`best.pt` is never overwritten by a worse later stage. `latest.pt` is written
only when a candidate loses to the selected checkpoint, avoiding an unnecessary
second ~587 MB copy during successful stages while still preserving a rejected
latest stage for diagnosis.

Both `.pt` files are large and should remain ignored by Git. The JSON report is
the artifact to push for analysis.
