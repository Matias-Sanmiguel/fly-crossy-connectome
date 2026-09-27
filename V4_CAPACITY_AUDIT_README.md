# Crossy V4 capacity audit

This is a diagnostic stage. It does **not** retrain the 25.7M-parameter MaleCNS.
It is designed to answer where V4 stops being able to imitate the teacher before
another multi-hour run is attempted.

## What it measures

1. **Teacher-forced exact trajectory**
   - The environment advances with the teacher for all 200 decisions.
   - The fly sees the exact same RGB sequence.
   - We measure exact action agreement, teacher-acceptable action rate,
     immediate-safe rate, first divergence, and first unsafe action.
   - A student error cannot corrupt later states in this test.

2. **Closed-loop exact trajectory**
   - The fly acts normally.
   - The teacher only labels each state reached by the fly.
   - This exposes covariate shift and recovery failures.

3. **Frozen motor decoder probe**
   - The full trained MaleCNS is frozen.
   - Teacher-forced motor states are collected at phase offsets
     0.2, 0.4, ..., 1.2 s.
   - Only a tiny 708 -> 5 linear decoder is fit.
   - Phase 0.0 is held out.
   - If this probe strongly beats the frozen random V4 decoder, the output
     interface is a bottleneck candidate.

4. **Fresh semantic probe**
   - Again the MaleCNS is frozen.
   - A fresh 708 -> 34 linear probe is trained on nonzero phase offsets and
     evaluated on the held-out exact trajectory.
   - It measures current-lane accuracy, next-lane accuracy, support accuracy,
     and traffic-radar MSE.

5. **Probe-decoder closed loop**
   - The same MaleCNS is run in the real game using only the newly fit linear
     motor decoder.
   - No edge gain, leak, sensory mapping, or connectome weight is changed.

## Run

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
cd python

pytest -q tests\test_v4_capacity_audit.py --basetemp .pytest-tmp

python -m fly_crossy.v4.audit_capacity --device cuda
```

This should be much cheaper than V4 training: there is no full-connectome
backpropagation. Most of the wall time is teacher/planner labeling.

The report is written to:

```text
reports\crossy-v4-capacity-audit.json
```

Do not overwrite or delete:

```text
runs\crossy-v4-expo-specialist\best.pt
```

The audit reads that checkpoint but never modifies it.
