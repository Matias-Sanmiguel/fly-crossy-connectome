# Crossy V4 Critical-State Surgery

This is intentionally the **last decoder-side diagnostic/intervention** in the V4 expo-specialist line.

It does two things in one run:

1. Measures separability of the first real critical student state in the normalized 708-dimensional MaleCNS motor state.
2. Immediately attempts a conservative correction instead of asking for another audit.

The MaleCNS stays fully frozen and stateful. The existing `mlp64` decoder is loaded from the best exact-seed branch curriculum checkpoint. During surgery, **only the final residual `64 -> 5` layer is trainable (325 parameters)**. The linear readout and the 708 -> 64 hidden layer remain frozen.

The surgical loss combines:

- planner preference loss on a very small set of actual student failure states;
- an explicit target-vs-bad-action margin at the current critical state;
- logit distillation on the already-working student prefix and a short teacher prefix;
- parameter-change regularization.

Three conservative recipes are tried from the same current decoder. A candidate is accepted only if it improves the exact deterministic expo seed **and** preserves at least 90% of the previous prefix actions. If accepted, the script advances to the newly exposed critical frontier and repeats.

## Hard stop

This experiment has an explicit stopping rule so the project does not become an endless chain of audits:

- If the decoder reaches the real expo target (`200` steps and score >= 80% of teacher), keep it and validate the expo build.
- If no surgical candidate improves the exact seed without prefix regression, stop decoder experiments and move to sensory/core work.
- If the maximum surgery rounds finish without real success, also move to sensory/core work.

The report writes this decision directly as `nextAction`.

## Run

```powershell
cd C:\Users\Nico\Documents\GitHub\fly-crossy-connectome
.\.venv\Scripts\Activate.ps1
cd python
pytest -q tests\test_v4_critical_state_surgery.py --basetemp .pytest-tmp
python -m fly_crossy.v4.critical_state_surgery --device cuda --smoke
python -m fly_crossy.v4.critical_state_surgery --device cuda
```

## Output

```text
runs\crossy-v4-critical-state-surgery\best_decoder.pt
runs\crossy-v4-critical-state-surgery\report.json
```

Important report fields:

- `initialSeparability`
- `rounds[*].separability`
- `rounds[*].candidates`
- `selectedClosedLoop`
- `stopReason`
- `nextAction`
