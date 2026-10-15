# Crossy V4 — Recurrent State Drift Audit

This audit freezes both the trained MaleCNS and the selected MLP64 decoder. It does **not train anything**.

It compares five ways of producing the 708-dimensional motor state:

- `stateful`: the normal persistent MaleCNS recurrent state.
- `reset`: rebuild from only the current frame (`history=1`).
- `short4`: rebuild from the latest 4 frames.
- `short8`: rebuild from the latest 8 frames.
- `short16`: rebuild from the latest 16 frames.

The audit has two complementary tests:

1. **Same-frame comparison**. It first records a teacher trajectory and the real stateful student trajectory. Every mode is then evaluated on those exact same frames and planner labels. This isolates the representation/history effect from environment divergence.
2. **Independent closed loop**. Every mode controls its own run from the exact expo start. This tests whether truncating history actually improves gameplay, not only offline metrics.

For each mode the report includes exact/acceptable/safe/progress action rates, early/middle/late trajectory segments, action agreement with the stateful controller, motor-state cosine/L2 drift, final score, steps, stagnation and terminal reason.

## Expected input checkpoints

```text
runs/crossy-v4-expo-specialist/best.pt
runs/crossy-v4-exact-seed-branch-curriculum/best_decoder.pt
```

## Run

From `python/`:

```powershell
pytest -q tests\test_v4_recurrent_state_drift_audit.py --basetemp .pytest-tmp
python -m fly_crossy.v4.recurrent_state_drift_audit --device cuda --smoke
python -m fly_crossy.v4.recurrent_state_drift_audit --device cuda
```

The full audit writes:

```text
runs/crossy-v4-recurrent-state-drift-audit/report.json
```

The most diagnostic comparison is:

```text
studentSameFrames.stateful.metrics.acceptableActionRate
vs
studentSameFrames.short{4,8,16}.metrics.acceptableActionRate
```

followed by the actual closed-loop scores under `closedLoopByMode`.

If a short-history mode materially improves the same-frame acceptable/safe rates **and** the independent closed-loop score, long recurrent history is hurting the controller. If stateful stays best, state drift is not the main bottleneck and the next change should move elsewhere (sensory representation/core or controller training objective).
