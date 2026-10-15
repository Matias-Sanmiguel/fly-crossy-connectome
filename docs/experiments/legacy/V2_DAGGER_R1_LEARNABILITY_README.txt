MaleCNS Crossy V2 — DAgger-state learnability gate
====================================================

This is the LAST diagnostic before deciding whether to spend the one allowed
production DAgger fine-tune round.

It DOES NOT modify decoder-r0.pt.

Protocol
--------
Dataset:
  artifacts/malecns-crossy-v2/dagger-r1-student-states.npz

Five grouped cross-validation folds:
  - split strictly by episode;
  - 96 episodes train / 24 validation per fold;
  - every student state is out-of-fold exactly once.

Probe architecture:
  exact same readout architecture as R0:
    21,022 -> 256 GELU LayerNorm -> 5

Diagnostic loss weighting:
  - sqrt inverse expert-action frequency;
  - disagreement states x2;
  - immediately-fatal critical errors an additional x2;
  - total sample weight clipped to [0.25, 12].

Reported out-of-fold metrics:
  - expert-action accuracy/coreMacro;
  - LEFT/RIGHT/WAIT;
  - recovery specifically on R0 disagreement states;
  - correction rate on the 120 immediately-fatal R0 critical errors;
  - vehicle/train/water critical-error correction;
  - last 1/3/5 states before each death;
  - improvement over R0 on exactly the same student-state dataset.

PASS thresholds
---------------
  OOF coreMacro >= 0.60
  LEFT >= 0.50
  RIGHT >= 0.50
  WAIT >= 0.40
  disagreement recovery >= 0.50
  critical correction >= 0.50
  coreMacro improvement over R0 >= +0.05

PASS:
  consume the single bounded DAgger fine-tune round.

REVIEW:
  do not do repeated DAgger. Re-open representation/readout diagnosis instead.

Run
---
From repository root:
  Expand-Archive "$env:USERPROFILE\Downloads\malecns-crossy-v2-dagger-r1-learnability.zip" -DestinationPath . -Force

Then:
  cd python
  pytest -q tests/test_v2_dagger_probe.py --basetemp .pytest-tmp
  python -m fly_crossy.v2.dagger_probe --device cuda
