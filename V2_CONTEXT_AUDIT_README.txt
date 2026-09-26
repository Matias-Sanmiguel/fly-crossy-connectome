MaleCNS Crossy V2 — context sufficiency audit
===============================================

Result entering this stage
--------------------------
The structural brain readout audit did NOT support a readout bottleneck:
  DN+VPN compressed baseline:
    coreMacro ~0.533
    disagreement recovery ~0.367
    fatal critical correction ~0.408

  Best broader population gains:
    coreMacro       only about +0.012
    disagreement    only about +0.029
    fatal correction about +0.075

So simply reading more neurons is not justified.

Question
--------
Is the missing information short temporal/self-motion context that the current
snapshot readout does not expose?

This audit uses ONLY the existing 2,994 DAgger R0 states. It changes no model.

Every condition has the same 8,224-dimensional input and same 256-unit probe:
  1. brain_now
  2. brain_history4
     current plus previous 3 decision states (~800 ms)
  3. brain_now_efference
     plus previous 4 student actions
  4. brain_now_efference_column
     plus previous actions + current lateral body position
  5. brain_history4_efference_column
     history + motor copy + lateral self-state
  6. context_only_efference_column
     sanity check: no brain activity at all

Brain features are compressed label-free to 8,192 CountSketch channels.
The remaining 32 slots are fixed motor/self-state context slots.
All conditions use the exact same grouped episode folds and correction weighting.

Interpretation
--------------
Strong evidence for missing context:
  >= +0.08 OOF coreMacro versus brain_now
AND
  >= +0.08 disagreement recovery OR fatal critical correction.

The condition that crosses the threshold tells us what is missing:
  history only      -> decoder needs longer temporal context
  efference only    -> motor/efference-copy context is missing
  + column          -> lateral self-state/path integration is missing
  combined          -> both temporal and self-state context matter

If none crosses:
  do NOT add context channels blindly.
  Move upstream to camera/retina information sufficiency.

Run
---
From repository root:
  Expand-Archive "$env:USERPROFILE\Downloads\malecns-crossy-v2-context-audit.zip" -DestinationPath . -Force

Then:
  cd python
  pytest -q tests/test_v2_context_audit.py --basetemp .pytest-tmp
  python -m fly_crossy.v2.context_audit --device cuda

Output:
  reports/malecns-crossy-v2-context-audit.json
