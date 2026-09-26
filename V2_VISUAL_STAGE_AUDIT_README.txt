MaleCNS Crossy V2 — visual-stage information audit
====================================================

Where we are
------------
The context audit did NOT support short-history / efference / lateral-self-state
as the primary bottleneck. The strongest context candidate (brain + previous
actions + column) improved coreMacro only slightly and did not cross the
pre-registered joint threshold.

This audit moves exactly one stage upstream.

Question
--------
At what stage is expert-decision information disappearing?

Exact same 2,994 R0 DAgger states are replayed and compared:

  1. camera_now
     40x30 RGB current frame only.

  2. camera_rgb_delta
     same RGB + current-minus-previous RGB.
     This is a direct visual temporal/motion control.

  3. retina_sequence
     ALL 10 x 30,906 frozen retinal/optic-lobe rates received by MaleCNS
     during the 200 ms decision, compressed label-free to 7,200 channels.

  4. brain_readout
     exact 21,022 DN+VPN current+delta, compressed label-free to 7,200.

  5. symbolic_oracle
     the existing 517-value ObservationV4.
     IMPORTANT: this is a PRIVILEGED DIAGNOSTIC CEILING ONLY.
     It is never a candidate final controller input.

Fairness
--------
Camera / retina / brain all use exactly 7,200 features and a 256-hidden-unit
probe. All stages use:
  - same 5 episode-grouped folds,
  - same expert labels,
  - same R0 student states,
  - same correction-focused loss weighting.

The symbolic oracle keeps its native 517 values and the same hidden size. It
exists only to prove whether the planner labels are recoverable from a rich,
authoritative state representation.

Interpretation
--------------
retina >> brain:
  frozen MaleCNS dynamics are destroying / failing to preserve useful visual
  decision information.

camera RGB+delta >> retina:
  frozen retinal frontend is the bottleneck.

camera RGB+delta >> camera now:
  temporal visual change is necessary and must be represented correctly.

oracle >> camera:
  the neural camera does not expose enough state/hazard timing information
  compared with a privileged state representation. This does NOT authorize
  using ObservationV4 in production.

"Strong" means:
  >= +0.08 OOF coreMacro
AND
  >= +0.08 disagreement recovery OR fatal critical correction.

Run
---
From repository root:
  Expand-Archive "$env:USERPROFILE\Downloads\malecns-crossy-v2-visual-stage-audit.zip" -DestinationPath . -Force

Then:
  cd python
  pytest -q tests/test_v2_visual_stage_audit.py --basetemp .pytest-tmp
  python -m fly_crossy.v2.visual_stage_audit --device cuda

Output:
  reports/malecns-crossy-v2-visual-stage-audit.json

This replays all 120 student trajectories through camera -> retina -> MaleCNS,
so the collection phase may take a few minutes.
