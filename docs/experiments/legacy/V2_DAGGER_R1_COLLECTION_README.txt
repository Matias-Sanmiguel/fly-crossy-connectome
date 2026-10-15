MaleCNS Crossy V2 — bounded DAgger R1 collection
=================================================

This stage DOES NOT TRAIN anything.

R0 argmax drives every actual game step.
At each pre-action state the depth-4 planner provides an OFFLINE expert label.

The collector stores:
- the exact 21,022 DN+VPN current+delta features seen by R0;
- R0 logits and action;
- planner action;
- disagreement;
- immediate student/expert terminal outcome;
- "critical error" = student action dies immediately while planner action survives;
- current/forward/destination lane kinds;
- episode/step/score/position;
- terminal reason and steps until terminal;
- R0 confidence/margin/entropy.

It also reports:
- on-policy agreement and balanced core-macro agreement;
- confusion matrix;
- critical errors by death cause;
- disagreement by lane;
- disagreement/confidence in the last 1/3/5 decisions before death;
- comparison with the original held-out validation metrics when the training report exists.

Install from repository root:
  Expand-Archive "$env:USERPROFILE\Downloads\malecns-crossy-v2-dagger-r1-collect.zip" -DestinationPath . -Force

Then:
  cd python
  pytest -q tests/test_v2_dagger_collect.py --basetemp .pytest-tmp
  python -m fly_crossy.v2.dagger_collect --device cuda

Default:
  120 R0 episodes
  max 120 steps
  completely new seed prefix: malecns-v2-dagger-r1

Generated feature data remains under artifacts/ and should stay ignored by Git:
  artifacts/malecns-crossy-v2/dagger-r1-student-states.npz

Report:
  reports/malecns-crossy-v2-dagger-r1-collection.json

STOP after collection. Do not fine-tune until the report has been inspected.
