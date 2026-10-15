MaleCNS Crossy V2 — planner preference/value distillation R1
================================================================

This is the bounded training experiment after BC, DAgger, structural audits,
context audits, visual-stage audits and PPO R1.

What changes
------------
Only the TRAINING TARGET changes.

Previous BC:
  planner chooses LEFT
  LEFT = 1
  every other action = 0

Preference distillation:
  preserve all five depth-4 root-action value tuples.

Each planner root-action value contains:
  alive at horizon
  score gain
  row gain
  mobility
  forward-safe
  centering

Loss
----
1. ACCEPTABLE SET
   Any action tied on the primary tuple:
     (alive, score gain, row gain)
   is valid.

2. PAIRWISE RANKING
   The lower-priority planner ordering is retained without pretending only one
   categorical action is correct.

3. IMMEDIATE-FATAL SUPPRESSION
   Offline authoritative step_game marks actions that die immediately.

There is NO one-hot planner cross-entropy.

Frozen
------
  perspective neural camera
  visual frontend
  138,968-cell MaleCNS
  global gain
  21,022 DN+VPN current+delta feature definition
  decoder architecture 21,022 -> 256 -> 5

Trainable
---------
  only the existing decoder weights, initialized from decoder-r0.pt.

Data
----
Fresh:
  64 train episodes x <=80 steps
  20 validation episodes x <=80 steps
  25% safe state-distribution exploration

Plus:
  the existing R0 DAgger student-state dataset, when present.
  Those exact states are replayed and relabeled with all five planner values.
  R0 disagreement states receive extra weight; immediately fatal R0 mistakes
  receive the largest weight.

Checkpoint selection
--------------------
Every 4 epochs, argmax gameplay runs on 12 fixed disjoint pref-dev worlds.

Selection order:
  1. mean score
  2. reached step limit
  3. median score
  4. mean length

Progress is intentionally first, unlike PPO R1's survival-first checkpoint
selection.

Run
---
From repository root:

  Expand-Archive "$env:USERPROFILE\Downloads\malecns-crossy-v2-preference-r1.zip" -DestinationPath . -Force

Then:

  cd python
  pytest -q tests/test_v2_preference_distill.py --basetemp .pytest-tmp

Train:

  python -m fly_crossy.v2.preference_distill --device cuda

Generated:

  artifacts/malecns-crossy-v2/preference-r1-train.npz
  artifacts/malecns-crossy-v2/preference-r1-validation.npz
  artifacts/malecns-crossy-v2/decoder-preference-r1.pt
  reports/malecns-crossy-v2-preference-r1-training.json

Final closed-loop gate
----------------------
Use the EXISTING evaluator unchanged:

  python -m fly_crossy.v2.decoder_eval `
    --device cuda `
    --checkpoint ..\artifacts\malecns-crossy-v2\decoder-preference-r1.pt `
    --output ..\reports\malecns-crossy-v2-preference-r1-closed-loop.json

STOP after that.

Do not run preference R2, PPO R2 or DAgger R2 before reviewing the 50 unseen
final worlds.
