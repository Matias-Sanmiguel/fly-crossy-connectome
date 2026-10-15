MaleCNS Crossy V2 — PPO readout R1
===================================

Why the training objective changes here
---------------------------------------
The diagnostics no longer support a single structural bottleneck:

- broader MaleCNS readouts did not materially beat DN+VPN;
- short history / efference / lateral self-state did not pass the context gate;
- camera -> retina -> brain stage probes did not expose a large monotonic loss;
- even the privileged symbolic ObservationV4 ceiling did not cleanly recover
  the planner's single chosen action.

That pattern means continuing to train "predict exactly which action the depth-4
planner chose" is no longer justified.

The depth-4 planner is a model-predictive controller: it searches counterfactual
future game states. A one-step categorical imitation head is being asked to
compress that search into a single action label, including arbitrary/tie-like
choices.

This experiment therefore changes ONLY the learning objective.

Frozen forever in this run:
  perspective neural camera
  frozen visual frontend
  frozen 138,968-cell MaleCNS
  fixed global gain
  21,022 DN+VPN current+delta features

Trainable:
  ACTOR:
    exact R0 decoder architecture
      21,022 -> 256 GELU LayerNorm -> 5
    initialized from decoder-r0.pt

  CRITIC (training only):
      21,022 -> 128 GELU LayerNorm -> 1
    discarded after training

Algorithm:
  clipped PPO
  closed-loop sampled actions
  reward comes ONLY from authoritative step_game
  planner is NOT called anywhere in the training loop

Default budget:
  60 updates x 1024 steps = 61,440 environment steps
  training episodes reset at 200 steps so one strong seed cannot monopolize the run

Development checkpoint selection:
  every 10 updates on fixed, disjoint malecns-v2-ppo-dev seeds.
  The final 50-seed evaluation remains untouched until training is complete.

Important:
  This is ONE bounded PPO run.
  Do not immediately tune hyperparameters or launch PPO R2.

Run
---
From repository root:
  Expand-Archive "$env:USERPROFILE\Downloads\malecns-crossy-v2-ppo-readout-r1.zip" -DestinationPath . -Force

Then:
  cd python
  pytest -q tests/test_v2_ppo_readout.py --basetemp .pytest-tmp

Train:
  python -m fly_crossy.v2.ppo_readout --device cuda

Generated:
  artifacts/malecns-crossy-v2/decoder-ppo-r1.pt
  reports/malecns-crossy-v2-ppo-r1-training.json

Final closed-loop gate using the EXISTING evaluator:
  python -m fly_crossy.v2.decoder_eval `
    --device cuda `
    --checkpoint ..\artifacts\malecns-crossy-v2\decoder-ppo-r1.pt `
    --output ..\reports\malecns-crossy-v2-ppo-r1-closed-loop.json

This evaluates action-mode selection separately, then 50 completely unseen final
worlds and the depth-4 planner on those exact same final worlds.

STOP after that evaluation and inspect the result before any further training.
