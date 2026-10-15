MaleCNS Crossy V2 — decoder R0
=================================

This is the first FINAL-CONTROLLER candidate.

Architecture:
  frozen perspective camera
  -> frozen visual front end
  -> frozen 138,968-cell MaleCNS
  -> 10,511 DN+VPN current rates + temporal delta
  -> ONE trainable 256-hidden-unit decoder
  -> 5 Crossy action logits

No connectome synapse is trained.
The planner is used only to provide offline labels/reference scores.

Install from repository root:
  Expand-Archive "$env:USERPROFILE\Downloads\malecns-crossy-v2-decoder-r0.zip" -DestinationPath . -Force

Then:
  cd python
  pytest -q tests/test_v2_decoder.py --basetemp .pytest-tmp

Train from the already-collected information-probe datasets:
  python -m fly_crossy.v2.decoder_train --device cuda

Then run the decisive closed-loop evaluation:
  python -m fly_crossy.v2.decoder_eval --device cuda

decoder_eval:
- selects argmax vs three sampling temperatures on held-out selection seeds;
- evaluates the selected action mode on 50 NEW final seeds;
- runs the depth-4 planner on those exact same worlds;
- reports neural performance as a ratio of the planner reference.

If R0 fails the closed-loop gate, do NOT change camera/retina/brain architecture.
The only planned learning intervention is one bounded DAgger round over student-visited states.
