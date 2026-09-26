Preference R1 - deterministic argmax final audit
================================================

The existing decoder evaluator selected sample:1.0 because its mode-selection
key prioritizes reached-step-limit before score.

On the preference R1 selection worlds:
  argmax:      mean 29.125, median 25, step limits 0
  sample:1.0:  mean 25.3125, median 20, step limits 1

So the 50-world final report used sample:1.0 even though argmax scored better.
This audit changes no weights and trains nothing. It evaluates deterministic
argmax on the exact same malecns-v2-decoder-final seeds and compares against the
same depth-4 planner.

From repository root:

  Expand-Archive "$env:USERPROFILE\Downloads\malecns-crossy-v2-preference-r1-argmax-final.zip" -DestinationPath . -Force

Then:

  cd python
  pytest -q tests/test_v2_preference_argmax_final.py --basetemp .pytest-tmp
  python -m fly_crossy.v2.preference_argmax_final --device cuda

Output:
  reports/malecns-crossy-v2-preference-r1-argmax-final.json

STOP after this audit. No R2 training is needed before reviewing it.
