Preference R1 forced-argmax sanity check

Why: decoder_eval selected sample:1.0 because it was the only selection mode with 1/16 step-limit episodes, even though argmax had higher mean and median score on the selection set.

Run from repo root:
  Expand-Archive "$env:USERPROFILE\Downloads\malecns-crossy-v2-preference-r1-argmax-audit.zip" -DestinationPath . -Force
  cd python
  python -m fly_crossy.v2.preference_final_argmax --device cuda

Output:
  reports\malecns-crossy-v2-preference-r1-final-argmax.json

No training. Stop after this result.
