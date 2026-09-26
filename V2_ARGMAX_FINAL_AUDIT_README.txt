MaleCNS Crossy V2 - deterministic ARGMAX final audit

Why:
The preference report selected sample:1.0 because it achieved 1/16 step-limits
during mode selection, even though argmax had higher mean score (29.125 vs 25.3125)
and median score (25 vs 20).

Before redesigning the controller, evaluate deterministic argmax on the exact
same 50 final seeds. No training, no changed weights.

Run from repository root:

Expand-Archive "$env:USERPROFILE\Downloads\malecns-crossy-v2-argmax-final-audit.zip" -DestinationPath . -Force
cd python
pytest -q tests/test_v2_argmax_final_audit.py --basetemp .pytest-tmp
python -m fly_crossy.v2.argmax_final_audit --device cuda

Output:
reports/malecns-crossy-v2-preference-r1-argmax-final.json

Stop after this report.
