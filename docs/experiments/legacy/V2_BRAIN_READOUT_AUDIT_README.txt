MaleCNS Crossy V2 — structural brain readout audit
===================================================

Why this exists
---------------
The DAgger-state learnability gate failed:
  - OOF coreMacro about 0.509
  - LEFT about 0.392
  - RIGHT about 0.511
  - WAIT about 0.220
  - disagreement recovery about 0.351
  - fatal critical correction about 0.400

So we DO NOT fine-tune R1 yet.

This audit asks the structural question:
  Does the needed decision information exist elsewhere inside the frozen MaleCNS,
  but get lost because R0 reads only DN+VPN?

Protocol
--------
Replays the exact same 120 R0 DAgger episodes. It aborts if R0 action replay
does not exactly match the stored student action.

For every exact same state it compares four recurrent populations:
  1. dn_vpn_readout  -- current 10,511-cell readout
  2. near_dn_hop1   -- dynamic cells <=1 backward hop from descending output
  3. near_dn_hop2   -- dynamic cells <=2 backward hops from descending output
  4. all_dynamic    -- all 108,062 recurrent/dynamic cells

IMPORTANT FAIRNESS CONTROL:
Every population is compressed WITHOUT LABELS using deterministic CountSketch to
the SAME feature width:
  4,096 current-rate features
  4,096 temporal-delta features
  = 8,192 total

Every group then uses:
  same 5 episode-grouped folds
  same expert labels
  same correction-focused weights
  same 8,192 -> 256 -> 5 probe capacity

No production weight is changed.

Interpretation
--------------
Strong evidence for a READOUT bottleneck requires:
  broader population coreMacro >= readout coreMacro + 0.08
AND
  disagreement recovery OR fatal critical correction >= readout + 0.08

If that happens:
  the selected MaleCNS internally has useful information that DN+VPN throws away.

If it does not:
  do not keep replacing readouts blindly; representation/input/state context becomes
  the leading suspect.

Run
---
From the repository root:
  Expand-Archive "$env:USERPROFILE\Downloads\malecns-crossy-v2-brain-readout-audit.zip" -DestinationPath . -Force

Then:
  cd python
  pytest -q tests/test_v2_brain_readout_audit.py --basetemp .pytest-tmp
  python -m fly_crossy.v2.brain_readout_audit --device cuda

Output:
  reports/malecns-crossy-v2-brain-readout-audit.json

This can take several minutes because it replays all 120 exact student trajectories
through camera -> retina -> 138,968-cell MaleCNS and then performs grouped probes.
