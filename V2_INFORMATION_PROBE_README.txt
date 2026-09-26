MaleCNS Crossy V2 — information probe
=========================================

Purpose
-------
This is a DIAGNOSTIC gate, not the final controller.

It asks one question:
Can a small supervised probe recover the planner's Crossy decision from the
FROZEN MaleCNS activity produced by the final perspective camera + retina?

Nothing inside the connectome or visual front end is trained.

Protocol
--------
- collect 64 training episodes and 20 completely separate validation episodes;
- each trajectory is mostly expert-driven, with 25% safe exploratory actions;
- EVERY visited state is labeled by the depth-4 offline planner;
- the exploration changes only which states are visited, never the neural input;
- feature = current DN+VPN rates plus their temporal delta;
- train linear probes on:
    DN only,
    VPN only,
    DN+VPN;
- train one 256-hidden-unit MLP diagnostic probe on DN+VPN;
- train a shuffled-label negative control;
- evaluate only on unseen seed prefixes.

Default feature dataset is written under artifacts/ and should remain ignored by Git.

Run
---
From repository root:
  Expand-Archive "$env:USERPROFILE\Downloads\malecns-crossy-v2-information-probe.zip" -DestinationPath . -Force

Then:
  cd python
  pytest -q tests/test_v2_information_probe.py --basetemp .pytest-tmp
  python -m fly_crossy.v2.information_probe --device cuda

This can take a few minutes because it renders real Crossy states, runs the
frozen retina + 138,968-cell MaleCNS, and asks the planner for every label.

If collection completes but training needs to be rerun:
  python -m fly_crossy.v2.information_probe --device cuda --reuse-dataset

PASS means information is recoverable. It does NOT yet mean the final decoder
plays well closed-loop.
