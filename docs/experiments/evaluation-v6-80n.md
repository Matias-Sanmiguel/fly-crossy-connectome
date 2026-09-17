# Environment v6 / 80-neuron final evaluation

Date: 2026-09-16

## Frozen environment

- `WORLD_VERSION = 6`
- freeze commit: `486a8265cf793daf9e1ea024b1c147b42a7370b3`
- reward v2
- 370-value ObservationV1
- full-width static grass blockers (-5 through +5)
- TypeScript/Python parity and full QA passed before training

## Training protocol

Three independent reduced-connectome PPO candidates were trained from fresh
weights with:

- 1,000,000 total environment steps
- 8 parallel environments
- learning rate 0.0003
- 80-node reduced MaleCNS-derived fixed graph
- trainable engineered sensory/readout interfaces

Selection used only 100 fixed validation seeds with a 200-step limit.
The selection criterion was highest validation mean score.

Selected run: `final-80n-v6-train-1m-01`.

## Selection evidence

See:

`release/eval-v6/evaluation/selection-v6.json`

## Final test

After selection, the chosen checkpoint was evaluated once on 200 new seeds with
a 500-step limit.

See:

`release/eval-v6/evaluation/final-test-v6-200x500.json`

Final results:

- score mean: 19.13
- score median: 12.5
- score max: 82.0
- survival mean: 187.63
- survival median: 87.5
- survival max: 500.0
- step-limit: 55 / 200
- vehicle terminal: 88 / 200
- train terminal: 52 / 200
- water terminal: 5 / 200

The `final-test-v6-*` seeds are now sealed evaluation evidence and must not be
used for further tuning.

## Runtime identity

The browser autoplay policy is a byte-identical copy of:

`release/eval-v6/training/connectome/policy.json`

The biomechanical development runtime loads:

`release/eval-v6/training/connectome/checkpoint.pt`

`runtime-artifacts-biomechanics.json` pins the exact graph and checkpoint
SHA-256 values.

## Scientific scope

The controller preserves a measured reduced MaleCNS internal topology, but its
observation interface, sensory projection, readout, and PPO training are
engineered. It is not a full biological simulation.

The historical rewired-control result prevents claims that the measured
reduced topology is superior to matched alternative topologies.
