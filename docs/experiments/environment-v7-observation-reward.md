# Environment v7: ObservationV2 + reward v3

Date: 2026-09-17

## Evidence motivating v7

The released 80-neuron v6 baseline was inspected visually and then evaluated
with 300 new diagnostic seeds (not the sealed final-test seeds).

Key diagnostic results:

- 21,770 left/right ping-pong returns
- 201/300 episodes had >=4 consecutive lateral alternations
- 102/300 episodes had >=100 decisions without increasing score
- 6,869 scenery blocks and 7,013 boundary blocks
- 11,295 repeated attempts against the same blocked target
- safe forward was ignored in 83.18% of one-step-safe opportunities
- 96 river sections encountered
- 5 river sections entered
- 0 river sections crossed
- safe river entry chosen only 6 / 11,346 times

The data show a learned failure mode, not only an 80-neuron capacity limit.

## v7 changes

World geometry and hazard generation are held fixed to the v6 layout by
separating:

- `WORLD_VERSION = 7`
- `WORLD_LAYOUT_VERSION = 6`

ObservationV2 keeps exactly 370 numeric inputs but changes semantics:

- static blocker is explicit cell code 8 instead of unknown 0
- cell normalization is `/8`
- train speed is normalized by 12
- car/truck/log speed is normalized by 5
- active policies must declare `observationVersion = 2`

Reward v3:

- progress: +1.00 per new max row
- terminal: -10.00
- step: -0.01
- stagnation: -0.10
- blocked action: -0.10 additional

No direct anti-left/right penalty and no river-specific bonus are introduced.
The goal is to remove the incentive for indefinite non-progress while leaving
the controller responsible for discovering valid navigation.

## Experimental boundary

`release/eval-v6/` remains immutable baseline evidence. v7 controllers must be
trained from fresh weights. `final-test-v6-*` seeds remain sealed and must not
be used for v7 tuning.
