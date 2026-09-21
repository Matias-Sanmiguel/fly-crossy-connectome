# Environment v8: Reward v4 carried-wait correction

Date: 2026-09-21

## Motivation

Environment v7 / Reward v3 penalized every non-progress step with
`STAGNATION_COST = -0.10`. A successful `wait` on a river log emits a
`carried` event and is the authoritative mechanism for inheriting log motion,
but it still received the same stagnation penalty as unproductive waiting.

Across the v7 capacity experiments, controllers entered and crossed rivers but
almost never used supported waits. Expanding the connectome from 80 to 1000
cells did not produce a reproducible supported-wait strategy.

## v8 change

World geometry, hazard generation, ObservationV2, action semantics, collision
rules, and all reward coefficients remain unchanged.

Reward v4 changes exactly one condition:

- a nonterminal step containing a `carried` event does not receive the
  stagnation penalty.

Therefore:

- ordinary wait without progress: `-0.11`
- blocked action without progress: `-0.21`
- forward progress by one new max row: `+0.99`
- successful carried wait on a log: `-0.01`
- carried wait that becomes terminal: `-10.11`

There is no river-crossing bonus and no positive reward for waiting on a log.
The change only removes the false classification of successful transport as
stagnation.

## Version boundary

- `WORLD_VERSION = 8`
- `WORLD_LAYOUT_VERSION = 6`
- Observation version remains `2`
- Reward metadata version is `4`

Environment v7 artifacts remain immutable historical evidence. v8 controllers
must be trained from fresh weights.
