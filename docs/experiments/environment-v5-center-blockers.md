# Environment v5: center blockers

Date: 2026-09-16

## Motivation

Environment v4 was frozen and a 100,000-step 80-neuron diagnostic training run
was completed. A deterministic held-out evaluation over 100 seeds produced a
policy dominated by `wait` and `forward`, with effectively no right movement
and almost no left movement.

The v4 static grass-obstacle generator intentionally excluded column 0. Because
the fly starts on column 0, that design guaranteed a spatially open center line
through every grass obstacle row. The bounded dynamic hazards still required
timing, but the static scenery never forced a lateral maneuver.

## v5 change

Environment v5 makes one authoritative gameplay change:

- interior static obstacle candidates change from
  `[-4,-3,-2,-1,1,2,3,4]`
  to `[-4,-3,-2,-1,0,1,2,3,4]`
- the blocker RNG namespace changes from `obstacles:v1` to `obstacles:v2`

Opening rows and mandatory recovery grass remain obstacle-free. Each eligible
grass row still has only 1-3 static blockers.

## Solvability

This does not remove the solvability guarantee. Generated groups are still
accepted only when the existing bounded solver finds a route using the exact
authoritative transition function and the exact seeded static blockers.

## Experimental boundary

v4 remains historical evidence and its diagnostic checkpoint must not be
relabeled as v5.

Before v5 training:

1. regenerate environment/world parity fixtures
2. run focused TS/Python scenery, world, observation, and transition tests
3. run the full TS and Python suites
4. run the production build
5. inspect the deterministic generation audit/fallback rate
6. commit/tag the validated v5 boundary
7. retrain 80n from scratch
