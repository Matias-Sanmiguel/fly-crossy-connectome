# Environment v6: full-width static blockers

Date: 2026-09-16

## Motivation

Environment v5 allowed static grass blockers in column 0 and successfully
changed the 80-neuron controller's deterministic held-out behavior from a
wait/forward policy into a forward/left policy.

After 100,000 training steps, the 100-seed v5 validation run was:

- score mean: 9.15
- score median: 6
- score max: 38
- action distribution:
  - forward: 67.11%
  - left: 32.89%
  - backward/right/wait: 0%
- terminal reasons:
  - vehicle: 62
  - water: 22
  - train: 10
  - step-limit: 6

This demonstrated that center blocking worked, but exposed another guaranteed
static route: v5 only generated blockers in columns -4 through +4 while the
playable field extends through columns -5 and +5. A policy could therefore
move to one edge and use that edge as a permanently obstacle-free grass lane.

Because manual attempts past +/-5 are blocked rather than terminal, an edge
action can also advance time without moving. The existing reward assigns the
same stagnation cost to any non-progressing transition, so this behavior is
not separately distinguished from waiting.

## v6 change

Environment v6 changes only the static blocker distribution:

- candidates become every playable integer column:
  `[-5,-4,-3,-2,-1,0,1,2,3,4,5]`
- blocker RNG namespace changes from `obstacles:v2` to `obstacles:v3`

Each eligible grass row still receives only 1-3 blockers. Opening rows and
mandatory recovery grass remain blocker-free. Plants remain pass-through.

## Solvability

The existing bounded group solver still evaluates the exact seeded blockers
through the authoritative transition function. A generated group is accepted
only if a valid route reaches its recovery grass row; otherwise generation
retries and ultimately has the existing safe fallback.

## Experimental boundary

v4 and v5 remain diagnostic historical evidence and must not be relabeled as
v6. Final v6 training must start from fresh weights.

Before training:

1. regenerate parity fixtures
2. run focused TS/Python scenery/world/parity tests
3. run full TS and Python suites
4. run asset audit and production build
5. inspect fallback-rate audit
6. commit/tag the v6 boundary
7. train 80n from scratch
