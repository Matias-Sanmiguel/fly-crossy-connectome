# Environment v4 freeze

Date: 2026-09-16

## Purpose

Environment v4 is the gameplay boundary used for final controller training.
It supersedes the pre-freeze Environment v3 development state.

Historical Environment v3 release/eval-v1 artifacts remain historical and
must not be relabeled as v4.

## What v4 freezes

The following are part of the experimental environment and require a new
environment version plus retraining if changed:

- deterministic world/section generation
- seven-row group structure and recovery grass
- static grass blockers (trees/rocks) and blocking behavior
- pass-through decorative plants
- manual side-limit blocking semantics
- road, rail, river, vehicle, train, and log transition rules
- hazard sizes, speeds, densities, and difficulty progression
- 0.2 s controller decision interval
- ObservationV1 shape, encoding, normalization, and controller inputs
- action semantics/order
- reward v2:
  - progress +1.00
  - terminal -10.00
  - step -0.01
  - stagnation -0.05

## Visual changes allowed after freeze

Renderer-only changes are permitted only when they cannot change game state,
observations, action timing, collisions, rewards, or controller inputs.

Examples:
- camera angle/zoom
- lighting
- material colors
- purely decorative scenery
- fly mesh/material appearance
- purely visual wing animation
- road-marking visuals

## Why the version changes

Static scenery collision and manual boundary blocking were added while the
code still reported WORLD_VERSION=3. Environment v4 establishes a clean,
explicit experimental boundary before final training.

Because WORLD_VERSION participates in deterministic world seeding, v4 also
defines a new deterministic seed namespace. Parity fixtures must therefore be
regenerated and audited before training.

## Required validation before training

1. Regenerate shared parity fixtures.
2. Run the focused TypeScript world/environment/scenery tests.
3. Run the Python environment/scenery tests.
4. Run the full TypeScript test suite.
5. Run the full Python test suite.
6. Run the production web build.
7. Commit the resulting v4 boundary before starting final training.

Do not modify historical release/eval-v1 checkpoints, hashes, or evaluation
reports to make them appear compatible with v4.
