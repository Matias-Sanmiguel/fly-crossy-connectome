# Active project context

Last updated: 2026-09-16

## Current development phase

We are intentionally PAUSING expansion from the 80-neuron controller to:

- 1,000
- 5,000
- 20,000
- 124,289 neurons

Current priority:

GAME POLISH -> GAMEPLAY FREEZE -> FINAL 80N TRAINING -> POPULATION PIPELINE

Do not begin the 1k-neuron population work unless explicitly requested.

---

## High-level target

The final installation should demonstrate:

connectome controller
-> FlyBody biomechanical body in MuJoCo
-> physical leg presses W/A/S/D keyboard keys
-> native contact confirmation
-> authoritative Crossy-style browser game movement

The browser renderer/game is being polished before larger controllers are
trained so every final controller is trained against the same frozen game.

---

## Verified biomechanical milestones

Already working and tested:

- FlyBody and physical keyboard exist in one MuJoCo world.
- Direction mapping:
  - forward -> W -> front_left
  - backward -> S -> front_right
  - left -> A -> middle_left
  - right -> D -> middle_right
- Physical contact gate requires correct key + correct tarsus + travel + force.
- Browser movement waits for physical `action_result`.
- Requested/intended actions cannot directly move the browser game.
- Recovery no longer normally requires reset.
- Recovery path uses a vertical Cartesian lifting phase before retracting.
- All four directional keys physically recover successfully.
- Motor lifecycle includes `LIFTING`.
- Browser session auto-restarts after terminal episodes.
- Python full suite was green after these changes.

Do not bypass or simplify this physical authority chain.

---

## Browser architecture

`src/main.tsx` switches between:

Normal:

`<App />`

Biomechanical:

`?biomechanics=1` -> `<BiomechanicsDebugApp />`

Both use the SAME `GameScene`.

Therefore renderer/visual improvements made to `GameScene` apply to both modes.

Normal mode is preferred for game development and visual QA because it does not
require waiting for MuJoCo physical actions.

Biomechanics mode is used to validate the final physical end-to-end chain.

---

## Neural controller status

An 80-neuron fixed-connectome PPO controller has been trained and physically
validated end-to-end.

Do not describe it as the full MaleCNS controller.

Planned final controller sizes:

- 80
- 1,000
- 5,000
- 20,000
- 124,289

Those larger populations are NOT implemented/trained yet.

### Reward v2

Current intended reward:

- progress: +1.00
- terminal: -10.00
- step: -0.01
- stagnation: -0.05

Reward v2 was introduced because reward v1 allowed a degenerate strategy that
waited approximately 85.8% of decisions.

80n-v2 held-out behavior:

- score mean: 14.417
- median: 13
- max: 34
- mean survival: 74.25
- wait frequency: 0.166
- forward frequency: 0.194
- left frequency: 0.319
- right frequency: 0.321

This was a clear behavioral improvement over 80n-v1.

However this is NOT the final 80n model because gameplay is currently being
changed before environment freeze.

Scientific caution:

The degree/normalization-matched rewired control scored higher than the real
reduced-connectome graph in this bounded evaluation.

Do not make biological superiority claims.

---

## Current environment state

Current `WORLD_VERSION` is 4.

Environment v4 is the frozen experimental gameplay environment for final
controller training.

The freeze boundary includes:

- deterministic section/group generation
- static grass blockers and their collision semantics
- manual side-limit blocking
- moving hazard dynamics
- observation schema and normalization
- action semantics and decision interval
- reward v2

From this point until the final controller/evaluation series is complete, do
not change gameplay rules, observations, rewards, collision semantics, hazard
speeds/densities, or world generation without creating a new environment
version and retraining.

Renderer-only changes remain allowed when they cannot affect game state,
observations, actions, timing, or controller inputs. Examples include camera,
lighting, colors, decorative meshes, and purely visual animation.

Required sequence from this freeze:

1. regenerate v4 parity fixtures
2. run complete TypeScript/Python QA
3. commit/tag the frozen environment boundary
4. smoke-train the final 80n controller
5. run the full final 80n training/evaluation
6. only then expand populations

---

## Game-polish stages 1–5 completed

The first five approved game-polish stages are implemented, but the environment
is NOT frozen yet.

### Section-based world generation

World generation now uses deterministic seven-row groups:

- six content rows selected from approved road, rail, or river section templates
- one mandatory recovery grass row
- road sections are dominant
- rail sections are occasional and one row long
- river sections are rare and exactly two rows long
- river rows use exactly five moving logs with logical sizes 2–4
- direct transitions between unlike challenge lane kinds are eliminated

The selected bounded solver strategy uses a deterministic global
progress-prioritized frontier. It expands states by:

1. greatest reached row
2. shallowest action depth
3. insertion order

Action priority is forward, left, right, wait, backward. No action is pruned,
the authoritative transition remains unchanged, and the existing limits remain
80 steps, 1,024 checked transitions, and eight generation attempts.

On the fixed 30-seed x 210-row audit corpus:

- overall all-grass fallback: 0 / 900 groups
- river-template fallback: 0 / 123 groups
- section frequencies: 70.17% road, 23.00% rail, 6.83% river
- row frequencies: 36.71% grass, 52.81% road, 6.57% rail, 3.90% river
- maximum river run: 2
- direct unlike challenge transitions: 0

TypeScript and Python generation remain exactly matched. The world and
environment parity fixtures were regenerated and the named
`remainder-boundary-water` case was retargeted to a seed that still exercises
river behavior at the negative circuit boundary.

### Lane scenery and surfaces

- Generic per-row traffic lights were removed from road and rail scenery.
  The asset remains registered but is not placed by the generic scenery system.
- Procedural substrates now cover the full 1.0 row depth, eliminating the
  previous 0.06 empty gap between adjacent lane rows.
- `road-straight.glb` is normalized as a one-row tile and instantiated as 25
  pooled tiles at X positions -12 through 12 for each visible road row.
- A seamless dark road substrate remains underneath for loading, failure, and
  crack prevention.
- `track-detailed.glb` uses the same pooled 25-tile row architecture for rail.
- The seamless rail substrate remains underneath.
- Procedural rails and sleepers render only while the Kenney track asset is
  unavailable; they are disabled when the detailed track is active.

Logical road, rail, hazard, and collision geometry did not change in these
visual stages.

Manual visual QA is still required for tile orientation, marking continuity,
track height, material seams, and loading/failure transitions before gameplay
freeze.

---

## Kenney visual work completed

A curated Kenney asset library is now integrated.

34 selected GLB files are registered and validated with:

- local byte count
- SHA-256
- original pack
- archive SHA-256
- upstream path
- CC0-1.0 license

Relevant files include:

- `src/game/kenneyAssets.ts`
- `src/game/kenneyLoader.ts`
- `src/game/hazardVisuals.ts`
- `src/game/createGameRenderer.ts`
- `public/assets/kenney/manifest.json`
- `scripts/generate-kenney-manifest.mjs`
- `scripts/check-assets.mjs`

Visual vehicle variants are selected deterministically.

### Car visual pools

Logical `car` can visually appear as:

- sedan
- sedan-sports
- hatchback-sports
- suv
- suv-luxury
- taxi
- police

Logical `truck` can visually appear as:

- truck
- van
- delivery
- garbage-truck
- ambulance
- firetruck

The visual model MUST NOT change the logical collision/gameplay class.

Vehicle variety has been visually reviewed and is broadly approved.

---

## Current visual/gameplay findings

Recent manual visual QA identified the following remaining issues.

### 1. Section generation completed

The excessive-rail issue has been replaced by the deterministic section grammar
documented above. Rail sections are one row long, river sections are two rows
long, and the shared TypeScript/Python fixtures reflect the new architecture.

---

### 2. Trains are not yet complete

Only one visual train currently appears.

We already imported curated Train Kit assets including multiple locomotives and
freight carriages.

Desired train experience is closer to Crossy Road:

empty/safe rail
-> warning
-> fast long train crosses
-> train clears
-> rail safe again

Train rendering should become a deterministic composition of:

- one locomotive
- several carriages

Potential locomotives:

- train-diesel-a
- train-diesel-b
- train-diesel-c
- train-locomotive-a

Available freight carriages include:

- box
- coal
- container-blue
- dirt
- flatbed-wood
- lumber
- tank-large
- wood

There is already a `train-warning` event and warning window in the transition
logic.

Reuse authoritative mechanics rather than inventing renderer-only state.

This area may intentionally change gameplay before environment freeze.

---

### 3. Grass scenery currently has no gameplay collision

Trees, rocks, and plants are generated by `scenery.ts` and are currently purely
visual.

The logical environment does not know they exist, so the fly can move straight
through trees and rocks.

Desired behavior:

- tree: blocks movement
- rock: blocks movement
- small plant: may remain decoration only

Attempting to hop into a blocking grass obstacle should prevent the movement,
not kill the fly.

This MUST be implemented authoritatively.

Do not implement only a Three.js collision hack.

It must also be represented consistently in training/observation if it becomes
part of final gameplay.

---

### 4. River logs are placeholders

Logs are currently procedural box-like hazards.

No external log model was selected.

Preferred direction is a small custom low-poly procedural log:

- low-sided cylinder
- brown body
- slightly lighter end caps
- deterministic variation
- retain existing authoritative logical `log`

This can remain a renderer-only improvement if dimensions do not alter
gameplay.

---

## Immediate work order

Current intended order:

1. improve lane type distribution / reduce excessive rail clustering
2. make trees and rocks authoritative blocking obstacles
3. implement deterministic locomotive + carriage train rendering and final train
   behavior
4. improve procedural river log visuals
5. manually inspect long sections of the world in Human/manual mode
6. then address game feel:
   - hop timing
   - camera
   - death feedback
   - splash
   - train warning feedback
   - sound
   - restart transitions
   - UI feedback
7. environment freeze
8. final 80n training
9. population pipeline

Do not jump ahead unless explicitly asked.

---

## Current QA workflow

For visual/gameplay inspection use the normal app and select:

Controller -> Human / manual

This allows manual traversal without depending on the currently weak autonomous
controller.

Use `?biomechanics=1` only when validating the physical controller chain.

---

## Reproducibility rules

- Never use nondeterministic visual selection for world objects.
- Cosmetic model choice should derive from seed/world identity.
- A visual asset variant must not silently modify logical hitboxes.
- Changes to authoritative world behavior require TypeScript/Python parity
  updates.
- Keep generated manifests and fixtures reproducible.
- External asset provenance must remain auditable.
