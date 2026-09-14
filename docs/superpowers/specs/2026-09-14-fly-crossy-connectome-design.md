# Fly Crossy Connectome — Design Specification

**Date:** 2026-09-14  
**Status:** Approved in conversation  
**Project:** `fly-crossy-connectome`

## 1. Purpose

Build an endless, isometric 3D crossing game inspired by Crossy Road in which a fruit fly can be controlled either by a person or by an experimental connectome-constrained controller. The game must keep a 3D MaleCNS brain visualization visible alongside the playfield and make the origin and meaning of every displayed signal explicit.

The project is both a playable browser game and a reproducible controller experiment. It must not claim that simulated activity is measured biological activity or that the controller reproduces animal cognition.

## 2. Scope

### First release

- Endless procedural 3D game with an isometric camera and a low-poly visual style.
- Human controls using keyboard and touch.
- Conventional neural baseline using the same observations and actions as the connectome controller.
- Reduced MaleCNS-derived recurrent controller with fixed internal topology and trainable sensory and motor interfaces.
- Always-visible 3D brain atlas and synchronized decision telemetry.
- Deterministic seeds for repeatable play, training, and evaluation.
- Python training and evaluation tools that export compact browser-compatible models.
- A stable controller protocol that can later connect a full MaleCNS service.

### Deferred

- Running or training the full MaleCNS graph inside the browser.
- A hosted GPU training service.
- Accounts, leaderboards, multiplayer, cosmetics, purchases, and cloud persistence.
- Claims of biological fidelity beyond the explicitly sourced connectivity constraint.

## 3. User Experience

### Layout

Desktop uses a two-column laboratory layout:

- The game occupies roughly 60 percent of the available width.
- The brain visualization and telemetry occupy roughly 40 percent.
- The brain remains visible while playing, paused, or viewing the game-over state.
- A top control bar exposes controller mode, seed, score, pause, reset, and repeat-seed actions.

On narrow screens, the game and brain panels stack vertically. Both remain present; the brain is not hidden behind a separate route or modal.

### Core loop

1. The player selects Human, Conventional AI, or Connectome mode.
2. A seeded world is created and the fly begins on a safe grass row.
3. The controller chooses one discrete action per decision interval.
4. The fly makes a short hop or hover of exactly one grid cell, or waits.
5. The world resolves vehicle motion, platform motion, collisions, progress, and reward.
6. The brain view and telemetry display the observation, selected action, reward, and model provenance for that same simulation step.
7. The run continues until the fly is struck, falls into water, leaves a valid moving platform, or is caught by another terminal hazard.
8. The user can begin a new seed or repeat the same seed immediately.

The fly cannot bypass the game by flying freely above hazards. Its wings provide the visual explanation for one-cell hops while preserving Crossy Road rules.

### Controls

- Keyboard: arrow keys and WASD for movement; Space waits for one decision step; Escape pauses or resumes.
- Touch: four directional controls plus wait, positioned without covering the playfield.
- Autonomous modes: pause, resume, speed selection, reset, and repeat-seed remain available.

## 4. World and Rules

### Coordinate model

The simulation uses integer grid coordinates independent of Three.js. Increasing row index means forward progress. Rendering interpolates between authoritative simulation states but never determines game outcomes.

### Procedural generation

The world is generated in deterministic chunks from a user-visible seed. A given version, seed, and action sequence must produce the same lane layout and outcomes.

Initial lane types are:

- Safe grass rows.
- Road lanes with cars and trucks.
- Railway lanes with warning signals and trains.
- River lanes with moving logs or equivalent platforms.
- Short safe breaks between hazardous groups.

Difficulty rises with forward distance by adjusting lane-group length, obstacle speed, obstacle spacing, direction combinations, and safe-gap frequency. Generation must always obey explicit solvability constraints; random generation must not knowingly create an impassable row group.

### Movement and collisions

Actions are `forward`, `backward`, `left`, `right`, and `wait`. Only one action is accepted per decision step. Movement has a short visual tween, but collision checks use the simulation state and swept hazard positions so fast obstacles cannot pass through the fly undetected.

On a river platform, the fly inherits platform displacement until it moves again. Leaving the navigable horizontal bounds or losing platform support is terminal.

### Score and reward

The visible score is the furthest row reached during the current run. Moving backward does not reduce the score.

Training reward uses the same environment state for both learned controllers:

- Positive reward for reaching a new furthest row.
- Large negative reward for a terminal event.
- Small time cost to discourage indefinite waiting.
- No hidden controller-specific reward terms.

Exact numeric coefficients belong in versioned experiment configuration rather than being embedded in UI code.

## 5. Architecture

### Browser application

The existing React, TypeScript, Three.js, and Vite template remains the foundation. Responsibilities are separated as follows:

- **Game simulation:** pure TypeScript state transitions, procedural generation, observations, rewards, and collisions.
- **Game renderer:** Three.js scene, camera, lighting, low-poly assets, animation interpolation, and effects.
- **Input adapters:** keyboard, touch, and autonomous controller scheduling.
- **Controller boundary:** shared observation/action protocol and controller lifecycle.
- **Brain view:** existing measured soma atlas, receiving synchronized activity frames keyed by real MaleCNS body IDs.
- **Telemetry:** controller identity, source category, step, observation summary, action, reward, score, and connection state.
- **Experiment recorder:** optional bounded in-memory run trace and export for evaluation or replay.

One authoritative simulation clock coordinates the game, controller decisions, activity frames, rewards, replay, and telemetry. Rendering may run at a higher rate, but it must not create a second source of time.

### Python tools

Python owns training, batch evaluation, and heavy full-connectome experiments. It communicates with the browser in two ways:

- Exported compact model files for local browser inference.
- A versioned local live protocol for controllers too large to run in the browser.

Training and the web application share fixtures for the observation schema, action enumeration, seeded episode expectations, and model metadata.

### Full MaleCNS extension

The first release implements the adapter contract but does not require a full MaleCNS runtime. A later sparse Python service can implement the same controller interface. Disconnecting the service pauses autonomous play and clears stale neural activity.

## 6. Controller Contract

### Observation

The controller receives a bounded egocentric window rather than the full future map. The versioned observation contains:

- Local cell or lane type around the fly.
- Occupancy and relative position of nearby hazards and platforms.
- Hazard or platform direction and normalized speed where visible.
- Fly position relative to navigable horizontal bounds.
- Current support state, previous action, and a small set of episode timing features.

All numeric ranges and categorical encodings are declared in the schema. Human, conventional, and connectome modes use the same underlying observation snapshot, even when the human does not see the raw vector.

### Action result

Every controller response includes:

- One of the five legal actions.
- Controller and model identity.
- Source category and normalization metadata.
- Optional activity values keyed by validated MaleCNS body ID.
- Optional diagnostic values that are not treated as neural activity.

Invalid, late, or incompatible responses do not advance the simulation. They produce a visible error and pause the run.

## 7. Learned Controllers

### Conventional baseline

A compact neural policy provides a capacity-matched non-connectome comparison. Its architecture, parameter count, training budget, observation encoding, and evaluation seeds are recorded with every released checkpoint.

### Reduced connectome controller

The reduced controller uses a documented subgraph derived from a versioned MaleCNS connectivity source. The internal recurrent topology is fixed. Trainable elements are restricted to:

- Sensory projection into selected populations.
- Explicit gain, time-scale, or normalization parameters allowed by the experiment configuration.
- Motor readout from selected descending or output populations to the five actions.

The initial implementation uses a differentiable rate-based recurrent model suitable for reinforcement learning. If spiking dynamics are added later, they become a separately named experiment rather than silently replacing the rate model.

Activity displayed in the brain scene is simulated model output. Atlas coordinates represent measured soma positions; they do not imply that the project contains full neurite morphology or recorded firing.

### Training

The default trainer uses Proximal Policy Optimization (PPO) with parallel seeded environments and a categorical five-action policy. Both learned controllers receive equivalent environment steps and held-out evaluation. Configuration, random seeds, checkpoint hashes, training curves, and software versions are recorded.

## 8. Evaluation

Released experiments report distributions, not only best runs:

- Mean, median, and maximum forward score.
- Survival steps and wall-clock-independent episode length.
- Terminal-event breakdown.
- Action distribution and waiting frequency.
- Performance on held-out world seeds.
- Training environment steps and parameter count.

Controls include:

- Capacity-matched conventional network.
- Rewired or shuffled reduced graph while preserving relevant size statistics.
- Selected population or pathway silencing.
- Untrained readout where useful as a sanity check.

All comparisons use a versioned seed suite and the same environment version. The UI and reports avoid language implying biological learning or superiority unless supported by the defined measurements.

## 9. Model Provenance and Attribution

The interface distinguishes:

- **Measured anatomy:** atlas coordinates and their dataset source.
- **Sourced connectivity:** the versioned edge dataset used to construct the reduced graph.
- **Simulated activity:** controller values computed by a model.
- **Trained parameters:** sensory interface, permitted dynamics parameters, and motor readout.
- **Manual control:** no neural output; only the encoded stimulus may be inspected.

The required linked credit to `fly-connectome-template` by Mert Cobanov remains reachable in one click in the web UI and remains in the repository documentation. MaleCNS, Flybody, and any other third-party data or assets retain their own notices and license terms. Modifications are identified as required by the template license.

## 10. State and Error Handling

The application has explicit states for loading anatomy, ready, running, paused, terminal, replaying, connecting, and error.

- Failure to load optional neural output does not prevent Human mode.
- An incompatible model file is rejected before a run begins.
- Unknown or duplicate body IDs, invalid values, schema mismatches, and non-monotonic replay times are rejected using the existing validation principles.
- Live-controller timeout or disconnect pauses the game and removes stale activity.
- Hidden browser tabs do not accumulate unbounded simulation time.
- Slow devices can reduce visual density or effects without changing simulation rules.
- A human-readable error is shown near the affected control; technical detail remains available for diagnosis.

## 11. Testing and Verification

### Unit tests

- Seeded procedural generation and chunk continuity.
- Lane solvability invariants.
- Movement, scoring, collision, platform support, and terminal conditions.
- Observation encoding and reward calculation.
- Controller schema and model metadata validation.
- Simulation-clock behavior and deterministic replays.

### Integration tests

- Complete deterministic episodes from fixed action traces.
- Browser and Python observation/action fixture compatibility.
- Controller swap and reset behavior.
- Brain activity frame synchronization and clearing on disconnect.
- Existing atlas and replay validation behavior.

### User-facing verification

- Keyboard and touch playthroughs.
- Human, conventional, and connectome mode smoke tests.
- Responsive desktop and narrow-screen layouts with both major panels visible.
- Stable frame pacing on the defined baseline device profile.
- Production build, asset hash checks, and local preview.

## 12. Delivery Sequence

Implementation proceeds in vertical slices:

1. Deterministic simulation and tests.
2. Playable low-poly renderer with human controls.
3. Always-visible brain layout and synchronized telemetry.
4. Controller/model contracts and a deterministic scripted controller.
5. Python-compatible environment fixtures and conventional training baseline.
6. Reduced MaleCNS subgraph pipeline, connectome controller, and exported inference.
7. Evaluation controls, reports, responsive polish, and full verification.

Each slice remains runnable and testable before the next is added.

## 13. Acceptance Criteria

The first release is complete when:

- A user can play an endless seeded game in a browser with keyboard or touch.
- The brain visualization remains visible and synchronized throughout play.
- Human, conventional, and reduced-connectome controllers can run through the same public interface.
- Manual mode never fabricates neural activity.
- Autonomous modes expose valid provenance and clear simulated-activity labeling.
- Repeating a seed and action trace reproduces the same episode outcome.
- Python can train/evaluate both learned controller families and export a browser-loadable reduced model.
- The evaluation suite runs held-out seeds and emits the defined metrics and control comparisons.
- Automated tests, type checking, asset checks, and the production build pass.
- Required template and dataset attribution remains present in both the UI and repository documentation.
