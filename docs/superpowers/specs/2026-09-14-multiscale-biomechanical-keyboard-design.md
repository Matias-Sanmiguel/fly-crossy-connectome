# Fly Crossy Connectome — Multiscale Brain and Biomechanical Keyboard Design

**Date:** 2026-09-14  
**Status:** Approved design, implementation pending written-spec review  
**Supersedes:** The controller, runtime, and delivery scope in `2026-09-14-fly-crossy-connectome-design.md`  
**Repository:** `fly-crossy-connectome`

## 1. Decision summary

The project will become a browser-based Crossy Road/Frogger experiment in which an autonomous fruit fly chooses actions with one of five independently trained MaleCNS-derived neural controllers. A shared biomechanical motor controller drives a 59-dimensional Flybody/FlyGym-compatible body in MuJoCo. The game accepts an action only after a simulated leg physically depresses the corresponding keyboard key and a contact/force sensor confirms the press.

The browser will show the game, the active MaleCNS neural population, the fly and physical keyboard, and decision/contact telemetry at the same time. Python owns neural inference, biomechanics, training, and authoritative orchestration; the browser owns presentation and the deterministic game environment. Docker Compose provides a CPU-default path and an optional NVIDIA GPU profile.

The five selectable neural population sizes are:

- 80 neurons
- 1,000 neurons
- 5,000 neurons
- 20,000 neurons
- 124,289 visible MaleCNS somata

Each population has its own graph artifact, configuration, checkpoint, training record, and evaluation results. The five populations share an observation schema, action vocabulary, motor controller, keyboard, game rules, and held-out evaluation seeds.

## 2. Goals and non-goals

### Goals

- Make the autonomous fly visibly play rather than merely emit an abstract action.
- Preserve causal authority: neural intention, physical actuation, confirmed key contact, then game action.
- Show simulated activity in real time for every selected population size without presenting it as biological recording.
- Provide five reproducible, independently trained sparse neural controllers derived from documented MaleCNS data.
- Support local execution through Docker on ordinary CPU machines, with an accelerated NVIDIA GPU mode when available.
- Keep the existing deterministic game, 80-neuron controller, atlas rendering, and evaluation assets useful during migration.
- Produce a clean, documented public GitHub repository that can be cloned, started, tested, and inspected by another person.

### Non-goals

- Claiming that simulated rate values are recorded spikes, biological thoughts, consciousness, or a faithful whole-brain emulation.
- Inventing synapses, neuron identities, anatomy, activity, training results, or performance numbers to fill missing source data.
- Training all five models in the browser.
- Requiring real-time speed for the 124,289-neuron mode on CPU.
- Controlling the user's operating-system keyboard. The keyboard is a MuJoCo object inside the simulation.
- Allowing a requested action to bypass failed biomechanics or an unconfirmed contact.

## 3. Source foundations and provenance

The implementation will use these upstream foundations:

- MaleCNS downloads and metadata: <https://male-cns.janelia.org/download/>
- FlyGym: <https://github.com/NeLy-EPFL/flygym>
- Flybody: <https://github.com/TuragaLab/flybody>
- Awesome Fly and the original template attribution: <https://github.com/cobanov/awesome-fly>

Before dependency integration, implementation preflight will select and record exact compatible tags or commit hashes. Container images, Python packages, JavaScript packages, model artifacts, and external datasets will be pinned in lockfiles or manifests. Downloaded data must include its source URL, upstream version or retrieval date, license/notice, expected byte size when available, and SHA-256 checksum.

The provenance model distinguishes five categories throughout the UI and repository:

1. **Measured anatomy:** sourced soma coordinates and identifiers.
2. **Sourced connectivity:** edges present in a named MaleCNS release.
3. **Derived topology:** documented filtering, direction, weighting, and population selection applied by this project.
4. **Simulated activity:** values produced by the controller, never represented as measured neural activity.
5. **Trained parameters:** sensory projections, allowed dynamics parameters, readouts, and checkpoints learned by this project.

Third-party code, datasets, and assets retain their required notices. No implementation commit or published commit may add Codex or another automated agent as coauthor.

## 4. System architecture

Docker Compose defines three services and one persistent data volume:

### `web`

- React, TypeScript, Three.js, and the existing deterministic game.
- Production static serving and same-origin reverse proxy for the simulation WebSocket and health endpoints.
- Renders the game, brain, biomechanical scene, keyboard, controls, and telemetry.
- Does not decide whether a physical key press succeeded.

### `simulation`

- Python, PyTorch, MuJoCo, and a pinned FlyGym/Flybody-compatible body definition.
- Loads one neural controller, the shared motor controller, the fly body, and the keyboard scene.
- Owns neural steps, joint commands, physics, contacts, key travel, and contact confirmation.
- Streams versioned snapshots and events to the browser.
- Runs on CPU by default and selects GPU only when the requested backend passes startup capability checks.

### `trainer`

- Uses the same Python package and schemas as `simulation`.
- Is disabled during normal play and enabled through an explicit Compose profile or command.
- Builds graph artifacts, trains controllers, evaluates checkpoints, and writes versioned reports.
- Never mutates a released checkpoint in place.

### `fly-data` volume

- Caches verified upstream data and generated immutable graph artifacts.
- Separates downloaded sources, derived graphs, checkpoints, and reports.
- Uses a manifest to prevent a partial or checksum-mismatched download from being treated as valid.

The existing browser-only controller remains available as a documented demo/fallback mode during development, but the five named population modes use the Python simulation path.

## 5. Sources of truth and clocks

The browser remains authoritative for deterministic game state. The Python service is authoritative for neural activity, biomechanics, contacts, and confirmation of a simulated keyboard press.

There are four coordinated time scales:

- **Physics tick:** fixed high-frequency MuJoCo step selected from validated model requirements.
- **Motor-control tick:** lower-frequency joint target update.
- **Neural decision tick:** one high-level action request at the game decision cadence.
- **Render stream:** capped near 30 frames per second, allowed to drop frames without changing simulation outcomes.

All messages carry a protocol version, session ID, episode ID, monotonically increasing sequence number, and simulation time. The browser never infers authoritative contact from animation. Rendering interpolates snapshots but cannot create a key event.

On reset or population change, the browser requests a coordinated transition. The simulation stops accepting intentions, clears queued motor commands, resets neural state, body, keyboard, and contact latches, acknowledges the new episode, and only then lets the browser reset/start the game. Old-session messages are discarded.

## 6. Neural populations and graph construction

### Nested population rule

The five graphs are nested and reproducible. The 80-neuron population is the current verified sensorimotor core. Larger sets add neurons without removing earlier members. Selection expands deterministically through relevant synaptic neighbors and documented functional/anatomical groups, with stable tie-breaking by source identifier.

The intended expansion is:

- **80:** current verified core and compatibility checkpoint.
- **1,000:** the core plus highest-priority connected sensory, interneuron, descending, and motor-relevant neighbors.
- **5,000:** broader visual, central, and descending pathways.
- **20,000:** extensive sensorimotor context with additional recurrent pathways.
- **124,289:** every soma displayed by the project's pinned MaleCNS atlas.

Exact membership is generated from versioned selection configuration, not hand-edited lists. Every graph manifest records requested size, actual soma count, connected count, isolated count, edge count, source release, filtering rules, and hash of the ordered neuron IDs and sparse edges.

If the connectivity source does not contain edges for every one of the 124,289 atlas somata, those somata remain visible and are represented as explicitly isolated nodes. Their activity follows the documented isolated-node dynamics; no edge is fabricated. Any source mismatch that prevents an exact requested soma count fails graph generation instead of silently changing the label.

### Sparse controller model

Each controller uses the same versioned recurrent-rate cell family but a distinct sparse graph and checkpoint. The source graph defines allowed internal connectivity. Sparse tensors or equivalent indexed operations are used; a dense 124,289 by 124,289 matrix is forbidden.

The observation is injected only through a documented set of selected sensory/interface neurons. Action logits are read only from documented descending/output populations. Trainable parameters are limited to explicitly declared categories, including sensory projections, allowed per-node gains/time constants, allowed edge weights or edge-class scales, and motor readout. Every experiment configuration declares which categories are frozen or trainable.

An activity frame contains neuron ID and current simulated value. The wire format is sparse and transmits changed/selected values rather than the full array every render frame. Periodic keyframes allow recovery after dropped deltas. Activity normalization is model metadata, not an undocumented UI transformation.

### Five independent controllers

Each size is trained and versioned independently. A larger checkpoint is not presented as a smaller checkpoint with extra decorative nodes. The artifact set for each mode includes:

- graph manifest and checksum;
- training configuration and seeds;
- checkpoint and checksum;
- observation/action/protocol versions;
- training summary and curves;
- held-out evaluation report;
- software and hardware metadata.

Training begins with behavior cloning from the same deterministic planner to establish useful action behavior. Each size then receives its own bounded PPO fine-tuning run in the deterministic game environment. Training budgets and achieved results may differ, but comparisons report those differences rather than implying equal convergence.

Evaluation uses a common held-out seed suite and reports score distribution, survival steps, terminal causes, action distribution, physical press success rate, end-to-end action latency, environment steps, parameter count, and wall time. Reports separate decision quality from motor failures.

## 7. Observation and action contract

All controllers receive the same bounded egocentric game observation already defined by the project. Its schema and normalization are versioned and shared between TypeScript and Python fixtures.

The legal neural intentions are:

- `forward`
- `backward`
- `left`
- `right`
- `wait`

The `wait` intention is a deliberate no-press decision and completes after its decision interval. A directional intention becomes a game action only after the corresponding keyboard contact is confirmed. The motor layer may use the Space key as its neutral/hold demonstration target, but Space is never treated as a successful directional action. If a release build exposes Space as a gameplay action, its meaning must be added through a protocol version change and shared fixture update.

## 8. Shared biomechanical motor controller

All five neural controllers feed one shared motor controller so controller comparisons are not confounded by five separately learned bodies. The motor controller consumes high-level intention plus body/keyboard proprioceptive state and emits the 59-dimensional joint action expected by the pinned fly model.

The physical keyboard maps legs to controls:

- front-left leg → W / forward;
- front-right leg → S / backward;
- middle-left leg → A / left;
- middle-right leg → D / right;
- both hind legs → two accessible zones of Space for neutral/hold demonstrations.

Each key is a MuJoCo body with constrained travel, restoring spring/damping, collision geometry, and contact/force sensing. Key positions are adapted to the fly's reachable workspace while remaining visually recognizable as a keyboard cluster.

The motor state machine is:

1. `neutral`: stable posture, no directional key latched.
2. `targeting`: select the mapped leg and collision-safe trajectory.
3. `reaching`: move the tarsus above the target key.
4. `pressing`: descend until travel and force thresholds are satisfied.
5. `confirmed`: latch exactly one press event for the current intention ID.
6. `retracting`: unload and return the leg.
7. `settling`: regain neutral posture before accepting another intention.
8. `failed`: timeout, wrong-key contact, fall, instability, or invalid state.

A press is confirmed only when the intended key exceeds both a minimum travel and force/contact threshold for the configured debounce duration. One intention ID can create at most one game action. Wrong-key or simultaneous ambiguous contacts do not count. A release threshold must be crossed before that key can be latched again.

If the motor controller misses or times out, the event is logged as a motor failure, the game receives `wait`, and the controller returns to neutral or pauses if recovery fails. The system never synthesizes the originally requested direction.

The initial motor implementation may combine a pose library, inverse kinematics, and closed-loop contact feedback. A learned locomotor/manipulation policy may replace it only behind the same tested state machine and confirmation contract.

## 9. End-to-end action flow

For one decision:

1. Browser freezes the next authoritative game decision and sends observation `N`.
2. Neural controller produces intention `N` and an activity update.
3. Motor controller starts the mapped leg trajectory tagged with `N`.
4. MuJoCo advances until confirmation, deliberate wait completion, or timeout.
5. Simulation sends a terminal result for `N`: `confirmed`, `waited`, or `failed`.
6. Browser applies the direction only for a matching `confirmed` result; `waited` and `failed` advance as `wait` under the configured game cadence.
7. Browser returns reward and next observation, both tagged with the resulting game step.

Only one intention may be in flight. Duplicate terminal results are idempotently ignored. Out-of-order or unknown IDs pause the session with a visible protocol error.

## 10. Protocol and transport

The browser and simulation use a versioned WebSocket protocol. JSON control messages are sufficient initially; high-volume transforms and neural deltas may move to a documented binary envelope without changing semantic message types.

Required client messages:

- `hello`: supported protocol versions and UI build.
- `configure`: population, backend preference, seed, and speed.
- `reset`: coordinated new episode request.
- `observation`: game step and encoded observation.
- `pause` / `resume`: explicit lifecycle control.
- `request_keyframe`: resynchronization request.

Required server messages:

- `ready`: accepted versions, resolved backend, artifact hashes, and capability limits.
- `reset_complete`: new episode/session state.
- `intention`: selected action and controller metadata.
- `snapshot`: body/joint/key transforms at a simulation timestamp.
- `neural_keyframe` / `neural_delta`: sparse activity payloads.
- `contact`: requested key, touched key, travel, force, debounce, and confirmation.
- `action_result`: confirmed/waited/failed terminal outcome for one intention.
- `metrics`: effective tick rates, latency, dropped render frames, and backend utilization when measurable.
- `paused` / `error`: stable error code plus safe human-readable message.

The server bounds message size, neuron-update count, queue length, and accepted numeric ranges. Backpressure drops superseded render snapshots and neural deltas but never drops action results, reset acknowledgements, errors, or the latest required keyframe. A reconnect creates a new session and cannot resume an ambiguous in-flight press.

## 11. Browser experience

### Desktop

The approved “hexapod station” layout contains:

- a top segmented selector for 80, 1,000, 5,000, 20,000, and 124,289 neurons;
- backend state (`GPU`, `CPU`, or `CPU lento`) and effective simulation/render rates;
- a Crossy Road panel;
- a MaleCNS activity panel that remains visible;
- a biomechanical fly and keyboard panel;
- telemetry for neural target, requested key, actual contact, force/travel, motor state, result, and latency.

The requested key and physically confirmed key are always visually distinct. Activity is labeled `actividad simulada`; disconnected or stale data clears rather than freezing as if current.

### Mobile and accessibility

Narrow screens stack game, keyboard/body, brain, then telemetry. The controller selector remains reachable without covering the game. Keyboard operation, visible focus, reduced-motion mode, high-contrast status cues, and text equivalents for color-coded states are required. Canvas panels expose concise live status outside the canvas so critical information is not visual-only.

Changing population warns when the selected CPU mode is expected to be slow, then performs the coordinated reset. No speed label may imply real time unless measured effective simulation speed is at least real time over the current reporting window.

## 12. CPU, GPU, and performance behavior

The default command starts CPU mode. An explicit `gpu` Compose profile enables NVIDIA container support. Startup capability checks confirm PyTorch device availability and any selected MuJoCo acceleration path; a failed GPU check falls back to CPU with a visible reason unless strict GPU mode was requested.

Correctness and event ordering are identical across backends within declared floating-point tolerances. Performance adaptations may change render density, activity-delta frequency, batching, or effective simulation speed, but not graph membership, accepted contacts, game rules, or the selected checkpoint.

The 124,289-neuron mode is supported on CPU as a correctness fallback, not promised at real-time speed. It may run below wall-clock real time; the UI reports effective simulation speed and latency honestly. Smaller modes target interactive local play. Concrete tick-rate and memory budgets are established by implementation benchmarks on a documented baseline CPU and GPU, then committed as release metadata rather than guessed in advance.

## 13. Failure handling and safety

- Model, graph, atlas, and checkpoint hashes are validated before a mode becomes ready.
- Missing data, incompatible schema, NaN/Inf activity, invalid body IDs, or shape mismatch reject startup or pause the active session.
- Simulation disconnect pauses autonomous play and clears activity/contact state.
- Physics instability, a fallen fly, repeated wrong-key contact, or motor recovery failure pauses with a diagnostic; it cannot advance the requested direction.
- Browser tab throttling cannot accumulate an unbounded physics or decision backlog.
- The service accepts only enumerated artifact IDs rooted in its manifest, never arbitrary client-supplied filesystem paths.
- Production traffic is same-origin through the web proxy; development CORS origins are explicit and narrow.
- Containers run as non-root where supported, expose only required ports, and do not bake credentials or downloaded datasets into images.
- Logs exclude credentials and bound high-volume activity/contact data.

## 14. Testing strategy

### Shared contract tests

- TypeScript and Python schema fixtures for every message and protocol version.
- Observation encoding and action enumeration parity.
- Session, episode, sequence, reset, idempotency, reconnect, and timeout behavior.
- Bounded payload, invalid-number, stale-message, and unsupported-version rejection.

### Graph and neural tests

- Exact requested counts and strict nesting across all five populations.
- Stable ordered IDs and graph hashes from the pinned inputs.
- No edge whose endpoints are absent and no edge absent from the declared source transformation.
- Explicit isolated-node count for the full atlas mode.
- Sparse memory representation; tests prevent accidental dense allocation.
- Checkpoint/graph/schema compatibility and deterministic inference fixtures within tolerance.
- Sparse neural delta/keyframe reconstruction.

### Biomechanics tests

- 59-dimensional action shape and joint-limit enforcement.
- Each mapped leg reaches only its intended key in deterministic calibration scenes.
- Travel, force, debounce, release, wrong-key, double-contact, timeout, and duplicate-latch cases.
- No directional game action without confirmed intended-key contact.
- Neutral return and recoverable failure behavior.
- CPU/GPU contact outcome parity within declared tolerance.

### Integration and end-to-end tests

- Neural intention → motor trajectory → physical contact → exactly one matching game action.
- Failed press → logged motor failure → game wait, never requested direction.
- Coordinated reset and population switch clears all five state domains.
- Deterministic seeded episodes and browser/Python state agreement.
- WebSocket backpressure preserves critical events.
- Docker CPU clean-clone startup and optional GPU smoke test.
- Desktop and mobile layout checks with all three visualization panels present.

### Evaluation and release verification

- The five released checkpoints run on the same held-out seed suite.
- Reports distinguish neural choices, motor success, and final game outcomes.
- Tests, type checks, production build, Python lint/type checks, data/hash audit, container health checks, and documented smoke run pass from a clean clone.
- Results shown in README or UI must be generated artifacts tied to committed configuration and hashes.

## 15. Delivery phases

1. **Compatibility preflight:** pin upstream versions/licenses, verify the current clean baseline, and define shared protocol fixtures.
2. **Runtime skeleton:** add the Python package, WebSocket lifecycle, Compose services, CPU health path, and optional GPU selection.
3. **Physical keyboard:** integrate the body and 59-joint interface, build calibrated MuJoCo keys, and prove contact-gated actions.
4. **Browser station:** render body/keyboard state, add five-mode controls, backend status, activity streaming, and responsive layout.
5. **Graph pipeline:** generate and validate the five nested MaleCNS artifacts with complete provenance.
6. **Controller pipeline:** train/export five independent behavior-cloned controllers, run bounded PPO fine-tunes, and retain reproducible experiment records.
7. **Integration:** connect each controller to the shared motor layer and run common held-out evaluation.
8. **Hardening:** performance work, CPU/GPU parity, error paths, clean-clone Docker testing, documentation, and visual QA.
9. **Publication:** sanitize the repository, create the public GitHub repository, push the verified history without coauthor trailers, and verify the public README/startup instructions.

Each phase ends with a runnable vertical slice and a review checkpoint. Training work may use fast smoke configurations during development; release checkpoints and reported metrics must be clearly distinguished from smoke artifacts.

## 16. Public repository and README

The default public repository name is `fly-crossy-connectome`. Publication occurs only after release verification. Existing unrelated upstream remotes are not overwritten; the final remote target is resolved and inspected before push.

The published repository excludes credentials, local caches, downloaded datasets, temporary visual-companion files, large untracked training outputs, and machine-specific paths. Git history and final commit messages contain no `Co-authored-by` trailer for Codex or other automated agents.

The README will include:

- short project explanation and an honest scientific-status statement;
- screenshot or short demo media produced from the verified build;
- architecture and causal action-flow diagram;
- quick start for CPU and NVIDIA GPU;
- hardware expectations and the honest 124,289-neuron CPU limitation;
- controller-mode table with graph/checkpoint provenance;
- development, test, training, evaluation, and data-download commands;
- protocol/version and artifact-manifest explanation;
- reproducibility notes and generated result links;
- licenses, upstream credits, dataset attribution, and citation guidance;
- limitations and a statement that activity is simulated.

## 17. Acceptance criteria

The expanded project is complete only when all of the following are true:

- The UI exposes all five named neural modes and loads a distinct verified graph/checkpoint for each.
- All five controllers can autonomously produce useful actions in the game under their released configuration; their measured results are reported without guaranteed score claims.
- The selected neural population is visible and updates from actual controller state with explicit simulated-activity labeling.
- The fly body uses the shared 59-dimensional motor interface to operate the MuJoCo keyboard.
- No directional game action occurs without a matching, debounced physical contact confirmation.
- Motor misses produce wait/failure behavior and never a fabricated successful press.
- Population changes and resets atomically reset brain, motor, body, keyboard, and game state.
- CPU mode runs every population correctly; the largest mode may be slower than real time and reports that fact.
- The optional GPU profile passes its smoke/parity checks on compatible NVIDIA hardware.
- Protocol messages and artifacts are versioned, bounded, and validated.
- Automated, clean-clone, Docker, responsive-layout, and manual play verification pass.
- Public documentation reproduces the verified startup path and does not claim biological fidelity or results that were not measured.
- The final public GitHub repository is clean, public, credited correctly, and contains no automated-agent coauthor attribution.

## 18. Scientific limitations

The neural controllers are engineering models constrained by selected source connectivity and project-defined dynamics. Neural values are simulated, sensory inputs are artificial encodings of a game, and motor outputs drive a simulated body toward an artificial keyboard. Nested graph size does not imply increasing biological fidelity or performance. Atlas-wide display does not imply atlas-wide connectivity. Behavior cloning and PPO optimize game behavior, not biological plausibility. These limitations remain prominent in the UI, README, and any shared result.
