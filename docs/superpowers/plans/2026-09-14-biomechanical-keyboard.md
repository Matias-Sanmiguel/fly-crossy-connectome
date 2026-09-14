# Biomechanical Fly Keyboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a 59-action biomechanical fly press W/A/S/D keys with mapped legs and release a game action only after MuJoCo confirms the intended physical contact.

**Architecture:** A generated MuJoCo keyboard scene is composed with a pinned Flybody/FlyGym-compatible model. A shared closed-loop motor state machine drives calibrated leg trajectories, while an independent contact gate debounces key travel and force and emits exactly one terminal result per intention.

**Tech Stack:** Python 3.12, MuJoCo, NumPy, FlyGym/Flybody assets, Pydantic protocol v2, pytest, Docker.

**Spec:** `docs/superpowers/specs/2026-09-14-multiscale-biomechanical-keyboard-design.md`

## Global Constraints

- The motor interface is exactly 59-dimensional and shared by all five neural controllers.
- Front-left presses W, front-right presses S, middle-left presses A, and middle-right presses D.
- Hind legs may reach two Space zones for neutral/hold demonstrations; Space is not a successful directional action.
- Confirmation requires intended-key travel, force/contact threshold, and debounce duration.
- Wrong-key, ambiguous, duplicate, and timed-out presses never become the requested directional game action.
- A key must cross its release threshold before another press can latch.
- The browser cannot infer contact from animation.
- Do not add automated-agent coauthor trailers to commits.

---

## File map

- `python/fly_crossy/biomechanics/keyboard.py` — key geometry/configuration and MJCF generation.
- `python/fly_crossy/biomechanics/body.py` — pinned fly model adapter and 59-action validation.
- `python/fly_crossy/biomechanics/contact.py` — travel/force debounce and one-shot latch.
- `python/fly_crossy/biomechanics/motor.py` — shared motor state machine and joint controller.
- `python/fly_crossy/biomechanics/world.py` — MuJoCo composition, stepping, snapshots, and recovery.
- `python/fly_crossy/biomechanics/calibration.py` — deterministic key/leg calibration artifact.
- `python/fly_crossy/biomechanics/config.py` — validated physics and threshold configuration.
- `python/tests/biomechanics/*` — unit and deterministic integration tests.
- `configs/biomechanics-v1.json` — committed physics, mapping, limits, and timing.
- `data/manifests/flybody-v1.json` — pinned source/checksum metadata; no downloaded mesh payloads.

### Task 1: Pin and validate the fly model interface

**Files:**
- Create: `data/manifests/flybody-v1.json`
- Create: `python/fly_crossy/biomechanics/__init__.py`
- Create: `python/fly_crossy/biomechanics/body.py`
- Create: `python/tests/biomechanics/test_body.py`
- Modify: `python/pyproject.toml`

**Interfaces:**
- Produces: `FlyBodyModel.load(manifest_path)`, `action_size == 59`, `joint_names`, `leg_sites`, and `validate_action(values) -> np.ndarray`.
- Produces: stable leg names `front_left`, `front_right`, `middle_left`, `middle_right`, `hind_left`, `hind_right`.

- [ ] **Step 1: Add a source manifest with immutable fields**

```json
{
  "schemaVersion": 1,
  "source": "https://github.com/TuragaLab/flybody",
  "model": "flybody-compatible",
  "licenseFile": "public/data/flybody/LICENSE",
  "noticeFile": "public/data/flybody/NOTICE.md",
  "actionSize": 59,
  "requiredSites": ["front_left_tarsus","front_right_tarsus","middle_left_tarsus","middle_right_tarsus","hind_left_tarsus","hind_right_tarsus"]
}
```

During implementation preflight, record the selected upstream commit and model-file SHA-256 in this manifest before the body is accepted; the loader rejects absent or non-64-hex checksums.

Pin `flygym==2.1.0` and `mujoco==3.9.0` in `python/pyproject.toml`; regenerate the CPU/CUDA hash-locked requirement files from the runtime plan before building the simulation image.

- [ ] **Step 2: Write failing action-shape and joint-limit tests**

```python
def test_body_requires_exactly_59_actions(body):
    assert body.validate_action(np.zeros(59)).shape == (59,)
    with pytest.raises(ValueError, match="59"):
        body.validate_action(np.zeros(58))

def test_body_clamps_nothing_silently(body):
    action = np.zeros(59)
    action[0] = body.upper_limits[0] + 0.01
    with pytest.raises(ValueError, match="joint limit"):
        body.validate_action(action)
```

- [ ] **Step 3: Run the body tests and verify failure**

Run: `cd python && pytest tests/biomechanics/test_body.py -q`  
Expected: FAIL because the biomechanics package is absent.

- [ ] **Step 4: Implement the immutable adapter**

```python
@dataclass(frozen=True, slots=True)
class FlyBodyModel:
    model: mujoco.MjModel
    joint_names: tuple[str, ...]
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    leg_sites: Mapping[str, int]

    @property
    def action_size(self) -> int:
        return 59
```

Resolve joints/sites once at load time, require unique names and finite limits, verify manifest/model hashes, and copy validated action arrays so callers cannot mutate queued commands.

- [ ] **Step 5: Run tests and commit**

Run: `cd python && pytest tests/biomechanics/test_body.py -q`  
Expected: PASS.

```bash
git add data/manifests/flybody-v1.json python/pyproject.toml python/fly_crossy/biomechanics python/tests/biomechanics/test_body.py
git commit -m "feat: validate 59-action fly body model"
```

### Task 2: Build a physical W/A/S/D/Space keyboard

**Files:**
- Create: `configs/biomechanics-v1.json`
- Create: `python/fly_crossy/biomechanics/config.py`
- Create: `python/fly_crossy/biomechanics/keyboard.py`
- Create: `python/tests/biomechanics/test_keyboard.py`

**Interfaces:**
- Consumes: pinned MuJoCo model tooling from Task 1 and protocol `KeyName`/`MotorPhase` types.
- Produces: `KeyboardConfig`, `build_keyboard_mjcf(config) -> str`, and `KeyIds` containing joint, geom, and force-sensor IDs.

- [ ] **Step 1: Add explicit keyboard configuration**

```json
{
  "schemaVersion": 1,
  "physicsHz": 500,
  "motorHz": 100,
  "decisionTimeoutSeconds": 1.5,
  "debounceSeconds": 0.02,
  "minimumTravelMeters": 0.00035,
  "releaseTravelMeters": 0.00010,
  "minimumForceNewtons": 0.00010,
  "maximumForceNewtons": 0.02000,
  "keys": {
    "W": {"center": [0.0, 0.0018, 0.0]},
    "A": {"center": [-0.0018, 0.0, 0.0]},
    "S": {"center": [0.0, -0.0018, 0.0]},
    "D": {"center": [0.0018, 0.0, 0.0]},
    "SPACE_LEFT": {"center": [-0.0010, -0.0028, 0.0]},
    "SPACE_RIGHT": {"center": [0.0010, -0.0028, 0.0]}
  }
}
```

- [ ] **Step 2: Write failing geometry and key-travel tests**

```python
def test_keyboard_has_six_spring_loaded_keys(keyboard_model):
    for name in ("W", "A", "S", "D", "SPACE_LEFT", "SPACE_RIGHT"):
        assert keyboard_model.key(name).travel_range == pytest.approx((0.0, 0.0005))
        assert keyboard_model.key(name).sensor_id >= 0

def test_key_returns_below_release_threshold(world):
    world.apply_key_force("W", 0.001)
    world.step_seconds(0.1)
    world.clear_external_forces()
    world.step_seconds(0.25)
    assert world.key_travel("W") < world.config.release_travel_meters
```

- [ ] **Step 3: Run the keyboard tests and verify failure**

Run: `cd python && pytest tests/biomechanics/test_keyboard.py -q`  
Expected: FAIL with missing keyboard module.

- [ ] **Step 4: Generate constrained keys with sensors**

```python
KEY_NAMES = ("W", "A", "S", "D", "SPACE_LEFT", "SPACE_RIGHT")

def build_keyboard_mjcf(config: KeyboardConfig) -> str:
    root = mjcf.RootElement(model="fly_keyboard")
    for name in KEY_NAMES:
        body = root.worldbody.add("body", name=f"key_{name.lower()}", pos=config.keys[name].center)
        body.add("joint", name=f"key_{name.lower()}_travel", type="slide", axis=(0, 0, -1), range=(0, 0.0005), stiffness=0.08, damping=0.002)
        body.add("geom", name=f"key_{name.lower()}_cap", type="box", size=(0.0007, 0.0007, 0.00015))
        root.sensor.add("force", name=f"key_{name.lower()}_force", site=f"key_{name.lower()}_site")
    return root.to_xml_string()
```

Validate finite positive thresholds and `release < minimum travel`; prohibit duplicate centers and unknown key names.

- [ ] **Step 5: Run keyboard tests and commit**

Run: `cd python && pytest tests/biomechanics/test_keyboard.py -q`  
Expected: PASS.

```bash
git add configs/biomechanics-v1.json python/fly_crossy/biomechanics/config.py python/fly_crossy/biomechanics/keyboard.py python/tests/biomechanics/test_keyboard.py
git commit -m "feat: add physical fly keyboard scene"
```

### Task 3: Contact confirmation and one-shot latching

**Files:**
- Create: `python/fly_crossy/biomechanics/contact.py`
- Create: `python/tests/biomechanics/test_contact.py`

**Interfaces:**
- Produces: `ContactGate.begin(intention_id, intended_key)`, `sample(ContactSample)`, `cancel()`, and `ContactOutcome`.
- Consumes: per-key travel, normal force, touching tarsus, and simulation timestamp.

- [ ] **Step 1: Write failing debounce, wrong-key, and release tests**

```python
def test_press_needs_force_travel_and_debounce(gate):
    gate.begin("i-1", "W")
    assert gate.sample(sample("W", travel=0.0004, force=0.0002, time=0.000)).kind == "pending"
    assert gate.sample(sample("W", travel=0.0004, force=0.0002, time=0.019)).kind == "pending"
    assert gate.sample(sample("W", travel=0.0004, force=0.0002, time=0.021)).kind == "confirmed"

def test_wrong_key_never_confirms_requested_key(gate):
    gate.begin("i-2", "W")
    assert gate.sample(sample("S", travel=0.0004, force=0.0002, time=0.03)).kind == "wrong-key"
```

Add tests for force without travel, travel without force, force above safe maximum, two keys, duplicate samples after confirmation, cancellation, timeout, and press again before release.

- [ ] **Step 2: Run and verify failures**

Run: `cd python && pytest tests/biomechanics/test_contact.py -q`  
Expected: FAIL because `ContactGate` is absent.

- [ ] **Step 3: Implement the explicit contact gate**

```python
@dataclass(frozen=True, slots=True)
class ContactSample:
    time: float
    key: KeyName
    tarsus: LegName
    travel: float
    normal_force: float

@dataclass(frozen=True, slots=True)
class ContactOutcome:
    kind: Literal["pending", "confirmed", "wrong-key", "ambiguous", "unsafe-force", "timed-out"]
    intention_id: str
    key: KeyName | None
```

Latch once per intention ID, require the action-to-key leg mapping, reset debounce on any invalid sample, and require release before rearming that key.

- [ ] **Step 4: Run contact and complete Python tests**

Run: `cd python && pytest tests/biomechanics/test_contact.py -q && pytest -q`  
Expected: PASS.

- [ ] **Step 5: Commit the physical confirmation gate**

```bash
git add python/fly_crossy/biomechanics/contact.py python/tests/biomechanics/test_contact.py
git commit -m "feat: gate actions on debounced key contact"
```

### Task 4: Shared 59-joint motor state machine

**Files:**
- Create: `python/fly_crossy/biomechanics/motor.py`
- Create: `python/fly_crossy/biomechanics/calibration.py`
- Create: `python/tests/biomechanics/test_motor.py`
- Create: `data/calibration/keyboard-reach-v1.json`

**Interfaces:**
- Consumes: `FlyBodyModel`, `KeyboardConfig`, `ContactGate`, high-level `Action`.
- Produces: `MotorController.request(intention)`, `update(state, dt) -> MotorCommand`, `MotorPhase`, and immutable calibration artifact.

- [ ] **Step 1: Write failing mapping and lifecycle tests**

```python
@pytest.mark.parametrize((action, leg, key), [
    ("forward", "front_left", "W"),
    ("backward", "front_right", "S"),
    ("left", "middle_left", "A"),
    ("right", "middle_right", "D"),
])
def test_action_mapping(action, leg, key, motor):
    target = motor.request(intention("i-1", action))
    assert (target.leg, target.key) == (leg, key)

def test_motor_cannot_accept_two_intentions(motor):
    motor.request(intention("i-1", "forward"))
    with pytest.raises(MotorBusy):
        motor.request(intention("i-2", "left"))
```

Test every legal transition through neutral, targeting, reaching, pressing, confirmed, retracting, settling, failed; forbid skipped transitions.

- [ ] **Step 2: Run and verify failures**

Run: `cd python && pytest tests/biomechanics/test_motor.py -q`  
Expected: FAIL because the motor module is absent.

- [ ] **Step 3: Implement mapped trajectories and closed-loop corrections**

```python
class MotorPhase(StrEnum):
    NEUTRAL = "neutral"
    TARGETING = "targeting"
    REACHING = "reaching"
    PRESSING = "pressing"
    CONFIRMED = "confirmed"
    RETRACTING = "retracting"
    SETTLING = "settling"
    FAILED = "failed"

ACTION_TARGETS = {
    "forward": ("front_left", "W"),
    "backward": ("front_right", "S"),
    "left": ("middle_left", "A"),
    "right": ("middle_right", "D"),
}
```

Generate neutral, pre-key, press, and retract poses through deterministic calibration. Use bounded damped-least-squares IK for the active leg and posture targets for the remaining joints. Validate all 59 targets before returning `MotorCommand`; instability or timeout transitions to `failed`.

- [ ] **Step 4: Generate and verify calibration reproducibly**

Run: `cd python && python -m fly_crossy.biomechanics.calibration --config ../configs/biomechanics-v1.json --output ../data/calibration/keyboard-reach-v1.json && pytest tests/biomechanics/test_motor.py -q`  
Expected: the artifact records model/config hashes, 59-value poses, reachable target error, and tests PASS.

- [ ] **Step 5: Commit the shared motor controller**

```bash
git add python/fly_crossy/biomechanics/motor.py python/fly_crossy/biomechanics/calibration.py python/tests/biomechanics/test_motor.py data/calibration/keyboard-reach-v1.json
git commit -m "feat: drive keyboard with shared fly motor controller"
```

### Task 5: MuJoCo world and contact-gated action results

**Files:**
- Create: `python/fly_crossy/biomechanics/world.py`
- Create: `python/tests/biomechanics/test_world.py`
- Modify: `python/fly_crossy/session.py`
- Modify: `python/fly_crossy/server.py`
- Modify: `python/tests/test_session.py`
- Modify: `python/tests/test_server.py`

**Interfaces:**
- Consumes: protocol/session from runtime plan and body/keyboard/motor/contact tasks.
- Produces: `BiomechanicalWorld.reset()`, `start_intention()`, `step_until_event()`, `snapshot()`, and v2 `contact`, `snapshot`, `action_result` messages.

- [ ] **Step 1: Write the failing causal integration test**

```python
def test_game_direction_is_emitted_only_after_confirmed_contact(world):
    world.start_intention(intention("i-1", "forward"))
    events = list(world.step_until_event(limit_seconds=1.5))
    terminal = [event for event in events if event.type == "action_result"]
    assert len(terminal) == 1
    assert terminal[0].outcome == "confirmed"
    assert terminal[0].action == "forward"
    assert any(event.type == "contact" and event.confirmed for event in events)

def test_timeout_emits_wait_not_requested_direction(failing_world):
    result = failing_world.run(intention("i-2", "left"))
    assert (result.outcome, result.action) == ("failed", "wait")
```

- [ ] **Step 2: Run the integration tests and verify failure**

Run: `cd python && pytest tests/biomechanics/test_world.py tests/test_server.py -q`  
Expected: FAIL because the server is not connected to biomechanics.

- [ ] **Step 3: Implement fixed-rate stepping and snapshots**

```python
@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    simulation_time: float
    body_position: tuple[float, float, float]
    body_quaternion: tuple[float, float, float, float]
    joint_positions: tuple[float, ...]
    key_travel: Mapping[KeyName, float]
    motor_phase: MotorPhase
```

Step physics at 500 Hz and motor targets at 100 Hz from config. Emit snapshots at no more than 30 Hz. End one intention on confirmation, deliberate wait completion, or 1.5-second timeout. Recovery must finish before the session accepts another observation.

- [ ] **Step 4: Add CPU deterministic mappings for all four directions**

Run: `cd python && pytest tests/biomechanics/test_world.py -q --maxfail=1`  
Expected: W/A/S/D deterministic calibration cases PASS, including exactly one result each.

- [ ] **Step 5: Run all native tests and Docker smoke**

Run: `cd python && pytest -q && cd .. && npm test && docker compose up --build --wait && bash scripts/docker-smoke.sh && docker compose down`  
Expected: all commands exit 0.

- [ ] **Step 6: Commit the biomechanical runtime**

```bash
git add python/fly_crossy/biomechanics/world.py python/tests/biomechanics/test_world.py python/fly_crossy/session.py python/fly_crossy/server.py python/tests/test_session.py python/tests/test_server.py
git commit -m "feat: confirm game actions through fly key presses"
```

## Plan completion checkpoint

Run: `cd python && pytest tests/biomechanics -q && pytest -q`  
Expected: every mapped leg presses its intended key, failure yields wait, all joint/contact bounds hold, and no unconfirmed directional action is emitted.
