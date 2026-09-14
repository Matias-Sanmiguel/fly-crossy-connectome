# Multiscale Runtime and Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a versioned Python simulation service, browser client, and Docker CPU/GPU runtime that can safely coordinate one neural intention at a time.

**Architecture:** A FastAPI WebSocket service owns controller and physics session state while the existing TypeScript game remains authoritative for game state. Shared JSON fixtures keep Python and TypeScript message validation aligned; Docker Compose exposes the service through the web origin and selects CPU by default or NVIDIA GPU explicitly.

**Tech Stack:** Python 3.12, FastAPI, Pydantic 2, PyTorch, React 19, TypeScript 5.9, Vite 8, Docker Compose, pytest, Node test runner.

**Spec:** `docs/superpowers/specs/2026-09-14-multiscale-biomechanical-keyboard-design.md`

## Global Constraints

- The browser is authoritative for deterministic game state; Python is authoritative for neural, biomechanical, contact, and confirmation state.
- Every message carries protocol version, session ID, episode ID, sequence number, and simulation time.
- Only one intention may be in flight, and duplicate terminal results are idempotently ignored.
- Backpressure may drop render snapshots and neural deltas but never action results, reset acknowledgements, errors, or the latest required keyframe.
- CPU is the default backend; an explicit `gpu` profile uses NVIDIA and falls back visibly unless strict GPU was requested.
- Production traffic is same-origin; payloads, numeric ranges, queues, and artifact IDs are bounded.
- Containers run as non-root and contain no credentials or downloaded datasets.
- Do not add automated-agent coauthor trailers to commits.

---

## File map

- `protocol/v2/*.json` — cross-language valid and invalid protocol fixtures.
- `python/fly_crossy/protocol.py` — Pydantic wire types and bounds.
- `python/fly_crossy/session.py` — single-flight session/episode state machine.
- `python/fly_crossy/server.py` — health, capabilities, and WebSocket endpoints.
- `python/fly_crossy/backend.py` — CPU/GPU resolution and public status.
- `python/tests/test_protocol.py` — Python fixture and bound validation.
- `python/tests/test_session.py` — reset, order, idempotency, and failure tests.
- `python/tests/test_server.py` — endpoint/WebSocket integration tests.
- `src/simulation/protocol.ts` — TypeScript wire types and runtime parser.
- `src/simulation/client.ts` — reconnect-safe browser transport.
- `tests/simulation-protocol.test.mjs` — TypeScript fixture parity.
- `tests/simulation-client.test.mjs` — queue/backpressure tests.
- `Dockerfile.web`, `Dockerfile.simulation`, `compose.yaml`, `.dockerignore` — reproducible local services.
- `python/requirements-linux-x86_64-{cpu,cu130}.txt` — hash-locked container dependencies.

### Task 1: Cross-language protocol v2

**Files:**
- Create: `protocol/v2/valid-session.json`
- Create: `protocol/v2/invalid-messages.json`
- Create: `python/fly_crossy/protocol.py`
- Create: `python/tests/test_protocol.py`
- Create: `src/simulation/protocol.ts`
- Test: `tests/simulation-protocol.test.mjs`

**Interfaces:**
- Produces: Python `ClientMessage`, `ServerMessage`, `parse_client_message()` and TypeScript `parseServerMessage(value: unknown): ServerMessage`.
- Produces: `PopulationSize = 80 | 1000 | 5000 | 20000 | 124289` and `BackendPreference = 'auto' | 'cpu' | 'gpu' | 'gpu-strict'` in both languages.
- Produces in both languages: `Action = 'forward' | 'backward' | 'left' | 'right' | 'wait'`, `KeyName = 'W' | 'A' | 'S' | 'D' | 'SPACE_LEFT' | 'SPACE_RIGHT'`, and the eight `MotorPhase` values from the spec.

- [ ] **Step 1: Add the valid session fixture**

```json
{
  "client": {"type":"hello","version":2,"sessionId":"s-01234567","episodeId":"e-01234567","sequence":0,"simulationTime":0,"supportedVersions":[2],"uiBuild":"test"},
  "server": {"type":"ready","version":2,"sessionId":"s-01234567","episodeId":"e-01234567","sequence":0,"simulationTime":0,"backend":"cpu","population":80,"graphHash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","checkpointHash":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}
}
```

- [ ] **Step 2: Write failing Python and TypeScript fixture tests**

```python
def test_valid_session_fixture_round_trips():
    fixture = json.loads(FIXTURE.read_text())
    assert parse_client_message(fixture["client"]).session_id == "s-01234567"
    assert ServerMessageAdapter.validate_python(fixture["server"]).population == 80
```

```js
test('protocol v2 fixture parses', () => {
  assert.equal(parseServerMessage(fixture.server).type, 'ready');
  assert.throws(() => parseServerMessage({...fixture.server, population: 81}), /population/i);
});
```

- [ ] **Step 3: Run both focused suites and verify missing-module failures**

Run: `node --experimental-strip-types --test --test-name-pattern="protocol v2" tests/simulation-protocol.test.mjs; cd python && pytest tests/test_protocol.py -q`  
Expected: FAIL because the v2 parsers do not exist.

- [ ] **Step 4: Implement bounded discriminated messages**

```python
PopulationSize = Literal[80, 1000, 5000, 20000, 124289]
BackendPreference = Literal["auto", "cpu", "gpu", "gpu-strict"]
KeyName = Literal["W", "A", "S", "D", "SPACE_LEFT", "SPACE_RIGHT"]

class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[2]
    session_id: Annotated[str, StringConstraints(pattern=r"^[se]-[a-z0-9]{8,64}$")]
    episode_id: Annotated[str, StringConstraints(pattern=r"^[se]-[a-z0-9]{8,64}$")]
    sequence: Annotated[int, Field(ge=0, le=2**53 - 1)]
    simulation_time: Annotated[float, Field(ge=0, le=86_400)]
```

Define the client and server variants listed in spec section 10. Bound JSON frames to 1 MiB, neural updates to 20,000 entries per delta, strings to 256 characters, forces/travel to finite declared ranges, and use `extra='forbid'`/explicit key checks in both languages.

- [ ] **Step 5: Run fixture, package, and browser tests**

Run: `cd python && pytest tests/test_protocol.py -q && cd .. && npm test`  
Expected: PASS.

- [ ] **Step 6: Commit the shared protocol**

```bash
git add protocol/v2 python/fly_crossy/protocol.py python/tests/test_protocol.py src/simulation/protocol.ts tests/simulation-protocol.test.mjs
git commit -m "feat: add bounded simulation protocol v2"
```

### Task 2: Single-flight simulation session state

**Files:**
- Create: `python/fly_crossy/session.py`
- Create: `python/tests/test_session.py`

**Interfaces:**
- Consumes: protocol v2 message models from Task 1.
- Produces: `SimulationSession.configure()`, `reset()`, `accept_observation()`, `finish_action()`, `pause()`, and `resume()`.
- Produces: `SessionFault(code: str, public_message: str)`.

- [ ] **Step 1: Write the failing single-flight and reset tests**

```python
def test_session_accepts_only_one_intention():
    session = configured_session()
    first = session.accept_observation(observation_message(sequence=2, step=0))
    assert first.intention_id == "i-00000001"
    with pytest.raises(SessionFault, match="in flight"):
        session.accept_observation(observation_message(sequence=3, step=0))

def test_reset_invalidates_old_action_result():
    session = configured_session()
    intention = session.accept_observation(observation_message(sequence=2, step=0))
    session.reset(reset_message(sequence=3, episode_id="e-new00001"))
    assert session.finish_action(intention.id, "confirmed") is None
```

- [ ] **Step 2: Verify the focused tests fail**

Run: `cd python && pytest tests/test_session.py -q`  
Expected: FAIL with `ModuleNotFoundError: fly_crossy.session`.

- [ ] **Step 3: Implement explicit lifecycle and monotonic checks**

```python
class SessionPhase(StrEnum):
    CONNECTING = "connecting"
    READY = "ready"
    ACTING = "acting"
    PAUSED = "paused"
    ERROR = "error"

@dataclass(slots=True)
class PendingIntention:
    id: str
    episode_id: str
    game_step: int
    requested_action: Action
```

Reject non-monotonic sequences, mismatched session/episode IDs, duplicate observations, and resume from error. Treat a duplicate completed intention ID as an idempotent no-op and retain the last 256 completion IDs.

- [ ] **Step 4: Run state tests and the complete Python suite**

Run: `cd python && pytest -q`  
Expected: PASS.

- [ ] **Step 5: Commit the session authority**

```bash
git add python/fly_crossy/session.py python/tests/test_session.py
git commit -m "feat: enforce single-flight simulation sessions"
```

### Task 3: Backend selection and FastAPI server

**Files:**
- Modify: `python/pyproject.toml`
- Create: `python/fly_crossy/backend.py`
- Create: `python/fly_crossy/server.py`
- Create: `python/tests/test_backend.py`
- Create: `python/tests/test_server.py`

**Interfaces:**
- Consumes: `SimulationSession` and protocol adapters.
- Produces: `resolve_backend(preference, torch_module) -> BackendStatus`.
- Produces: FastAPI `app`, `GET /healthz`, `GET /api/capabilities`, and `WS /api/simulation`.

- [ ] **Step 1: Add server test dependencies and failing endpoint tests**

```toml
dependencies = [
  "fastapi==0.116.1",
  "numpy>=2.0",
  "pydantic==2.11.7",
  "torch>=2.12",
  "uvicorn[standard]==0.35.0"
]
test = ["httpx==0.28.1", "pytest>=8.0"]
```

```python
def test_health_is_small_and_ready(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "protocol": 2}

def test_gpu_auto_falls_back_to_cpu(fake_torch_without_cuda):
    status = resolve_backend("auto", fake_torch_without_cuda)
    assert (status.resolved, status.fallback_reason) == ("cpu", "cuda-unavailable")
```

- [ ] **Step 2: Run the focused server tests**

Run: `cd python && pytest tests/test_backend.py tests/test_server.py -q`  
Expected: FAIL because backend/server modules are absent.

- [ ] **Step 3: Implement deterministic backend resolution**

```python
@dataclass(frozen=True, slots=True)
class BackendStatus:
    requested: BackendPreference
    resolved: Literal["cpu", "gpu"]
    device: str
    fallback_reason: str | None

def resolve_backend(preference: BackendPreference, torch_module=torch) -> BackendStatus:
    available = bool(torch_module.cuda.is_available())
    if preference == "gpu-strict" and not available:
        raise BackendUnavailable("CUDA was requested strictly but is unavailable")
    use_gpu = available and preference in {"auto", "gpu", "gpu-strict"}
    return BackendStatus(preference, "gpu" if use_gpu else "cpu", "cuda:0" if use_gpu else "cpu", None if use_gpu or preference == "cpu" else "cuda-unavailable")
```

- [ ] **Step 4: Implement endpoints and bounded WebSocket reads**

Accept one `hello` before configuration. Close with stable application codes for oversize, incompatible version, malformed message, or sequence violation. Send public `error` messages without traceback details and log exceptions server-side with session IDs only.

```python
@app.get("/healthz")
async def healthz() -> dict[str, object]:
    return {"status": "ok", "protocol": 2}

@app.websocket("/api/simulation")
async def simulation_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    await serve_session(websocket, max_bytes=1_048_576)
```

- [ ] **Step 5: Run complete Python tests**

Run: `cd python && pytest -q`  
Expected: PASS.

- [ ] **Step 6: Commit the runnable service**

```bash
git add python/pyproject.toml python/fly_crossy/backend.py python/fly_crossy/server.py python/tests/test_backend.py python/tests/test_server.py
git commit -m "feat: serve simulation sessions over websocket"
```

### Task 4: Browser simulation client and critical-event queue

**Files:**
- Create: `src/simulation/client.ts`
- Create: `src/simulation/queue.ts`
- Create: `tests/simulation-client.test.mjs`

**Interfaces:**
- Consumes: `parseServerMessage` from Task 1.
- Produces: `SimulationClient`, `SimulationClientState`, `createSimulationClient(url, options)`.
- Produces: `OutboundQueue.enqueue(message)` and `drain()` with critical-message preservation.

- [ ] **Step 1: Write failing queue and stale-session tests**

```js
test('queue drops superseded observations but keeps reset', () => {
  const queue = new OutboundQueue(4);
  queue.enqueue(resetMessage(1));
  queue.enqueue(observationMessage(2));
  queue.enqueue(observationMessage(3));
  assert.deepEqual(queue.drain().map(item => item.type), ['reset', 'observation']);
});

test('client ignores frames from an old session', () => {
  const client = createHarness('s-current1');
  client.receive({...ready, sessionId: 's-old00001'});
  assert.equal(client.state.phase, 'connecting');
});
```

- [ ] **Step 2: Run and verify failures**

Run: `node --experimental-strip-types --test tests/simulation-client.test.mjs`  
Expected: FAIL because the client and queue are absent.

- [ ] **Step 3: Implement the finite-state client**

```ts
export type SimulationClientState = {
  phase: 'connecting' | 'ready' | 'acting' | 'paused' | 'error' | 'closed';
  sessionId: string;
  episodeId: string;
  backend: 'cpu' | 'gpu' | null;
  error: string | null;
};
```

Use injected socket/timer factories in tests. Keep one pending observation, reject overlapping intentions locally, request a keyframe after a delta gap, and always create a new session ID on reconnect.

- [ ] **Step 4: Run browser tests and build**

Run: `npm test && npm run build`  
Expected: PASS.

- [ ] **Step 5: Commit the browser client**

```bash
git add src/simulation/client.ts src/simulation/queue.ts tests/simulation-client.test.mjs
git commit -m "feat: add resilient simulation websocket client"
```

### Task 5: CPU-default and optional GPU Docker runtime

**Files:**
- Create: `Dockerfile.web`
- Create: `Dockerfile.simulation`
- Create: `compose.yaml`
- Create: `deploy/nginx.conf`
- Create: `.dockerignore`
- Create: `scripts/docker-smoke.sh`
- Create: `python/requirements-linux-x86_64-cpu.txt`
- Modify: `python/requirements-linux-x86_64-cu130.txt`
- Modify: `README.md`

**Interfaces:**
- Consumes: FastAPI app and Vite production build.
- Produces: `docker compose up --build` CPU path and `docker compose --profile gpu up --build simulation-gpu web` GPU path.

- [ ] **Step 1: Write the smoke script before container definitions**

```bash
#!/usr/bin/env bash
set -euo pipefail
curl --fail --silent --show-error http://127.0.0.1:8080/api/healthz | python -c 'import json,sys; assert json.load(sys.stdin)=={"status":"ok","protocol":2}'
curl --fail --silent --show-error http://127.0.0.1:8080/ | grep -q 'Fly Crossy Connectome'
```

- [ ] **Step 2: Verify Compose configuration is currently absent**

Run: `docker compose config`  
Expected: FAIL because `compose.yaml` does not exist.

- [ ] **Step 3: Add non-root multi-stage images and Compose services**

```yaml
services:
  simulation:
    build: {context: ., dockerfile: Dockerfile.simulation}
    environment: {FLY_BACKEND: cpu, FLY_DATA_ROOT: /data}
    volumes: [fly-data:/data]
    healthcheck: {test: [CMD, python, -c, "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"], interval: 5s, timeout: 2s, retries: 12}
  simulation-gpu:
    profiles: [gpu]
    extends: {service: simulation}
    environment: {FLY_BACKEND: gpu, FLY_DATA_ROOT: /data}
    gpus: all
  trainer:
    profiles: [training]
    build: {context: ., dockerfile: Dockerfile.simulation}
    command: [python, -m, fly_crossy.train, --help]
    environment: {FLY_BACKEND: cpu, FLY_DATA_ROOT: /data}
    volumes: [fly-data:/data]
  web:
    build: {context: ., dockerfile: Dockerfile.web}
    ports: ["8080:8080"]
    depends_on: {simulation: {condition: service_healthy}}
  web-gpu:
    profiles: [gpu]
    build: {context: ., dockerfile: Dockerfile.web}
    environment: {SIMULATION_UPSTREAM: simulation-gpu:8000}
    ports: ["8080:8080"]
    depends_on: {simulation-gpu: {condition: service_healthy}}
volumes: {fly-data: {}}
```

Run containers as UID/GID 10001. Render the nginx upstream from `SIMULATION_UPSTREAM` (`simulation:8000` for CPU and `simulation-gpu:8000` for GPU), proxy `/api/`, forward WebSocket upgrade headers, copy only lockfiles before dependency installation, and exclude `.git`, `.worktrees`, `.superpowers`, caches, releases, credentials, and local data from build context. Start GPU mode with `docker compose --profile gpu up --build simulation-gpu web-gpu`; target service names prevent the default CPU pair from starting.

- [ ] **Step 4: Generate hash-locked CPU and GPU dependency files**

Run: `cd python && python -m pip install pip-tools==7.5.0 && pip-compile --generate-hashes --output-file requirements-linux-x86_64-cpu.txt pyproject.toml && pip-compile --generate-hashes --extra-index-url https://download.pytorch.org/whl/cu130 --output-file requirements-linux-x86_64-cu130.txt pyproject.toml`  
Expected: both files pin transitive versions and hashes; Docker builds install with `pip install --require-hashes -r ...`.

- [ ] **Step 5: Validate and run the CPU smoke path**

Run: `docker compose config && docker compose up --build --wait && bash scripts/docker-smoke.sh`  
Expected: config succeeds, services become healthy, and smoke script exits 0.

- [ ] **Step 6: Exercise the opt-in trainer profile**

Run: `docker compose --profile training run --rm trainer`  
Expected: the trainer command prints its help and exits 0 without starting a training run.

- [ ] **Step 7: Tear down only this Compose project and run native suites**

Run: `docker compose down && npm test && cd python && pytest -q`  
Expected: all tests PASS; named volume remains intact.

- [ ] **Step 8: Commit the runtime slice**

```bash
git add Dockerfile.web Dockerfile.simulation compose.yaml deploy/nginx.conf .dockerignore scripts/docker-smoke.sh python/requirements-linux-x86_64-cpu.txt python/requirements-linux-x86_64-cu130.txt README.md
git commit -m "feat: dockerize cpu and gpu simulation runtime"
```

## Plan completion checkpoint

Run: `npm test && npm run build && cd python && pytest -q && cd .. && docker compose config`  
Expected: every command exits 0, CPU service health is documented, and protocol fixtures pass in both languages.
