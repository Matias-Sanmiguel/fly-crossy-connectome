from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from fly_crossy.server import _recover_world

from fly_crossy.biomechanics.world import WorldActionResult
from fly_crossy.backend import BackendStatus
from fly_crossy.protocol import Error, MAX_FRAME_BYTES, MAX_NEURAL_UPDATES, parse_server_message
from fly_crossy.session import SimulationSession
from fly_crossy.server import (
    ArtifactRegistryError,
    CLOSE_ARTIFACT_UNAVAILABLE,
    CLOSE_INCOMPATIBLE_VERSION,
    CLOSE_MALFORMED_MESSAGE,
    CLOSE_ORIGIN_FORBIDDEN,
    CLOSE_OVERSIZE_FRAME,
    CLOSE_SEQUENCE_VIOLATION,
    app,
    create_app,
    load_verified_artifact_registry,
    SessionSender,
)


SAME_ORIGIN = {"origin": "http://testserver"}


@pytest.fixture
def artifact_manifest(tmp_path: Path) -> Path:
    graph = tmp_path / "graph.bin"
    checkpoint = tmp_path / "checkpoint.bin"
    graph.write_bytes(b"verified graph")
    checkpoint.write_bytes(b"verified checkpoint")
    manifest = tmp_path / "artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "population": 80,
                        "graph": {
                            "path": graph.name,
                            "sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
                        },
                        "checkpoint": {
                            "path": checkpoint.name,
                            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest


@pytest.fixture
def client(artifact_manifest: Path) -> TestClient:
    with TestClient(create_app(artifact_manifest=artifact_manifest)) as test_client:
        yield test_client


def hello() -> dict[str, object]:
    return {
        "type": "hello",
        "version": 2,
        "sessionId": "s-00000001",
        "episodeId": "e-00000001",
        "sequence": 0,
        "simulationTime": 0.0,
        "supportedVersions": [2],
        "uiBuild": "test",
    }


def configure(*, backend: str = "cpu") -> dict[str, object]:
    return {
        "type": "configure",
        "version": 2,
        "sessionId": "s-00000001",
        "episodeId": "e-00000001",
        "sequence": 1,
        "simulationTime": 0.0,
        "population": 80,
        "backend": backend,
        "seed": 7,
        "speed": 1.0,
    }


def observation() -> dict[str, object]:
    return {
        "type": "observation",
        "version": 2,
        "sessionId": "s-00000001",
        "episodeId": "e-00000001",
        "sequence": 2,
        "simulationTime": 1.0,
        "gameStep": 0,
        "observation": [0.0] * 370,
        "reward": 0.0,
    }


def request_keyframe(sequence: int = 2) -> dict[str, object]:
    return {
        "type": "request_keyframe",
        "version": 2,
        "sessionId": "s-00000001",
        "episodeId": "e-00000001",
        "sequence": sequence,
        "simulationTime": 1.0,
    }


def test_health_is_small_and_ready(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "protocol": 2}


def test_capabilities_expose_protocol_bounds(client: TestClient) -> None:
    response = client.get("/api/capabilities")

    assert response.status_code == 200
    assert response.json() == {
        "protocol": 2,
        "populations": [80, 1000, 5000, 20000, 124289],
        "backends": ["cpu", "gpu"],
        "maxFrameBytes": MAX_FRAME_BYTES,
    }


def test_sender_fails_closed_before_writing_an_oversized_non_neural_frame() -> None:
    class RecordingSocket:
        def __init__(self) -> None:
            self.payloads: list[str] = []

        async def send_text(self, payload: str) -> None:
            self.payloads.append(payload)

    socket = RecordingSocket()
    session = SimulationSession("s-00000001", "e-00000001")
    sender = SessionSender(socket, session)  # type: ignore[arg-type]
    oversized = Error.model_construct(
        type="error",
        version=2,
        session_id="s-00000001",
        episode_id="e-00000001",
        sequence=0,
        simulation_time=0.0,
        code="OVERSIZE",
        message="x" * MAX_FRAME_BYTES,
    )

    with pytest.raises(ValueError, match="1 MiB|256"):
        asyncio.run(sender.send(oversized))

    assert socket.payloads == []
    assert sender.sequence == 0


def test_sender_chunks_a_large_neural_frame_with_contiguous_bounded_sequences() -> None:
    class RecordingSocket:
        def __init__(self) -> None:
            self.payloads: list[str] = []

        async def send_text(self, payload: str) -> None:
            self.payloads.append(payload)

    socket = RecordingSocket()
    session = SimulationSession("s-00000001", "e-00000001")
    sender = SessionSender(socket, session)  # type: ignore[arg-type]
    updates = [
        {"neuronId": 2**53 - 1 - index, "value": 999_999.999_999_999_9}
        for index in range(MAX_NEURAL_UPDATES)
    ]

    asyncio.run(sender.send_neural(
        "neural_keyframe",
        revision=9,
        simulation_time=3.0,
        updates=updates,
    ))

    frames = [parse_server_message(payload) for payload in socket.payloads]
    assert len(frames) > 1
    assert [frame.sequence for frame in frames] == list(range(len(frames)))
    assert [frame.chunk_index for frame in frames] == list(range(len(frames)))
    assert {frame.chunk_count for frame in frames} == {len(frames)}
    assert all(len(payload.encode("utf-8")) <= MAX_FRAME_BYTES for payload in socket.payloads)
    assert sender.sequence == len(frames)


def test_socket_requires_hello_before_configuration(client: TestClient) -> None:
    with client.websocket_connect("/api/simulation", headers=SAME_ORIGIN) as socket:
        socket.send_json(configure())
        with pytest.raises(WebSocketDisconnect) as error:
            socket.receive_json()

    assert error.value.code == CLOSE_SEQUENCE_VIOLATION


def test_socket_configures_and_emits_the_safe_wait_placeholder(
    client: TestClient, artifact_manifest: Path
) -> None:
    with client.websocket_connect("/api/simulation", headers=SAME_ORIGIN) as socket:
        socket.send_json(hello())
        socket.send_json(configure())
        ready = socket.receive_json()
        socket.send_json(observation())
        intention = socket.receive_json()
        result = socket.receive_json()

    assert ready["type"] == "ready"
    assert ready["sequence"] == 0
    assert ready["backend"] == "cpu"
    assert ready["graphHash"] == hashlib.sha256(
        artifact_manifest.with_name("graph.bin").read_bytes()
    ).hexdigest()
    assert ready["checkpointHash"] == hashlib.sha256(
        artifact_manifest.with_name("checkpoint.bin").read_bytes()
    ).hexdigest()
    assert "fallbackReason" not in ready
    assert intention == {
        "type": "intention",
        "version": 2,
        "sessionId": "s-00000001",
        "episodeId": "e-00000001",
        "sequence": 1,
        "simulationTime": 1.0,
        "intentionId": "i-00000001",
        "action": "wait",
        "motorPhase": "neutral",
    }
    assert result == {
        "type": "action_result",
        "version": 2,
        "sessionId": "s-00000001",
        "episodeId": "e-00000001",
        "sequence": 2,
        "simulationTime": 1.0,
        "intentionId": "i-00000001",
        "result": "waited",
    }


def test_socket_answers_keyframe_request_with_an_honest_empty_keyframe(
    client: TestClient,
) -> None:
    with client.websocket_connect("/api/simulation", headers=SAME_ORIGIN) as socket:
        socket.send_json(hello())
        socket.send_json(configure())
        socket.receive_json()
        socket.send_json(request_keyframe())
        keyframe = socket.receive_json()
        socket.send_json({
            "type": "pause", **{
                key: value for key, value in request_keyframe(3).items()
                if key != "type"
            },
        })
        paused = socket.receive_json()

    assert keyframe == {
        "type": "neural_keyframe",
        "version": 2,
        "sessionId": "s-00000001",
        "episodeId": "e-00000001",
        "sequence": 1,
        "simulationTime": 1.0,
        "revision": 0,
        "chunkIndex": 0,
        "chunkCount": 1,
        "updates": [],
    }
    assert paused["type"] == "paused"
    assert paused["sequence"] == 2


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        pytest.param(
            "{",
            CLOSE_MALFORMED_MESSAGE,
            id="malformed-json",
        ),
        pytest.param(
            "x" * (MAX_FRAME_BYTES + 1),
            CLOSE_OVERSIZE_FRAME,
            id="oversize-frame",
        ),
    ],
)
def test_socket_closes_invalid_first_frames_with_stable_codes(
    client: TestClient,
    payload: str,
    code: int,
) -> None:
    with client.websocket_connect(
        "/api/simulation",
        headers=SAME_ORIGIN,
    ) as socket:
        socket.send_text(payload)

        with pytest.raises(WebSocketDisconnect) as error:
            socket.receive_json()

    assert error.value.code == code

def test_socket_closes_incompatible_hello_with_a_stable_code(client: TestClient) -> None:
    with client.websocket_connect("/api/simulation", headers=SAME_ORIGIN) as socket:
        socket.send_json({**hello(), "version": 1, "supportedVersions": [1]})
        with pytest.raises(WebSocketDisconnect) as error:
            socket.receive_json()

    assert error.value.code == CLOSE_INCOMPATIBLE_VERSION


def test_socket_rejects_configuration_without_a_verified_artifact_registry() -> None:
    with TestClient(app) as default_client:
        with default_client.websocket_connect("/api/simulation", headers=SAME_ORIGIN) as socket:
            socket.send_json(hello())
            socket.send_json(configure())
            error = socket.receive_json()
            with pytest.raises(WebSocketDisconnect) as closed:
                socket.receive_json()

    assert error == {
        "type": "error",
        "version": 2,
        "sessionId": "s-00000001",
        "episodeId": "e-00000001",
        "sequence": 0,
        "simulationTime": 0.0,
        "code": "ARTIFACT_UNAVAILABLE",
        "message": "No verified artifact is available for this population.",
    }
    assert closed.value.code == CLOSE_ARTIFACT_UNAVAILABLE


def test_socket_exposes_cuda_fallback_from_backend_status(
    artifact_manifest: Path,
) -> None:
    def cpu_fallback(_: str) -> BackendStatus:
        return BackendStatus("auto", "cpu", "cpu", "cuda-unavailable")

    fallback_app = create_app(
        artifact_manifest=artifact_manifest,
        backend_resolver=cpu_fallback,
    )
    with TestClient(fallback_app) as fallback_client:
        with fallback_client.websocket_connect("/api/simulation", headers=SAME_ORIGIN) as socket:
            socket.send_json(hello())
            socket.send_json(configure(backend="auto"))
            ready = socket.receive_json()

    assert ready["fallbackReason"] == "cuda-unavailable"


def test_socket_omits_fallback_reason_for_a_successful_gpu_backend(
    artifact_manifest: Path,
) -> None:
    def gpu_backend(_: str) -> BackendStatus:
        return BackendStatus("gpu", "gpu", "cuda:0", None)

    gpu_app = create_app(
        artifact_manifest=artifact_manifest,
        backend_resolver=gpu_backend,
    )
    with TestClient(gpu_app) as gpu_client:
        with gpu_client.websocket_connect("/api/simulation", headers=SAME_ORIGIN) as socket:
            socket.send_json(hello())
            socket.send_json(configure(backend="gpu"))
            ready = socket.receive_json()

    assert ready["backend"] == "gpu"
    assert "fallbackReason" not in ready


def test_socket_accepts_a_same_origin_connection(client: TestClient) -> None:
    with client.websocket_connect("/api/simulation", headers=SAME_ORIGIN):
        pass


@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
)
def test_socket_accepts_supported_local_vite_origins(
    client: TestClient,
    origin: str,
) -> None:
    with client.websocket_connect(
        "/api/simulation",
        headers={"origin": origin},
    ):
        pass


def test_socket_rejects_a_disallowed_origin(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect(
            "/api/simulation", headers={"origin": "https://attacker.invalid"}
        ):
            pass

    assert error.value.code == CLOSE_ORIGIN_FORBIDDEN


def test_socket_rejects_a_nonzero_hello_sequence_before_configuration(client: TestClient) -> None:
    with client.websocket_connect("/api/simulation", headers=SAME_ORIGIN) as socket:
        socket.send_json({**hello(), "sequence": 10})
        socket.send_json(configure())
        with pytest.raises(WebSocketDisconnect) as error:
            socket.receive_json()

    assert error.value.code == CLOSE_SEQUENCE_VIOLATION


def test_invalid_configure_sequence_wins_over_missing_artifact() -> None:
    with TestClient(app) as default_client:
        with default_client.websocket_connect("/api/simulation", headers=SAME_ORIGIN) as socket:
            socket.send_json(hello())
            socket.send_json({**configure(), "sequence": 0})
            error = socket.receive_json()
            with pytest.raises(WebSocketDisconnect) as closed:
                socket.receive_json()

    assert error["code"] == "SEQUENCE_VIOLATION"
    assert closed.value.code == CLOSE_SEQUENCE_VIOLATION


def test_verified_registry_loader_hashes_manifest_files(tmp_path: Path) -> None:
    graph = tmp_path / "graph.bin"
    checkpoint = tmp_path / "checkpoint.bin"
    graph.write_bytes(b"graph bytes")
    checkpoint.write_bytes(b"checkpoint bytes")
    manifest = tmp_path / "artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "population": 80,
                        "graph": {
                            "path": graph.name,
                            "sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
                        },
                        "checkpoint": {
                            "path": checkpoint.name,
                            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    registry = load_verified_artifact_registry(manifest)

    assert registry[80].graph_hash == hashlib.sha256(graph.read_bytes()).hexdigest()
    assert registry[80].checkpoint_hash == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_registry_loader_rejects_unverified_or_zero_hash_identities(tmp_path: Path) -> None:
    graph = tmp_path / "graph.bin"
    checkpoint = tmp_path / "checkpoint.bin"
    graph.write_bytes(b"graph bytes")
    checkpoint.write_bytes(b"checkpoint bytes")
    manifest = tmp_path / "artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "population": 80,
                        "graph": {"path": graph.name, "sha256": "0" * 64},
                        "checkpoint": {"path": checkpoint.name, "sha256": "0" * 64},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactRegistryError, match="all zero"):
        load_verified_artifact_registry(manifest)

    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "population": 80,
                        "graph": {"path": graph.name, "sha256": "a" * 64},
                        "checkpoint": {"path": checkpoint.name, "sha256": "b" * 64},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactRegistryError, match="does not match"):
        load_verified_artifact_registry(manifest)


def test_app_construction_rejects_a_forged_all_zero_registry() -> None:
    forged_registry = {
        80: {"graph_hash": "0" * 64, "checkpoint_hash": "0" * 64},
    }

    with pytest.raises(TypeError, match="artifact_registry"):
        create_app(artifact_registry=forged_registry)

def test_socket_direction_waits_for_biomechanical_confirmation(
    artifact_manifest: Path,
) -> None:
    class FakeWorld:
        def __init__(self) -> None:
            self.ready = True
            self.actions: list[str] = []
            self.recovery_steps = 0
            self.requires_reset = False
            self.failure_reason: str | None = None

        def run(
            self,
            intention: object,
            limit_seconds: float = 1.5,
        ) -> WorldActionResult:
            assert limit_seconds == 1.5

            action = getattr(intention, "action")
            intention_id = getattr(intention, "intention_id")

            self.actions.append(action)
            self.ready = False

            return WorldActionResult(
                intention_id=intention_id,
                outcome="confirmed",
                action=action,
                requested_action=action,
                completion_time=1.25,
            )

        def step(self) -> tuple[object, ...]:
            self.recovery_steps += 1
            self.ready = True
            return ()

        def reset(self) -> None:
            self.ready = True
            self.requires_reset = False
            self.failure_reason = None

    world = FakeWorld()

    bridge_app = create_app(
        artifact_manifest=artifact_manifest,
        action_selector=lambda _: "forward",
        world_factory=lambda: world,  # type: ignore[arg-type]
    )

    with TestClient(bridge_app) as bridge_client:
        with bridge_client.websocket_connect(
            "/api/simulation",
            headers=SAME_ORIGIN,
        ) as socket:
            socket.send_json(hello())
            socket.send_json(configure())
            socket.receive_json()

            socket.send_json(observation())

            intention = socket.receive_json()
            result = socket.receive_json()

    assert intention["type"] == "intention"
    assert intention["action"] == "forward"
    assert intention["motorPhase"] == "targeting"

    assert result["type"] == "action_result"
    assert result["intentionId"] == intention["intentionId"]
    assert result["result"] == "confirmed"
    assert result["simulationTime"] == 1.25

    assert world.actions == ["forward"]
    assert world.recovery_steps == 1
    assert world.ready
    
def test_recovery_timeout_performs_required_coordinated_reset() -> None:
    class StuckWorld:
        def __init__(self) -> None:
            self.ready = False
            self.requires_reset = False
            self.failure_reason: str | None = None
            self.steps = 0
            self.resets = 0

        def step(self) -> tuple[object, ...]:
            self.steps += 1

            if self.steps == 3:
                self.requires_reset = True
                self.failure_reason = "recovery-timeout"

            return ()

        def reset(self) -> None:
            self.resets += 1
            self.ready = True
            self.requires_reset = False
            self.failure_reason = None

    world = StuckWorld()

    _recover_world(world)  # type: ignore[arg-type]

    assert world.ready
    assert world.steps == 3
    assert world.resets == 1
    
def test_wait_action_bypasses_biomechanical_world(
    artifact_manifest: Path,
) -> None:
    class FakeWorld:
        def __init__(self) -> None:
            self.run_calls = 0

        def run(
            self,
            intention: object,
            limit_seconds: float = 1.5,
        ) -> WorldActionResult:
            self.run_calls += 1
            raise AssertionError(
                "wait must not enter the biomechanical world"
            )

    world = FakeWorld()

    bridge_app = create_app(
        artifact_manifest=artifact_manifest,
        action_selector=lambda _: "wait",
        world_factory=lambda: world,  # type: ignore[arg-type]
    )

    with TestClient(bridge_app) as bridge_client:
        with bridge_client.websocket_connect(
            "/api/simulation",
            headers=SAME_ORIGIN,
        ) as socket:
            socket.send_json(hello())
            socket.send_json(configure())
            socket.receive_json()

            socket.send_json(observation())

            intention = socket.receive_json()
            result = socket.receive_json()

    assert intention["type"] == "intention"
    assert intention["action"] == "wait"
    assert intention["motorPhase"] == "neutral"

    assert result["type"] == "action_result"
    assert result["result"] == "waited"

    assert world.run_calls == 0
