from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from fly_crossy.backend import BackendStatus
from fly_crossy.protocol import MAX_FRAME_BYTES
from fly_crossy.server import (
    ArtifactIdentity,
    CLOSE_ARTIFACT_UNAVAILABLE,
    CLOSE_INCOMPATIBLE_VERSION,
    CLOSE_MALFORMED_MESSAGE,
    CLOSE_ORIGIN_FORBIDDEN,
    CLOSE_OVERSIZE_FRAME,
    CLOSE_SEQUENCE_VIOLATION,
    app,
    create_app,
)


TEST_ARTIFACTS = {
    80: ArtifactIdentity(
        graph_hash="a" * 64,
        checkpoint_hash="b" * 64,
    )
}
SAME_ORIGIN = {"origin": "http://testserver"}


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app(artifact_registry=TEST_ARTIFACTS)) as test_client:
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


def test_socket_requires_hello_before_configuration(client: TestClient) -> None:
    with client.websocket_connect("/api/simulation", headers=SAME_ORIGIN) as socket:
        socket.send_json(configure())
        with pytest.raises(WebSocketDisconnect) as error:
            socket.receive_json()

    assert error.value.code == CLOSE_SEQUENCE_VIOLATION


def test_socket_configures_and_emits_the_safe_wait_placeholder(client: TestClient) -> None:
    with client.websocket_connect("/api/simulation", headers=SAME_ORIGIN) as socket:
        socket.send_json(hello())
        socket.send_json(configure())
        ready = socket.receive_json()
        socket.send_json(observation())
        intention = socket.receive_json()

    assert ready["type"] == "ready"
    assert ready["sequence"] == 0
    assert ready["backend"] == "cpu"
    assert ready["graphHash"] == "a" * 64
    assert ready["checkpointHash"] == "b" * 64
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


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ("{", CLOSE_MALFORMED_MESSAGE),
        ("x" * (MAX_FRAME_BYTES + 1), CLOSE_OVERSIZE_FRAME),
    ],
)
def test_socket_closes_invalid_first_frames_with_stable_codes(
    client: TestClient, payload: str, code: int
) -> None:
    with client.websocket_connect("/api/simulation", headers=SAME_ORIGIN) as socket:
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


def test_socket_exposes_cuda_fallback_from_backend_status() -> None:
    def cpu_fallback(_: str) -> BackendStatus:
        return BackendStatus("auto", "cpu", "cpu", "cuda-unavailable")

    fallback_app = create_app(
        artifact_registry=TEST_ARTIFACTS,
        backend_resolver=cpu_fallback,
    )
    with TestClient(fallback_app) as fallback_client:
        with fallback_client.websocket_connect("/api/simulation", headers=SAME_ORIGIN) as socket:
            socket.send_json(hello())
            socket.send_json(configure(backend="auto"))
            ready = socket.receive_json()

    assert ready["fallbackReason"] == "cuda-unavailable"


def test_socket_accepts_a_same_origin_connection(client: TestClient) -> None:
    with client.websocket_connect("/api/simulation", headers=SAME_ORIGIN):
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
