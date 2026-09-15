from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pytest
from pydantic import ValidationError

import fly_crossy.protocol as protocol
from fly_crossy.protocol import (
    MAX_FRAME_BYTES,
    MAX_NEURAL_UPDATES,
    ServerMessageAdapter,
    parse_client_message,
    parse_server_message,
)


FIXTURE = Path(__file__).parents[2] / "protocol" / "v2" / "valid-session.json"


def test_valid_session_fixture_round_trips() -> None:
    fixture = json.loads(FIXTURE.read_text())

    assert parse_client_message(fixture["client"]).session_id == "s-01234567"
    assert ServerMessageAdapter.validate_python(fixture["server"]).population == 80


def test_client_variants_are_discriminated_and_strict() -> None:
    envelope = {
        "version": 2,
        "sessionId": "s-01234567",
        "episodeId": "e-01234567",
        "sequence": 1,
        "simulationTime": 1.5,
    }

    variants = [
        {"type": "configure", **envelope, "population": 1000, "backend": "gpu-strict", "seed": 7, "speed": 1},
        {"type": "reset", **envelope, "episodeId": "e-abcdefgh", "seed": 7},
        {"type": "observation", **envelope, "gameStep": 4, "observation": [0.0] * 370, "reward": 0.5},
        {"type": "pause", **envelope},
        {"type": "resume", **envelope},
        {"type": "request_keyframe", **envelope},
    ]

    assert [parse_client_message(message).type for message in variants] == [
        "configure", "reset", "observation", "pause", "resume", "request_keyframe",
    ]
    with pytest.raises(ValidationError, match="extra"):
        parse_client_message({"type": "pause", **envelope, "unexpected": True})
    with pytest.raises(ValidationError, match="sessionId"):
        parse_client_message({"type": "pause", **envelope, "sessionId": "bad"})


def test_server_variants_are_bounded_and_reject_invalid_fixture_messages() -> None:
    fixture = json.loads(FIXTURE.read_text())
    envelope = fixture["server"].copy()
    envelope.pop("backend")
    envelope.pop("population")
    envelope.pop("graphHash")
    envelope.pop("checkpointHash")
    envelope.pop("type")
    variants = [
        {"type": "reset_complete", **envelope},
        {"type": "intention", **envelope, "intentionId": "i-01234567", "action": "forward", "motorPhase": "targeting"},
        {"type": "snapshot", **envelope, "body": [0, 0, 0], "joints": [0.0], "keys": {"W": 0.1}},
        {
            "type": "neural_keyframe", **envelope, "revision": 4,
            "chunkIndex": 0, "chunkCount": 1,
            "updates": [{"neuronId": 1, "value": 0.25}],
        },
        {
            "type": "neural_delta", **envelope, "baseRevision": 4, "revision": 5,
            "chunkIndex": 0, "chunkCount": 1,
            "updates": [{"neuronId": 1, "value": 0.25}],
        },
        {"type": "contact", **envelope, "intentionId": "i-01234567", "requestedKey": "W", "touchedKey": "W", "travel": 0.1, "force": 2, "debounce": 0.02, "confirmed": True},
        {"type": "action_result", **envelope, "intentionId": "i-01234567", "result": "confirmed"},
        {"type": "metrics", **envelope, "physicsHz": 1000, "motorHz": 100, "neuralHz": 10, "renderHz": 30, "latencyMs": 12, "droppedRenderFrames": 0},
        {"type": "paused", **envelope, "code": "CLIENT_PAUSED", "message": "paused"},
        {"type": "error", **envelope, "code": "PROTOCOL_ERROR", "message": "bad frame"},
    ]

    assert [ServerMessageAdapter.validate_python(message).type for message in variants] == [
        "reset_complete", "intention", "snapshot", "neural_keyframe", "neural_delta", "contact", "action_result", "metrics", "paused", "error",
    ]
    invalid = json.loads((FIXTURE.parent / "invalid-messages.json").read_text())
    for case in invalid["server"]:
        with pytest.raises(ValidationError):
            ServerMessageAdapter.validate_python(case)
    with pytest.raises(ValidationError, match="updates"):
        ServerMessageAdapter.validate_python({
            "type": "neural_delta", **envelope,
            "baseRevision": 4, "revision": 5, "chunkIndex": 0, "chunkCount": 1,
            "updates": [{"neuronId": index, "value": 0.0} for index in range(20_001)],
        })


def test_neural_frames_require_reconstructible_revision_and_chunk_metadata() -> None:
    fixture = json.loads(FIXTURE.read_text())
    envelope = {
        "version": 2,
        "sessionId": fixture["server"]["sessionId"],
        "episodeId": fixture["server"]["episodeId"],
        "sequence": 1,
        "simulationTime": 1.0,
    }

    with pytest.raises(ValidationError):
        ServerMessageAdapter.validate_python({
            "type": "neural_keyframe", **envelope, "updates": [],
        })
    with pytest.raises(ValidationError):
        ServerMessageAdapter.validate_python({
            "type": "neural_delta", **envelope,
            "revision": 2, "chunkIndex": 0, "chunkCount": 1, "updates": [],
        })
    with pytest.raises(ValidationError, match="nonzero"):
        ServerMessageAdapter.validate_python({
            "type": "neural_keyframe", **envelope,
            "revision": 2, "chunkIndex": 0, "chunkCount": 1,
            "updates": [{"neuronId": 1, "value": 0.0}],
        })


def test_ready_exposes_only_the_bounded_cuda_fallback_reason() -> None:
    fixture = json.loads(FIXTURE.read_text())

    assert ServerMessageAdapter.validate_python(
        {**fixture["server"], "fallbackReason": "cuda-unavailable"}
    ).fallback_reason == "cuda-unavailable"
    with pytest.raises(ValidationError, match="fallbackReason"):
        ServerMessageAdapter.validate_python(
            {**fixture["server"], "fallbackReason": "other"}
        )
    with pytest.raises(ValidationError, match="fallbackReason"):
        ServerMessageAdapter.validate_python(
            {**fixture["server"], "fallbackReason": None}
        )


def test_client_parser_rejects_oversized_frames_and_nonfinite_numbers() -> None:
    frame = json.dumps({
        "type": "hello", "version": 2, "sessionId": "s-01234567", "episodeId": "e-01234567",
        "sequence": 0, "simulationTime": 0, "supportedVersions": [2], "uiBuild": "x" * (1024 * 1024),
    })

    with pytest.raises(ValueError, match="1 MiB"):
        parse_client_message(frame)
    with pytest.raises(ValidationError, match="finite"):
        parse_client_message({
            "type": "observation", "version": 2, "sessionId": "s-01234567", "episodeId": "e-01234567",
            "sequence": 0, "simulationTime": 0, "gameStep": 0, "observation": [float("nan")], "reward": 0,
        })


@pytest.mark.parametrize(("field", "value"), [
    ("sequence", "0"),
    ("sequence", True),
    ("simulationTime", "0"),
    ("simulationTime", True),
])
def test_python_server_parser_rejects_non_numeric_json_values(field: str, value: object) -> None:
    fixture = json.loads(FIXTURE.read_text())

    with pytest.raises(ValidationError):
        ServerMessageAdapter.validate_python({**fixture["server"], field: value})


def test_protocol_declares_the_pydantic_alias_configuration_floor() -> None:
    pyproject = tomllib.loads((FIXTURE.parents[2] / "python" / "pyproject.toml").read_text())

    assert "pydantic>=2.11" in pyproject["project"]["dependencies"]


def test_server_parser_rejects_a_valid_neural_shape_over_the_actual_byte_limit() -> None:
    message = {
        "type": "neural_keyframe",
        "version": 2,
        "sessionId": "s-01234567",
        "episodeId": "e-01234567",
        "sequence": 7,
        "simulationTime": 1.0,
        "revision": 3,
        "chunkIndex": 0,
        "chunkCount": 1,
        "updates": [
            {"neuronId": 2**53 - 1, "value": 999_999.999_999_999_9}
            for _ in range(MAX_NEURAL_UPDATES)
        ],
    }
    payload = json.dumps(message, separators=(",", ":"), allow_nan=False)
    assert len(payload.encode("utf-8")) > MAX_FRAME_BYTES

    with pytest.raises(ValueError, match="1 MiB"):
        parse_server_message(payload)


def test_neural_frames_split_deterministically_by_count_and_actual_bytes() -> None:
    updates = [
        {"neuronId": 2**53 - 1 - index, "value": 999_999.999_999_999_9}
        for index in range(MAX_NEURAL_UPDATES)
    ]
    arguments = {
        "message_type": "neural_keyframe",
        "session_id": "s-01234567",
        "episode_id": "e-01234567",
        "start_sequence": 7,
        "simulation_time": 1.0,
        "revision": 3,
        "updates": updates,
    }

    first = protocol.chunk_neural_frames(**arguments)
    second = protocol.chunk_neural_frames(**arguments)

    assert [frame.model_dump() for frame in first] == [frame.model_dump() for frame in second]
    assert len(first) > 1
    assert [frame.chunk_index for frame in first] == list(range(len(first)))
    assert {frame.chunk_count for frame in first} == {len(first)}
    assert [update.neuron_id for frame in first for update in frame.updates] == [
        2**53 - 1 - index for index in range(MAX_NEURAL_UPDATES)
    ]
    for frame in first:
        payload = protocol.serialize_server_message(frame)
        assert len(payload.encode("utf-8")) <= MAX_FRAME_BYTES
        assert len(frame.updates) <= MAX_NEURAL_UPDATES
        assert parse_server_message(payload) == frame
