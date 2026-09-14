from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fly_crossy.protocol import ServerMessageAdapter, parse_client_message


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
        {"type": "neural_keyframe", **envelope, "updates": [{"neuronId": 1, "value": 0.25}]},
        {"type": "neural_delta", **envelope, "updates": [{"neuronId": 1, "value": 0.25}]},
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
            "updates": [{"neuronId": index, "value": 0.0} for index in range(20_001)],
        })


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
