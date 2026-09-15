"""Small FastAPI transport for bounded protocol-v2 simulation sessions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from fly_crossy.backend import BackendUnavailable, resolve_backend
from fly_crossy.protocol import (
    MAX_FRAME_BYTES,
    Configure,
    Error,
    Hello,
    MAX_NEURAL_UPDATES,
    Pause,
    Paused,
    RequestKeyframe,
    Reset,
    ResetComplete,
    Resume,
    Ready,
    parse_client_message,
)
from fly_crossy.session import SessionFault, SimulationSession


logger = logging.getLogger(__name__)

CLOSE_OVERSIZE_FRAME = 4400
CLOSE_INCOMPATIBLE_VERSION = 4401
CLOSE_MALFORMED_MESSAGE = 4402
CLOSE_SEQUENCE_VIOLATION = 4403
CLOSE_BACKEND_UNAVAILABLE = 4404

_PLACEHOLDER_HASH = "0" * 64
_POPULATIONS = [80, 1000, 5000, 20_000, 124_289]

app = FastAPI()


class FrameFault(Exception):
    """A bounded WebSocket frame that is safe to reject publicly."""

    def __init__(self, close_code: int, public_message: str) -> None:
        self.close_code = close_code
        self.public_message = public_message
        super().__init__(public_message)


@dataclass(slots=True)
class SessionSender:
    websocket: WebSocket
    session: SimulationSession
    sequence: int = 0

    async def send(self, message: BaseModel) -> None:
        sequenced = message.model_copy(update={"sequence": self.sequence})
        self.sequence += 1
        await self.websocket.send_json(sequenced.model_dump(by_alias=True))

    async def error(self, code: str, message: str) -> None:
        await self.send(
            Error.model_validate(
                {
                    "type": "error",
                    "version": 2,
                    "sessionId": self.session.session_id,
                    "episodeId": self.session.episode_id,
                    "sequence": 0,
                    "simulationTime": 0.0,
                    "code": code,
                    "message": message,
                }
            )
        )


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    return {"status": "ok", "protocol": 2}


@app.get("/api/capabilities")
async def capabilities() -> dict[str, object]:
    return {
        "protocol": 2,
        "populations": _POPULATIONS,
        "backends": ["cpu", "gpu"],
        "maxFrameBytes": MAX_FRAME_BYTES,
    }


@app.websocket("/api/simulation")
async def simulation_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    await serve_session(websocket, max_bytes=MAX_FRAME_BYTES)


async def serve_session(websocket: WebSocket, *, max_bytes: int) -> None:
    """Serve one connection; state transitions remain owned by SimulationSession."""
    session_id: str | None = None
    sender: SessionSender | None = None
    try:
        first = _parse_frame(await _receive_frame(websocket, max_bytes))
        if not isinstance(first, Hello):
            await websocket.close(code=CLOSE_SEQUENCE_VIOLATION)
            return

        session = SimulationSession(first.session_id, first.episode_id)
        session_id = session.session_id
        sender = SessionSender(websocket, session)

        while True:
            message = _parse_frame(await _receive_frame(websocket, max_bytes))
            await _apply_message(message, sender)
    except WebSocketDisconnect:
        return
    except FrameFault as fault:
        if sender is not None:
            await sender.error("MALFORMED_MESSAGE", fault.public_message)
        await websocket.close(code=fault.close_code)
    except SessionFault as fault:
        if sender is not None:
            await sender.error(fault.code, fault.public_message)
        await websocket.close(
            code=CLOSE_SEQUENCE_VIOLATION
            if fault.code == "SEQUENCE_VIOLATION"
            else CLOSE_MALFORMED_MESSAGE
        )
    except BackendUnavailable as error:
        if sender is not None:
            await sender.error("BACKEND_UNAVAILABLE", str(error))
        await websocket.close(code=CLOSE_BACKEND_UNAVAILABLE)
    except Exception:
        logger.exception("Simulation session failed session_id=%s", session_id)
        if sender is not None:
            await sender.error("INTERNAL_ERROR", "Simulation session error.")
        await websocket.close(code=1011)


async def _apply_message(message: object, sender: SessionSender) -> None:
    session = sender.session
    if isinstance(message, Hello):
        raise SessionFault("SEQUENCE_VIOLATION", "Hello is only valid as the first message.")
    if isinstance(message, Configure):
        session.configure(message)
        backend = resolve_backend(message.backend)
        await sender.send(
            Ready.model_validate(
                {
                    "type": "ready",
                    "version": 2,
                    "sessionId": session.session_id,
                    "episodeId": session.episode_id,
                    "sequence": 0,
                    "simulationTime": message.simulation_time,
                    "backend": backend.resolved,
                    "population": message.population,
                    "graphHash": _PLACEHOLDER_HASH,
                    "checkpointHash": _PLACEHOLDER_HASH,
                    "acceptedVersions": [2],
                    "maxFrameBytes": MAX_FRAME_BYTES,
                    "maxNeuralUpdates": MAX_NEURAL_UPDATES,
                }
            )
        )
        return
    if isinstance(message, Reset):
        session.reset(message)
        await sender.send(
            ResetComplete.model_validate(
                {
                    "type": "reset_complete",
                    "version": 2,
                    "sessionId": session.session_id,
                    "episodeId": session.episode_id,
                    "sequence": 0,
                    "simulationTime": message.simulation_time,
                }
            )
        )
        return
    if isinstance(message, Pause):
        session.pause(message)
        await sender.send(
            Paused.model_validate(
                {
                    "type": "paused",
                    "version": 2,
                    "sessionId": session.session_id,
                    "episodeId": session.episode_id,
                    "sequence": 0,
                    "simulationTime": message.simulation_time,
                    "code": "PAUSED",
                    "message": "Session paused.",
                }
            )
        )
        return
    if isinstance(message, Resume):
        session.resume(message)
        return
    if isinstance(message, RequestKeyframe):
        raise SessionFault("UNSUPPORTED_MESSAGE", "Keyframes are not available in this runtime skeleton.")

    intention = session.accept_observation(message)
    await sender.send(intention)


async def _receive_frame(websocket: WebSocket, max_bytes: int) -> str | bytes:
    frame = await websocket.receive()
    if frame["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(frame.get("code", 1000))
    payload = frame.get("bytes") if frame.get("bytes") is not None else frame.get("text")
    if not isinstance(payload, (str, bytes)):
        raise FrameFault(CLOSE_MALFORMED_MESSAGE, "Protocol frame must be text or bytes.")
    size = len(payload.encode("utf-8")) if isinstance(payload, str) else len(payload)
    if size > max_bytes:
        raise FrameFault(CLOSE_OVERSIZE_FRAME, "Protocol JSON frame exceeds 1 MiB.")
    return payload


def _parse_frame(payload: str | bytes) -> object:
    try:
        return parse_client_message(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as error:
        if _is_incompatible_version(payload):
            raise FrameFault(CLOSE_INCOMPATIBLE_VERSION, "Protocol version 2 is required.") from error
        raise FrameFault(CLOSE_MALFORMED_MESSAGE, "Protocol message is malformed.") from error


def _is_incompatible_version(payload: str | bytes) -> bool:
    try:
        decoded = json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(decoded, dict):
        return False
    if decoded.get("version") != 2:
        return True
    versions = decoded.get("supportedVersions")
    return isinstance(versions, list) and all(isinstance(version, int) for version in versions) and 2 not in versions
