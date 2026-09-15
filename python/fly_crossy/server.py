"""Small FastAPI transport for bounded protocol-v2 simulation sessions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from fly_crossy.biomechanics.motor import MotorIntention
from fly_crossy.biomechanics.world import BiomechanicalWorld
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping
from urllib.parse import urlsplit

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from fly_crossy.backend import BackendStatus, BackendUnavailable, resolve_backend
from fly_crossy.protocol import (
    BackendPreference,
    MAX_FRAME_BYTES,
    Configure,
    Error,
    Hello,
    MAX_NEURAL_UPDATES,
    NeuralUpdate,
    Pause,
    Paused,
    RequestKeyframe,
    Reset,
    ResetComplete,
    Resume,
    Ready,
    ServerMessage,
    Action,
    Observation,
    chunk_neural_frames,
    parse_client_message,
    serialize_server_message,
)
from fly_crossy.session import SessionFault, SimulationSession


logger = logging.getLogger(__name__)

CLOSE_OVERSIZE_FRAME = 4400
CLOSE_INCOMPATIBLE_VERSION = 4401
CLOSE_MALFORMED_MESSAGE = 4402
CLOSE_SEQUENCE_VIOLATION = 4403
CLOSE_BACKEND_UNAVAILABLE = 4404
CLOSE_ORIGIN_FORBIDDEN = 4405
CLOSE_ARTIFACT_UNAVAILABLE = 4406

_POPULATIONS = [80, 1000, 5000, 20_000, 124_289]
_HASH_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_DEFAULT_ARTIFACT_REGISTRY: Mapping[int, "_ArtifactIdentity"] = MappingProxyType({})
_DEVELOPMENT_ORIGINS = frozenset({"http://127.0.0.1:5173", "http://localhost:5173"})

ActionSelector = Callable[[Observation], Action]
WorldFactory = Callable[[], BiomechanicalWorld]


def _safe_wait_selector(_: Observation) -> Action:
    return "wait"

class FrameFault(Exception):
    """A bounded WebSocket frame that is safe to reject publicly."""

    def __init__(self, close_code: int, public_message: str) -> None:
        self.close_code = close_code
        self.public_message = public_message
        super().__init__(public_message)


class ArtifactUnavailable(RuntimeError):
    """Raised when a configure request has no verified controller identity."""


class ArtifactRegistryError(ValueError):
    """Raised when a purported controller manifest cannot be verified."""


@dataclass(frozen=True, slots=True, init=False)
class _ArtifactIdentity:
    """Verified graph and checkpoint identities registered for one population."""

    graph_hash: str
    checkpoint_hash: str

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("Artifact identities must be loaded from a verified manifest.")


def _verified_identity(graph_hash: str, checkpoint_hash: str) -> _ArtifactIdentity:
    identity = object.__new__(_ArtifactIdentity)
    object.__setattr__(identity, "graph_hash", graph_hash)
    object.__setattr__(identity, "checkpoint_hash", checkpoint_hash)
    return identity


def load_verified_artifact_registry(manifest_path: Path) -> Mapping[int, _ArtifactIdentity]:
    """Load only manifest-declared graph/checkpoint files whose bytes hash exactly."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactRegistryError("Artifact manifest must be readable JSON.") from error
    if not isinstance(manifest, dict) or set(manifest) != {"artifacts"}:
        raise ArtifactRegistryError("Artifact manifest must contain only an artifacts list.")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list):
        raise ArtifactRegistryError("Artifact manifest artifacts must be a list.")

    root = manifest_path.parent.resolve()
    registry: dict[int, _ArtifactIdentity] = {}
    for entry in artifacts:
        if not isinstance(entry, dict) or set(entry) != {"population", "graph", "checkpoint"}:
            raise ArtifactRegistryError("Each artifact entry must declare population, graph, and checkpoint.")
        population = entry["population"]
        if type(population) is not int or population not in _POPULATIONS:
            raise ArtifactRegistryError("Artifact population is unsupported.")
        if population in registry:
            raise ArtifactRegistryError("Artifact manifest has duplicate populations.")
        graph_hash = _verify_artifact_reference(root, entry["graph"], "graph")
        checkpoint_hash = _verify_artifact_reference(root, entry["checkpoint"], "checkpoint")
        registry[population] = _verified_identity(graph_hash, checkpoint_hash)
    return MappingProxyType(registry)


def _verify_artifact_reference(root: Path, value: Any, label: str) -> str:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ArtifactRegistryError(f"Artifact {label} must declare only path and sha256.")
    relative_path = value["path"]
    expected_hash = value["sha256"]
    if not isinstance(relative_path, str) or not relative_path:
        raise ArtifactRegistryError(f"Artifact {label} path is invalid.")
    if not isinstance(expected_hash, str) or not _HASH_PATTERN.fullmatch(expected_hash):
        raise ArtifactRegistryError(f"Artifact {label} sha256 is invalid.")
    if expected_hash == "0" * 64:
        raise ArtifactRegistryError(f"Artifact {label} sha256 must not be all zero.")

    reference = Path(relative_path)
    if reference.is_absolute():
        raise ArtifactRegistryError(f"Artifact {label} path must be relative to the manifest.")
    candidate = (root / reference).resolve()
    if not candidate.is_relative_to(root):
        raise ArtifactRegistryError(f"Artifact {label} path escapes the manifest directory.")
    if not candidate.is_file():
        raise ArtifactRegistryError(f"Artifact {label} file is unavailable.")
    actual_hash = _sha256_file(candidate)
    if actual_hash != expected_hash:
        raise ArtifactRegistryError(f"Artifact {label} sha256 does not match its manifest.")
    return actual_hash


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class SessionSender:
    websocket: WebSocket
    session: SimulationSession
    sequence: int = 0

    async def send(self, message: ServerMessage) -> None:
        sequenced = message.model_copy(update={"sequence": self.sequence})
        payload = serialize_server_message(sequenced)
        await self.websocket.send_text(payload)
        self.sequence += 1

    async def send_neural(
        self,
        message_type: Literal["neural_keyframe", "neural_delta"],
        *,
        revision: int,
        simulation_time: float,
        updates: Sequence[NeuralUpdate | dict[str, Any]],
        base_revision: int | None = None,
    ) -> None:
        frames = chunk_neural_frames(
            message_type=message_type,
            session_id=self.session.session_id,
            episode_id=self.session.episode_id,
            start_sequence=self.sequence,
            simulation_time=simulation_time,
            revision=revision,
            base_revision=base_revision,
            updates=updates,
        )
        for frame in frames:
            await self.send(frame)

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


def create_app(
    *,
    artifact_manifest: Path | None = None,
    allowed_origins: frozenset[str] = _DEVELOPMENT_ORIGINS,
    backend_resolver: Callable[[BackendPreference], BackendStatus] = resolve_backend,
    action_selector: ActionSelector = _safe_wait_selector,
    world_factory: WorldFactory | None = None,
) -> FastAPI:
    """Create a service with explicit artifacts and browser-origin boundaries."""
    artifact_registry = (
        _DEFAULT_ARTIFACT_REGISTRY
        if artifact_manifest is None
        else load_verified_artifact_registry(artifact_manifest)
    )
    application = FastAPI()

    @application.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"status": "ok", "protocol": 2}

    @application.get("/api/capabilities")
    async def capabilities() -> dict[str, object]:
        return {
            "protocol": 2,
            "populations": _POPULATIONS,
            "backends": ["cpu", "gpu"],
            "maxFrameBytes": MAX_FRAME_BYTES,
        }

    @application.websocket("/api/simulation")
    async def simulation_socket(websocket: WebSocket) -> None:
        if not _origin_is_allowed(websocket, allowed_origins):
            await websocket.close(code=CLOSE_ORIGIN_FORBIDDEN)
            return
        await websocket.accept()
        await serve_session(
            websocket,
            max_bytes=MAX_FRAME_BYTES,
            artifact_registry=artifact_registry,
            backend_resolver=backend_resolver,
            action_selector=action_selector,
            world_factory=world_factory,
        )

    return application


app = create_app()


async def serve_session(
    websocket: WebSocket,
    *,
    max_bytes: int,
    artifact_registry: Mapping[int, _ArtifactIdentity] = _DEFAULT_ARTIFACT_REGISTRY,
    backend_resolver: Callable[[BackendPreference], BackendStatus] = resolve_backend,
    action_selector: ActionSelector = _safe_wait_selector,
    world_factory: WorldFactory | None = None,
) -> None:
    """Serve one connection; state transitions remain owned by SimulationSession."""
    session_id: str | None = None
    sender: SessionSender | None = None
    try:
        first = _parse_frame(await _receive_frame(websocket, max_bytes))
        if not isinstance(first, Hello) or first.sequence != 0:
            await websocket.close(code=CLOSE_SEQUENCE_VIOLATION)
            return

        session = SimulationSession(
            first.session_id,
            first.episode_id,
            initial_sequence=first.sequence,
        )
        session_id = session.session_id
        sender = SessionSender(websocket, session)
        world = world_factory() if world_factory is not None else None

        while True:
            message = _parse_frame(await _receive_frame(websocket, max_bytes))
            await _apply_message(
                message,
                sender,
                artifact_registry,
                backend_resolver,
                action_selector,
                world,
            )
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
    except ArtifactUnavailable as error:
        if sender is not None:
            await sender.error("ARTIFACT_UNAVAILABLE", str(error))
        await websocket.close(code=CLOSE_ARTIFACT_UNAVAILABLE)
    except Exception:
        logger.exception("Simulation session failed session_id=%s", session_id)
        if sender is not None:
            await sender.error("INTERNAL_ERROR", "Simulation session error.")
        await websocket.close(code=1011)


async def _apply_message(
    message: object,
    sender: SessionSender,
    artifact_registry: Mapping[int, _ArtifactIdentity],
    backend_resolver: Callable[[BackendPreference], BackendStatus],
    action_selector: ActionSelector,
    world: BiomechanicalWorld | None,
) -> None:
    session = sender.session
    if isinstance(message, Hello):
        raise SessionFault("SEQUENCE_VIOLATION", "Hello is only valid as the first message.")
    if isinstance(message, Configure):
        session.configure(message)
        artifact = artifact_registry.get(message.population)
        if artifact is None:
            raise ArtifactUnavailable("No verified artifact is available for this population.")
        backend = backend_resolver(message.backend)
        ready = {
            "type": "ready",
            "version": 2,
            "sessionId": session.session_id,
            "episodeId": session.episode_id,
            "sequence": 0,
            "simulationTime": message.simulation_time,
            "backend": backend.resolved,
            "population": message.population,
            "graphHash": artifact.graph_hash,
            "checkpointHash": artifact.checkpoint_hash,
            "acceptedVersions": [2],
            "maxFrameBytes": MAX_FRAME_BYTES,
            "maxNeuralUpdates": MAX_NEURAL_UPDATES,
        }
        if backend.fallback_reason is not None:
            ready["fallbackReason"] = backend.fallback_reason
        await sender.send(
            Ready.model_validate(ready)
        )
        return
    if isinstance(message, Reset):
        session.reset(message)
        if world is not None:
            world.reset()
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
        session.request_keyframe(message)
        await sender.send_neural(
            "neural_keyframe",
            revision=0,
            simulation_time=message.simulation_time,
            updates=[],
        )
        return

    if not isinstance(message, Observation):
        raise SessionFault(
            "INVALID_MESSAGE",
            "Unsupported simulation message.",
        )

    action = action_selector(message)

    if action not in ("forward", "backward", "left", "right", "wait"):
        raise SessionFault(
            "INVALID_ACTION",
            "Controller returned an unsupported action.",
        )

    motor_phase = "neutral" if action == "wait" else "targeting"

    intention = session.accept_observation(
        message,
        action=action,
        motor_phase=motor_phase,
    )

    await sender.send(intention)

    # The safe placeholder can complete without instantiating MuJoCo.
    if world is None:
        if action != "wait":
            raise SessionFault(
                "BIOMECHANICS_UNAVAILABLE",
                "Directional action requires the biomechanical runtime.",
            )

        terminal = session.finish_action(
            intention.intention_id,
            "waited",
            completion_time=message.simulation_time,
        )

        if terminal is not None:
            await sender.send(terminal)

        return

    physical_result = await asyncio.to_thread(
        world.run,
        MotorIntention(
            intention.intention_id,
            action,
        ),
        1.5,
    )

    recovery_pending = not world.ready

    terminal = session.finish_action(
        intention.intention_id,
        physical_result.outcome,
        completion_time=physical_result.completion_time,
        recovery_pending=recovery_pending,
    )

    if terminal is not None:
        await sender.send(terminal)

    if recovery_pending:
        await asyncio.to_thread(
            _recover_world,
            world,
        )
        session.finish_recovery()

def _recover_world(
    world: BiomechanicalWorld,
) -> None:
    """Recover physically when possible, otherwise perform the required reset."""

    for _ in range(1500):
        if world.ready:
            return

        if world.requires_reset:
            reason = world.failure_reason or "unknown"

            logger.warning(
                "Biomechanical recovery requires coordinated reset: %s",
                reason,
            )

            world.reset()

            if not world.ready:
                raise RuntimeError(
                    "Biomechanical reset failed to restore neutral state."
                )

            return

        world.step()

    # A recovery that never reached ready is not allowed to kill the
    # WebSocket session. Reset is the explicit fail-safe boundary.
    logger.warning(
        "Biomechanical recovery exceeded server deadline; "
        "performing coordinated reset."
    )

    world.reset()

    if not world.ready:
        raise RuntimeError(
            "Biomechanical reset failed to restore neutral state."
        )


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


def _origin_is_allowed(websocket: WebSocket, allowed_origins: frozenset[str]) -> bool:
    origin = websocket.headers.get("origin")
    if origin is None:
        return False
    if origin in allowed_origins:
        return True

    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
        return False
    host = websocket.headers.get("host")
    if host != parsed.netloc:
        return False
    scheme = websocket.headers.get("x-forwarded-proto", websocket.url.scheme)
    if scheme == "ws":
        scheme = "http"
    elif scheme == "wss":
        scheme = "https"
    return parsed.scheme == scheme
