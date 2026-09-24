"""Strict, bounded wire schemas for simulation protocol version 2."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

MAX_FRAME_BYTES = 1_048_576
MAX_NEURAL_UPDATES = 20_000
MAX_NEURAL_CHUNKS = 256
MAX_STRING_LENGTH = 256
MAX_SEQUENCE = 2**53 - 1
MAX_SIMULATION_TIME = 86_400.0

PopulationSize: TypeAlias = Literal[80, 1000, 5000, 20000, 124289]
BackendPreference: TypeAlias = Literal["auto", "cpu", "gpu", "gpu-strict"]
ResolvedBackend: TypeAlias = Literal["cpu", "gpu"]
FallbackReason: TypeAlias = Literal["cuda-unavailable"]
Action: TypeAlias = Literal["forward", "backward", "left", "right", "wait"]
KeyName: TypeAlias = Literal["W", "A", "S", "D", "SPACE_LEFT", "SPACE_RIGHT"]
MotorPhase: TypeAlias = Literal[
    "neutral",
    "targeting",
    "reaching",
    "pressing",
    "confirmed",
    "lifting",
    "retracting",
    "settling",
    "failed",
]

ShortString = Annotated[str, StringConstraints(min_length=1, max_length=MAX_STRING_LENGTH)]
Hash = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
SessionOrEpisodeId = Annotated[str, StringConstraints(pattern=r"^[se]-[a-z0-9]{8,64}$")]
IntentionId = Annotated[str, StringConstraints(pattern=r"^i-[a-z0-9]{8,64}$")]
FiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
UnitFloat = Annotated[StrictFloat, Field(ge=0, le=1, allow_inf_nan=False)]
PositionFloat = Annotated[StrictFloat, Field(ge=-10_000, le=10_000, allow_inf_nan=False)]
JointFloat = Annotated[StrictFloat, Field(ge=-100, le=100, allow_inf_nan=False)]


class ProtocolModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_by_alias=True,
        validate_by_name=False,
        alias_generator=lambda name: "".join(
            part.capitalize() if index else part for index, part in enumerate(name.split("_"))
        ),
        allow_inf_nan=False,
    )


class Envelope(ProtocolModel):
    version: Literal[2]
    session_id: SessionOrEpisodeId
    episode_id: SessionOrEpisodeId
    sequence: Annotated[StrictInt, Field(ge=0, le=MAX_SEQUENCE)]
    simulation_time: Annotated[StrictFloat, Field(ge=0, le=MAX_SIMULATION_TIME, allow_inf_nan=False)]


class Hello(Envelope):
    type: Literal["hello"]
    supported_versions: Annotated[list[Literal[2]], Field(min_length=1, max_length=1)]
    ui_build: ShortString


class Configure(Envelope):
    type: Literal["configure"]
    population: PopulationSize
    backend: BackendPreference
    seed: Annotated[StrictInt, Field(ge=0, le=2**32 - 1)]
    speed: Annotated[StrictFloat, Field(gt=0, le=100, allow_inf_nan=False)]


class Reset(Envelope):
    type: Literal["reset"]
    seed: Annotated[StrictInt, Field(ge=0, le=2**32 - 1)] | None = None


class Observation(Envelope):
    type: Literal["observation"]
    game_step: Annotated[StrictInt, Field(ge=0, le=MAX_SEQUENCE)]
    observation: Annotated[list[FiniteFloat], Field(min_length=517, max_length=517)]
    reward: Annotated[StrictFloat, Field(ge=-1_000_000, le=1_000_000, allow_inf_nan=False)]


class Pause(Envelope):
    type: Literal["pause"]


class Resume(Envelope):
    type: Literal["resume"]


class RequestKeyframe(Envelope):
    type: Literal["request_keyframe"]


ClientMessage: TypeAlias = Hello | Configure | Reset | Observation | Pause | Resume | RequestKeyframe
ClientMessageAdapter = TypeAdapter(Annotated[ClientMessage, Field(discriminator="type")])


class Ready(Envelope):
    type: Literal["ready"]
    backend: ResolvedBackend
    population: PopulationSize
    graph_hash: Hash
    checkpoint_hash: Hash
    accepted_versions: Annotated[list[Literal[2]], Field(min_length=1, max_length=1)] | None = None
    max_frame_bytes: Annotated[StrictInt, Field(ge=1, le=MAX_FRAME_BYTES)] | None = None
    max_neural_updates: Annotated[StrictInt, Field(ge=1, le=MAX_NEURAL_UPDATES)] | None = None
    fallback_reason: FallbackReason | None = None

    @field_validator("fallback_reason", mode="before")
    @classmethod
    def reject_explicit_null_fallback_reason(cls, value: object) -> object:
        if value is None:
            raise ValueError("fallbackReason must be omitted or a supported reason")
        return value


class ResetComplete(Envelope):
    type: Literal["reset_complete"]


class Intention(Envelope):
    type: Literal["intention"]
    intention_id: IntentionId
    action: Action
    motor_phase: MotorPhase


class Snapshot(Envelope):
    type: Literal["snapshot"]
    body: Annotated[list[PositionFloat], Field(min_length=3, max_length=3)]
    joints: Annotated[list[JointFloat], Field(max_length=256)]
    keys: Annotated[dict[KeyName, UnitFloat], Field(max_length=6)]


class NeuralUpdate(ProtocolModel):
    neuron_id: Annotated[StrictInt, Field(ge=0, le=MAX_SEQUENCE)]
    value: Annotated[StrictFloat, Field(ge=-1_000_000, le=1_000_000, allow_inf_nan=False)]


class NeuralKeyframe(Envelope):
    type: Literal["neural_keyframe"]
    revision: Annotated[StrictInt, Field(ge=0, le=MAX_SEQUENCE)]
    chunk_index: Annotated[StrictInt, Field(ge=0, lt=MAX_NEURAL_CHUNKS)]
    chunk_count: Annotated[StrictInt, Field(ge=1, le=MAX_NEURAL_CHUNKS)]
    updates: Annotated[list[NeuralUpdate], Field(max_length=MAX_NEURAL_UPDATES)]

    @model_validator(mode="after")
    def validate_chunk_position(self) -> Self:
        if self.chunk_index >= self.chunk_count:
            raise ValueError("chunkIndex must be less than chunkCount")
        if any(update.value == 0 for update in self.updates):
            raise ValueError("neural keyframe updates must be nonzero")
        return self


class NeuralDelta(Envelope):
    type: Literal["neural_delta"]
    base_revision: Annotated[StrictInt, Field(ge=0, le=MAX_SEQUENCE)]
    revision: Annotated[StrictInt, Field(ge=1, le=MAX_SEQUENCE)]
    chunk_index: Annotated[StrictInt, Field(ge=0, lt=MAX_NEURAL_CHUNKS)]
    chunk_count: Annotated[StrictInt, Field(ge=1, le=MAX_NEURAL_CHUNKS)]
    updates: Annotated[list[NeuralUpdate], Field(max_length=MAX_NEURAL_UPDATES)]

    @model_validator(mode="after")
    def validate_revision_and_chunk_position(self) -> Self:
        if self.revision <= self.base_revision:
            raise ValueError("revision must be greater than baseRevision")
        if self.chunk_index >= self.chunk_count:
            raise ValueError("chunkIndex must be less than chunkCount")
        return self


class Contact(Envelope):
    type: Literal["contact"]
    intention_id: IntentionId
    requested_key: KeyName
    touched_key: KeyName | None
    travel: UnitFloat
    force: Annotated[StrictFloat, Field(ge=0, le=10_000, allow_inf_nan=False)]
    debounce: Annotated[StrictFloat, Field(ge=0, le=10, allow_inf_nan=False)]
    confirmed: bool


class ActionResult(Envelope):
    type: Literal["action_result"]
    intention_id: IntentionId
    result: Literal["confirmed", "waited", "failed"]


class Metrics(Envelope):
    type: Literal["metrics"]
    physics_hz: Annotated[StrictFloat, Field(ge=0, le=100_000, allow_inf_nan=False)]
    motor_hz: Annotated[StrictFloat, Field(ge=0, le=100_000, allow_inf_nan=False)]
    neural_hz: Annotated[StrictFloat, Field(ge=0, le=100_000, allow_inf_nan=False)]
    render_hz: Annotated[StrictFloat, Field(ge=0, le=100_000, allow_inf_nan=False)]
    latency_ms: Annotated[StrictFloat, Field(ge=0, le=1_000_000, allow_inf_nan=False)]
    dropped_render_frames: Annotated[StrictInt, Field(ge=0, le=MAX_SEQUENCE)]
    backend_utilization: UnitFloat | None = None


class Paused(Envelope):
    type: Literal["paused"]
    code: ShortString
    message: ShortString


class Error(Envelope):
    type: Literal["error"]
    code: ShortString
    message: ShortString


ServerMessage: TypeAlias = (
    Ready | ResetComplete | Intention | Snapshot | NeuralKeyframe | NeuralDelta | Contact | ActionResult | Metrics | Paused | Error
)
ServerMessageAdapter = TypeAdapter(Annotated[ServerMessage, Field(discriminator="type")])


def _load_frame(value: Any) -> Any:
    """Decode one bounded JSON control frame without accepting non-object payloads."""
    if isinstance(value, bytes):
        if len(value) > MAX_FRAME_BYTES:
            raise ValueError("Protocol JSON frame exceeds 1 MiB.")
        value = value.decode("utf-8")
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_FRAME_BYTES:
            raise ValueError("Protocol JSON frame exceeds 1 MiB.")
        value = json.loads(value)
    else:
        try:
            encoded = json.dumps(value, separators=(",", ":"), allow_nan=True).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError("Protocol frame must be JSON serializable.") from error
        if len(encoded) > MAX_FRAME_BYTES:
            raise ValueError("Protocol JSON frame exceeds 1 MiB.")
    if not isinstance(value, dict):
        raise ValueError("Protocol frame must be a JSON object.")
    return value


def parse_client_message(value: Any) -> ClientMessage:
    """Validate a bounded v2 client message from decoded JSON or a JSON frame."""
    return ClientMessageAdapter.validate_python(_load_frame(value))


def parse_server_message(value: Any) -> ServerMessage:
    """Validate a bounded v2 server message from decoded JSON or a JSON frame."""
    return ServerMessageAdapter.validate_python(_load_frame(value))


def serialize_server_message(message: ServerMessage | BaseModel | dict[str, Any]) -> str:
    """Validate and serialize exactly one server frame within the advertised byte limit."""
    value = (
        message.model_dump(by_alias=True, exclude_none=True)
        if isinstance(message, BaseModel)
        else message
    )
    validated = ServerMessageAdapter.validate_python(value)
    payload = json.dumps(
        validated.model_dump(by_alias=True, exclude_none=True),
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(payload.encode("utf-8")) > MAX_FRAME_BYTES:
        raise ValueError("Protocol JSON frame exceeds 1 MiB.")
    return payload


def chunk_neural_frames(
    *,
    message_type: Literal["neural_keyframe", "neural_delta"],
    session_id: str,
    episode_id: str,
    start_sequence: int,
    simulation_time: float,
    revision: int,
    updates: Sequence[NeuralUpdate | dict[str, Any]],
    base_revision: int | None = None,
) -> list[NeuralKeyframe | NeuralDelta]:
    """Build deterministic count- and byte-bounded neural wire chunks."""
    validated_updates = [
        update if isinstance(update, NeuralUpdate) else NeuralUpdate.model_validate(update)
        for update in updates
    ]
    parts = [
        validated_updates[index:index + MAX_NEURAL_UPDATES]
        for index in range(0, len(validated_updates), MAX_NEURAL_UPDATES)
    ] or [[]]

    while True:
        frames = [
            _neural_frame(
                message_type=message_type,
                session_id=session_id,
                episode_id=episode_id,
                sequence=start_sequence + chunk_index,
                simulation_time=simulation_time,
                base_revision=base_revision,
                revision=revision,
                chunk_index=chunk_index,
                chunk_count=len(parts),
                updates=part,
            )
            for chunk_index, part in enumerate(parts)
        ]
        oversized_index: int | None = None
        for index, frame in enumerate(frames):
            try:
                serialize_server_message(frame)
            except ValueError:
                if len(parts[index]) <= 1:
                    raise
                oversized_index = index
                break
        if oversized_index is None:
            return frames

        oversized = parts[oversized_index]
        midpoint = len(oversized) // 2
        parts[oversized_index:oversized_index + 1] = [
            oversized[:midpoint],
            oversized[midpoint:],
        ]


def _neural_frame(
    *,
    message_type: Literal["neural_keyframe", "neural_delta"],
    session_id: str,
    episode_id: str,
    sequence: int,
    simulation_time: float,
    revision: int,
    chunk_index: int,
    chunk_count: int,
    updates: list[NeuralUpdate],
    base_revision: int | None,
) -> NeuralKeyframe | NeuralDelta:
    value: dict[str, Any] = {
        "type": message_type,
        "version": 2,
        "sessionId": session_id,
        "episodeId": episode_id,
        "sequence": sequence,
        "simulationTime": simulation_time,
        "revision": revision,
        "chunkIndex": chunk_index,
        "chunkCount": chunk_count,
        "updates": [update.model_dump(by_alias=True) for update in updates],
    }
    if message_type == "neural_delta":
        if base_revision is None:
            raise ValueError("baseRevision is required for neural deltas")
        value["baseRevision"] = base_revision
        return NeuralDelta.model_validate(value)
    if base_revision is not None:
        raise ValueError("baseRevision is valid only for neural deltas")
    return NeuralKeyframe.model_validate(value)
