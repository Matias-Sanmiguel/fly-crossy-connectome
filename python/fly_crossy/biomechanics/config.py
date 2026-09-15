"""Validated SI-unit configuration for the biomechanical keyboard."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fly_crossy.protocol import KeyName


KEY_NAMES: tuple[KeyName, ...] = (
    "W",
    "A",
    "S",
    "D",
    "SPACE_LEFT",
    "SPACE_RIGHT",
)

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


def _to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


class ConfigModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        extra="forbid",
        frozen=True,
        validate_by_alias=True,
        validate_by_name=False,
    )


class KeyConfig(ConfigModel):
    """One key center in public SI metres."""

    center: tuple[FiniteFloat, FiniteFloat, FiniteFloat]


class KeyboardConfig(ConfigModel):
    """Complete keyboard physics contract; every dimensional value is SI."""

    schema_version: Literal[1]
    physics_hz: PositiveInt
    motor_hz: PositiveInt
    decision_timeout_seconds: PositiveFloat
    debounce_seconds: PositiveFloat
    minimum_travel_meters: PositiveFloat
    release_travel_meters: PositiveFloat
    minimum_force_newtons: PositiveFloat
    maximum_force_newtons: PositiveFloat
    key_travel_meters: PositiveFloat
    key_half_extents_meters: tuple[PositiveFloat, PositiveFloat, PositiveFloat]
    key_mass_kilograms: PositiveFloat
    return_stiffness_newtons_per_meter: PositiveFloat
    return_damping_newton_seconds_per_meter: PositiveFloat
    keys: dict[KeyName, KeyConfig]

    @classmethod
    def load(cls, path: str | Path) -> Self:
        try:
            payload = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"unable to read keyboard config: {exc}") from exc
        return cls.model_validate_json(payload)

    @model_validator(mode="after")
    def validate_physics_contract(self) -> Self:
        if set(self.keys) != set(KEY_NAMES) or len(self.keys) != len(KEY_NAMES):
            raise ValueError("keys must contain exactly W/A/S/D/SPACE_LEFT/SPACE_RIGHT")
        centers = tuple(key.center for key in self.keys.values())
        if len(set(centers)) != len(centers):
            raise ValueError("key centers must be unique")
        if self.release_travel_meters >= self.minimum_travel_meters:
            raise ValueError("releaseTravelMeters must be below minimumTravelMeters")
        if self.minimum_travel_meters >= self.key_travel_meters:
            raise ValueError("minimumTravelMeters must be below keyTravelMeters")
        if self.maximum_force_newtons <= self.minimum_force_newtons:
            raise ValueError("maximumForceNewtons must exceed minimumForceNewtons")
        if self.debounce_seconds >= self.decision_timeout_seconds:
            raise ValueError("debounceSeconds must be below decisionTimeoutSeconds")
        if self.physics_hz % self.motor_hz != 0:
            raise ValueError("physicsHz must be an integer multiple of motorHz")
        return self


__all__ = ["KEY_NAMES", "KeyConfig", "KeyboardConfig"]
