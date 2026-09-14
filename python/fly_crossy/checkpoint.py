from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from .env import WORLD_VERSION
from .schema import ACTION_ORDER


CHECKPOINT_FORMAT_VERSION = 1
SUPPORTED_CONTROLLERS = ("conventional", "connectome")


@dataclass(frozen=True, slots=True)
class ValidatedCheckpoint:
    controller: str
    environment_version: int
    observation_size: int
    actions: int
    hidden_size: int | None
    model: Mapping[str, object]
    state_dict: Mapping[str, Tensor]
    training: dict[str, Any]


def _strict_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"Checkpoint {label} must be an integer.")
    return value


def _strict_positive_integer(value: object, label: str) -> int:
    result = _strict_integer(value, label)
    if result <= 0:
        raise ValueError(f"Checkpoint {label} must be positive.")
    return result


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Checkpoint {label} must be a non-empty string.")
    return value.strip()


def _finite_positive_number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Checkpoint {label} must be a positive finite number.")
    return float(value)


def _validate_state_dict(value: object) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError("Checkpoint model state must be a mapping.")
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise ValueError("Checkpoint model state must map string names to tensors.")
        if not tensor.is_floating_point():
            raise ValueError(
                f"Checkpoint tensor {name!r} must use a floating-point dtype."
            )
        try:
            finite = bool(torch.isfinite(tensor).all().item())
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"Checkpoint tensor {name!r} cannot be validated.") from error
        if not finite:
            raise ValueError(f"Checkpoint tensor {name!r} must contain only finite values.")
    return value


def validate_checkpoint(
    value: object,
    *,
    expected_controller: str | None = None,
    expected_environment_version: int = WORLD_VERSION,
    expected_observation_size: int | None = None,
) -> ValidatedCheckpoint:
    """Validate untrusted checkpoint metadata and weights before any coercion."""
    if not isinstance(value, Mapping):
        raise ValueError("Evaluation checkpoint must contain an object.")

    try:
        format_version = _strict_integer(value["format_version"], "format version")
        environment_version = _strict_integer(
            value["environment_version"], "environment version"
        )
        controller = _required_string(value["controller"], "controller")
        model = value["model"]
        training_value = value["training"]
        state_dict_value = value["model_state_dict"]
    except KeyError as error:
        raise ValueError("Checkpoint is missing required metadata.") from error

    if format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format version must be {CHECKPOINT_FORMAT_VERSION}."
        )
    if environment_version != expected_environment_version:
        raise ValueError(
            f"Checkpoint environment version {environment_version} does not match "
            f"evaluation environment version {expected_environment_version}."
        )
    if controller not in SUPPORTED_CONTROLLERS:
        raise ValueError("Checkpoint controller must be conventional or connectome.")
    if expected_controller is not None and controller != expected_controller:
        raise ValueError(
            f"Evaluation checkpoint must contain a {expected_controller} controller."
        )
    if not isinstance(model, Mapping):
        raise ValueError("Checkpoint model metadata must be a mapping.")
    if not isinstance(training_value, Mapping):
        raise ValueError("Checkpoint training metadata must be a mapping.")

    try:
        observation_size = _strict_positive_integer(
            model["observation_size"], "observation size"
        )
        actions = _strict_positive_integer(model["actions"], "action count")
        training_seed = _required_string(training_value["seed"], "training seed")
        training_steps = _strict_positive_integer(
            training_value["steps"], "training steps"
        )
        training_envs = _strict_positive_integer(
            training_value["envs"], "training environment count"
        )
        learning_rate = _finite_positive_number(
            training_value["learning_rate"], "training learning rate"
        )
        world_seeds_value = training_value["world_seeds"]
    except KeyError as error:
        raise ValueError("Checkpoint is missing required model or training metadata.") from error

    if actions != len(ACTION_ORDER):
        raise ValueError("Checkpoint action count does not match the canonical action order.")
    if (
        expected_observation_size is not None
        and observation_size != expected_observation_size
    ):
        raise ValueError("Checkpoint observation size is incompatible with the environment.")
    if training_steps % training_envs != 0:
        raise ValueError(
            "Checkpoint training steps must be divisible by its environment count."
        )
    if not isinstance(world_seeds_value, list) or not world_seeds_value:
        raise ValueError("Checkpoint concrete training world seeds must be a non-empty list.")
    world_seeds = [
        _required_string(item, "concrete training world seed")
        for item in world_seeds_value
    ]
    if len(world_seeds) != len(set(world_seeds)):
        raise ValueError("Checkpoint concrete training world seeds must be unique.")
    if any(not seed.startswith(f"{training_seed}:") for seed in world_seeds):
        raise ValueError(
            "Checkpoint concrete training world seeds do not match its root seed."
        )

    hidden_size: int | None = None
    if controller == "conventional":
        try:
            hidden_size = _strict_positive_integer(model["hidden_size"], "hidden size")
        except KeyError as error:
            raise ValueError("Checkpoint dense model is missing its hidden size.") from error
    else:
        graph = model.get("graph")
        if not isinstance(graph, Mapping):
            raise ValueError("Checkpoint connectome model is missing its graph mapping.")

    state_dict = _validate_state_dict(state_dict_value)
    training = dict(training_value)
    training.update(
        {
            "seed": training_seed,
            "steps": training_steps,
            "envs": training_envs,
            "learning_rate": learning_rate,
            "world_seeds": world_seeds,
        }
    )
    return ValidatedCheckpoint(
        controller=controller,
        environment_version=environment_version,
        observation_size=observation_size,
        actions=actions,
        hidden_size=hidden_size,
        model=model,
        state_dict=state_dict,
        training=training,
    )
