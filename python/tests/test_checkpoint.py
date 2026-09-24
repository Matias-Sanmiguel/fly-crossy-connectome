from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import Tensor

import fly_crossy.evaluate as evaluate_module
import fly_crossy.export as export_module
from fly_crossy.checkpoint import validate_checkpoint
from fly_crossy.env import WORLD_VERSION
from fly_crossy.schema import OBSERVATION_INPUT_SIZE


RELEASE_ROOT = Path(__file__).resolve().parents[2] / "release" / "eval-v1" / "training"
CHECKPOINTS = {
    "conventional": RELEASE_ROOT / "conventional" / "checkpoint.pt",
    "connectome": RELEASE_ROOT / "connectome" / "checkpoint.pt",
}
RELEASE_ENVIRONMENT_VERSION = 3

STATE_KEYS = {
    "conventional": (
        "hidden_1.weight",
        "hidden_1.bias",
        "hidden_2.weight",
        "hidden_2.bias",
        "actor.weight",
        "actor.bias",
        "critic.weight",
        "critic.bias",
    ),
    "connectome": (
        "recurrent_gain",
        "time_constant",
        "sensory.weight",
        "actor.weight",
        "actor.bias",
        "critic.weight",
        "critic.bias",
    ),
}


def _upgrade_observation_boundary_for_current_schema(
    payload: dict[str, object],
    controller: str,
) -> None:
    # Adapt only the temporary test copy. The released checkpoint remains
    # historical evidence with its original 370-value observation boundary.
    model = payload["model"]
    state_dict = payload["model_state_dict"]
    old_size = int(model["observation_size"])
    if old_size == OBSERVATION_INPUT_SIZE:
        return

    model["observation_size"] = OBSERVATION_INPUT_SIZE
    input_key = (
        "hidden_1.weight"
        if controller == "conventional"
        else "sensory.weight"
    )
    old_weight = state_dict[input_key]
    upgraded = torch.zeros(
        old_weight.shape[0],
        OBSERVATION_INPUT_SIZE,
        dtype=old_weight.dtype,
    )
    copy_width = min(old_size, OBSERVATION_INPUT_SIZE)
    upgraded[:, :copy_width] = old_weight[:, :copy_width]
    state_dict[input_key] = upgraded


def _payload(
    controller: str,
    *,
    environment_version: int = WORLD_VERSION,
) -> dict[str, object]:
    payload = torch.load(CHECKPOINTS[controller], map_location="cpu", weights_only=True)
    payload["model"] = dict(payload["model"])
    payload["training"] = dict(payload["training"])
    payload["model_state_dict"] = dict(payload["model_state_dict"])
    payload["environment_version"] = environment_version
    if environment_version != RELEASE_ENVIRONMENT_VERSION:
        _upgrade_observation_boundary_for_current_schema(payload, controller)
    return payload


def _wrong_shape(tensor: Tensor) -> Tensor:
    if tensor.ndim == 0:
        shape = (1,)
    elif tensor.ndim == 1:
        shape = (tensor.shape[0] + 1,)
    else:
        shape = (tensor.shape[0] + 1, *tensor.shape[1:])
    return torch.zeros(shape, dtype=tensor.dtype)


@pytest.mark.parametrize("controller", ("conventional", "connectome"))
def test_valid_released_checkpoint_has_an_exact_preconstruction_schema(
    controller: str,
) -> None:
    validated = validate_checkpoint(
        _payload(
            controller,
            environment_version=RELEASE_ENVIRONMENT_VERSION,
        ),
        expected_controller=controller,
        expected_environment_version=RELEASE_ENVIRONMENT_VERSION,
    )

    assert validated.environment_version == RELEASE_ENVIRONMENT_VERSION
    assert set(validated.state_dict) == set(STATE_KEYS[controller])


@pytest.mark.parametrize(
    ("controller", "key"),
    [
        pytest.param(controller, key, id=f"{controller}-{key}")
        for controller, keys in STATE_KEYS.items()
        for key in keys
    ],
)
def test_validator_rejects_every_wrong_tensor_shape(
    controller: str, key: str
) -> None:
    payload = _payload(controller)
    state_dict = payload["model_state_dict"]
    state_dict[key] = _wrong_shape(state_dict[key])

    with pytest.raises(ValueError, match="shape"):
        validate_checkpoint(payload, expected_controller=controller)


@pytest.mark.parametrize("mutation", ("missing", "extra"))
@pytest.mark.parametrize("controller", ("conventional", "connectome"))
def test_validator_rejects_missing_and_extra_state_keys(
    controller: str, mutation: str
) -> None:
    payload = _payload(controller)
    state_dict = payload["model_state_dict"]
    if mutation == "missing":
        del state_dict["actor.bias"]
    else:
        state_dict["unexpected.weight"] = torch.zeros(1)

    with pytest.raises(ValueError, match="keys"):
        validate_checkpoint(payload, expected_controller=controller)


@pytest.mark.parametrize("field", ("observation_size", "actions", "hidden_size"))
def test_validator_rejects_absurd_allocation_metadata_before_model_construction(
    field: str,
) -> None:
    payload = _payload("conventional")
    payload["model"][field] = 10**12

    with pytest.raises(ValueError, match="action count|shape"):
        validate_checkpoint(payload, expected_controller="conventional")


@pytest.mark.parametrize(
    ("controller", "mutation"),
    [
        pytest.param(controller, mutation, id=f"{controller}-{mutation}")
        for controller in ("conventional", "connectome")
        for mutation in ("wrong-shape", "missing-key", "extra-key")
    ]
    + [pytest.param("conventional", "absurd-hidden", id="conventional-absurd-hidden")],
)
@pytest.mark.parametrize("consumer", ("export", "evaluate"))
def test_consumers_do_not_construct_models_for_malformed_checkpoint_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controller: str,
    mutation: str,
    consumer: str,
) -> None:
    payload = _payload(controller)
    if mutation == "wrong-shape":
        payload["model_state_dict"]["actor.bias"] = torch.zeros(6)
    elif mutation == "missing-key":
        del payload["model_state_dict"]["actor.bias"]
    elif mutation == "extra-key":
        payload["model_state_dict"]["unexpected.weight"] = torch.zeros(1)
    else:
        payload["model"]["hidden_size"] = 10**12
    checkpoint = tmp_path / f"{controller}.pt"
    torch.save(payload, checkpoint)
    constructor_calls: list[tuple[object, ...]] = []

    def constructor_sentinel(*args: object, **kwargs: object) -> None:
        constructor_calls.append((*args, kwargs))
        raise AssertionError("model construction must not run")

    class_name = "DensePolicy" if controller == "conventional" else "FixedGraphPolicy"
    module = export_module if consumer == "export" else evaluate_module
    monkeypatch.setattr(module, class_name, constructor_sentinel)

    with pytest.raises(ValueError, match="shape|keys"):
        if consumer == "export":
            export_module.export_policy(checkpoint, tmp_path / "policy.json")
        else:
            evaluate_module._load_checkpoint(
                checkpoint,
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                controller,
                ["smoke"],
                WORLD_VERSION,
            )

    assert constructor_calls == []
