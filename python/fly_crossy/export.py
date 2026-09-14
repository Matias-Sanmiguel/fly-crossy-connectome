from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .models import DensePolicy
from .schema import ACTION_ORDER


def _finite_row_major(tensor: Tensor, label: str) -> list[float]:
    values = tensor.detach().cpu().contiguous()
    if not torch.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values.")
    return [float(value) for value in values.reshape(-1).tolist()]


def _export_layer(name: str, layer: nn.Linear, activation: str) -> dict[str, Any]:
    return {
        "name": name,
        "inputSize": layer.in_features,
        "outputSize": layer.out_features,
        "activation": activation,
        "weights": _finite_row_major(layer.weight, f"{name} weights"),
        "bias": _finite_row_major(layer.bias, f"{name} bias"),
    }


def export_policy(checkpoint: str | Path, output: str | Path) -> dict[str, Any]:
    """Export a conventional actor checkpoint to the validated browser schema."""
    checkpoint_path = Path(checkpoint)
    output_path = Path(output)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("controller") != "conventional":
        raise ValueError("Checkpoint is not a conventional controller checkpoint.")
    model_config = saved.get("model")
    state_dict = saved.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing model configuration or weights.")

    try:
        model = DensePolicy(
            observation_size=int(model_config["observation_size"]),
            hidden_size=int(model_config["hidden_size"]),
            actions=int(model_config["actions"]),
        )
        model.load_state_dict(state_dict)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Checkpoint contains an incompatible dense policy.") from error
    if model.actor.out_features != len(ACTION_ORDER):
        raise ValueError("Checkpoint actor does not match the canonical action order.")

    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    payload: dict[str, Any] = {
        "version": 1,
        "observationVersion": 1,
        "actions": [action.value for action in ACTION_ORDER],
        "source": {
            "kind": "predicted",
            "name": "Conventional PPO baseline",
            "normalization": (
                "ObservationV1 cells divided by 7; motion, support, previous-action "
                "one-hot, and edge distance unchanged"
            ),
            "checkpointHash": checkpoint_hash,
        },
        "network": {
            "kind": "dense",
            "inputSize": model.hidden_1.in_features,
            "layers": [
                _export_layer("hidden_1", model.hidden_1, "tanh"),
                _export_layer("hidden_2", model.hidden_2, "tanh"),
                _export_layer("actor", model.actor, "linear"),
            ],
        },
        "activityBodyIds": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _main() -> None:
    parser = argparse.ArgumentParser(description="Export a PPO checkpoint for the browser.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    export_policy(arguments.checkpoint, arguments.output)


if __name__ == "__main__":
    _main()

