from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .connectome import ReducedGraphArtifact
from .models import DensePolicy, FixedGraphPolicy
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


def _finite_scalar(tensor: Tensor, label: str) -> float:
    value = float(tensor.detach().cpu())
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"{label} must be finite.")
    return value


def _load_dense_policy(
    model_config: dict[str, object], state_dict: dict[str, object]
) -> DensePolicy:
    try:
        model = DensePolicy(
            observation_size=int(model_config["observation_size"]),
            hidden_size=int(model_config["hidden_size"]),
            actions=int(model_config["actions"]),
        )
        model.load_state_dict(state_dict)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Checkpoint contains an incompatible dense policy.") from error
    return model


def _load_fixed_graph_policy(
    model_config: dict[str, object], state_dict: dict[str, object]
) -> FixedGraphPolicy:
    graph_value = model_config.get("graph")
    if not isinstance(graph_value, dict):
        raise ValueError("Connectome checkpoint is missing its reduced graph artifact.")
    try:
        graph = ReducedGraphArtifact.from_checkpoint(graph_value)
        model = FixedGraphPolicy(
            graph=graph,
            observation_size=int(model_config["observation_size"]),
            actions=int(model_config["actions"]),
        )
        model.load_state_dict(state_dict)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Checkpoint contains an incompatible fixed graph policy.") from error
    return model


def export_policy(checkpoint: str | Path, output: str | Path) -> dict[str, Any]:
    """Export a conventional or fixed-graph actor checkpoint for browser inference."""
    checkpoint_path = Path(checkpoint)
    output_path = Path(output)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    controller = saved.get("controller") if isinstance(saved, dict) else None
    if controller not in ("conventional", "connectome"):
        raise ValueError("Checkpoint controller must be conventional or connectome.")
    model_config = saved.get("model")
    state_dict = saved.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing model configuration or weights.")

    model = (
        _load_fixed_graph_policy(model_config, state_dict)
        if controller == "connectome"
        else _load_dense_policy(model_config, state_dict)
    )
    if model.actor.out_features != len(ACTION_ORDER):
        raise ValueError("Checkpoint actor does not match the canonical action order.")

    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if isinstance(model, FixedGraphPolicy):
        graph = model.graph
        source: dict[str, Any] = {
            "kind": "predicted",
            "name": "Reduced MaleCNS fixed-graph PPO controller",
            "normalization": (
                "ObservationV1 uses its declared 370-value encoding; raw tanh node "
                "activity is mapped as (activity + 1) / 2 for atlas display"
            ),
            "checkpointHash": checkpoint_hash,
            "datasetVersion": graph.dataset_version,
            "sourceUrl": graph.source_url,
            "license": graph.license,
            "selectionRule": graph.selection_rule,
            "graphSourceHash": graph.source_sha256,
            "graphArtifactHash": graph.artifact_sha256,
        }
        recurrent_gain = max(
            0.0, _finite_scalar(model.recurrent_gain, "Recurrent gain")
        )
        time_constant = min(
            1.0, max(1e-4, _finite_scalar(model.time_constant, "Time constant"))
        )
        network: dict[str, Any] = {
            "kind": "fixed-graph",
            "inputSize": model.sensory.in_features,
            "bodyIds": [int(item) for item in graph.body_ids],
            "activation": "tanh",
            "sensoryWeights": _finite_row_major(
                model.sensory.weight, "sensory weights"
            ),
            "recurrentSource": [int(item) for item in graph.edge_index[0]],
            "recurrentTarget": [int(item) for item in graph.edge_index[1]],
            "recurrentWeights": [float(item) for item in graph.edge_weight],
            "recurrentGain": recurrent_gain,
            "timeConstant": time_constant,
            "actorWeights": _finite_row_major(model.actor.weight, "actor weights"),
            "actorBias": _finite_row_major(model.actor.bias, "actor bias"),
        }
        activity_body_ids = [int(item) for item in graph.body_ids]
    else:
        source = {
            "kind": "predicted",
            "name": "Conventional PPO baseline",
            "normalization": (
                "ObservationV1 cells divided by 7; motion, support, previous-action "
                "one-hot, and edge distance unchanged"
            ),
            "checkpointHash": checkpoint_hash,
        }
        network = {
            "kind": "dense",
            "inputSize": model.hidden_1.in_features,
            "layers": [
                _export_layer("hidden_1", model.hidden_1, "tanh"),
                _export_layer("hidden_2", model.hidden_2, "tanh"),
                _export_layer("actor", model.actor, "linear"),
            ],
        }
        activity_body_ids = []

    payload: dict[str, Any] = {
        "version": 1,
        "observationVersion": 1,
        "actions": [action.value for action in ACTION_ORDER],
        "source": source,
        "network": network,
        "activityBodyIds": activity_body_ids,
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
