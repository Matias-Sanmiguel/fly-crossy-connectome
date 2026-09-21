from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .checkpoint import validate_checkpoint
from .connectome import ReducedGraphArtifact
from .models import (
    DensePolicy,
    FeedbackNestedPopulationFixedGraphPolicy,
    FixedGraphPolicy,
    GatedNestedPopulationFixedGraphPolicy,
    NestedPopulationFixedGraphPolicy,
    PopulationFixedGraphPolicy,
)
from .schema import ACTION_ORDER, OBSERVATION_VERSION


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
    observation_size: int,
    hidden_size: int,
    actions: int,
    state_dict: Mapping[str, Tensor],
) -> DensePolicy:
    try:
        model = DensePolicy(
            observation_size=observation_size,
            hidden_size=hidden_size,
            actions=actions,
        )
        model.load_state_dict(state_dict)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Checkpoint contains an incompatible dense policy.") from error
    return model


def _load_fixed_graph_policy(
    graph: ReducedGraphArtifact,
    observation_size: int,
    actions: int,
    state_dict: Mapping[str, Tensor],
) -> FixedGraphPolicy:
    try:
        model = FixedGraphPolicy(
            graph=graph,
            observation_size=observation_size,
            actions=actions,
        )
        model.load_state_dict(state_dict)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Checkpoint contains an incompatible fixed graph policy.") from error
    return model


def _load_population_fixed_graph_policy(
    graph: ReducedGraphArtifact,
    observation_size: int,
    actions: int,
    state_dict: Mapping[str, Tensor],
) -> PopulationFixedGraphPolicy:
    try:
        model = PopulationFixedGraphPolicy(
            graph=graph,
            observation_size=observation_size,
            actions=actions,
        )
        model.load_state_dict(state_dict)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            "Checkpoint contains an incompatible population fixed graph policy."
        ) from error
    return model



def _load_nested_fixed_graph_policy(
    graph: ReducedGraphArtifact,
    core_graph: ReducedGraphArtifact,
    observation_size: int,
    actions: int,
    state_dict: Mapping[str, Tensor],
) -> NestedPopulationFixedGraphPolicy:
    try:
        model = NestedPopulationFixedGraphPolicy(
            graph=graph,
            core_graph=core_graph,
            observation_size=observation_size,
            actions=actions,
        )
        model.load_state_dict(state_dict)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            "Checkpoint contains an incompatible nested fixed graph policy."
        ) from error
    return model


def _load_gated_nested_fixed_graph_policy(
    graph: ReducedGraphArtifact,
    core_graph: ReducedGraphArtifact,
    observation_size: int,
    actions: int,
    state_dict: Mapping[str, Tensor],
) -> GatedNestedPopulationFixedGraphPolicy:
    try:
        model = GatedNestedPopulationFixedGraphPolicy(
            graph=graph,
            core_graph=core_graph,
            observation_size=observation_size,
            actions=actions,
        )
        model.load_state_dict(state_dict)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            "Checkpoint contains an incompatible gated nested fixed graph policy."
        ) from error
    return model


def _load_feedback_nested_fixed_graph_policy(
    graph: ReducedGraphArtifact,
    core_graph: ReducedGraphArtifact,
    observation_size: int,
    actions: int,
    state_dict: Mapping[str, Tensor],
) -> FeedbackNestedPopulationFixedGraphPolicy:
    try:
        model = FeedbackNestedPopulationFixedGraphPolicy(
            graph=graph,
            core_graph=core_graph,
            observation_size=observation_size,
            actions=actions,
        )
        model.load_state_dict(state_dict)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            "Checkpoint contains an incompatible feedback nested fixed graph policy."
        ) from error
    return model

def export_policy(checkpoint: str | Path, output: str | Path) -> dict[str, Any]:
    """Export a conventional or fixed-graph actor checkpoint for browser inference."""
    checkpoint_path = Path(checkpoint)
    output_path = Path(output)
    try:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Checkpoint could not be loaded: {checkpoint_path}") from error
    checkpoint_metadata = validate_checkpoint(saved)
    controller = checkpoint_metadata.controller
    state_dict = checkpoint_metadata.state_dict

    if controller == "connectome":
        graph = checkpoint_metadata.graph
        if graph is None:
            raise ValueError("Checkpoint connectome graph is incompatible.")
        if checkpoint_metadata.connectome_interface == "nested-feedback":
            core_graph = checkpoint_metadata.core_graph
            if core_graph is None:
                raise ValueError("Checkpoint nested core graph is incompatible.")
            model = _load_feedback_nested_fixed_graph_policy(
                graph,
                core_graph,
                checkpoint_metadata.observation_size,
                checkpoint_metadata.actions,
                state_dict,
            )
        elif checkpoint_metadata.connectome_interface == "nested-gated":
            core_graph = checkpoint_metadata.core_graph
            if core_graph is None:
                raise ValueError("Checkpoint nested core graph is incompatible.")
            model = _load_gated_nested_fixed_graph_policy(
                graph,
                core_graph,
                checkpoint_metadata.observation_size,
                checkpoint_metadata.actions,
                state_dict,
            )
        elif checkpoint_metadata.connectome_interface == "nested":
            core_graph = checkpoint_metadata.core_graph
            if core_graph is None:
                raise ValueError("Checkpoint nested core graph is incompatible.")
            model = _load_nested_fixed_graph_policy(
                graph,
                core_graph,
                checkpoint_metadata.observation_size,
                checkpoint_metadata.actions,
                state_dict,
            )
        elif checkpoint_metadata.connectome_interface == "population":
            model = _load_population_fixed_graph_policy(
                graph,
                checkpoint_metadata.observation_size,
                checkpoint_metadata.actions,
                state_dict,
            )
        else:
            model = _load_fixed_graph_policy(
                graph,
                checkpoint_metadata.observation_size,
                checkpoint_metadata.actions,
                state_dict,
            )
    else:
        hidden_size = checkpoint_metadata.hidden_size
        if hidden_size is None:
            raise ValueError("Checkpoint dense model is missing its hidden size.")
        model = _load_dense_policy(
            checkpoint_metadata.observation_size,
            hidden_size,
            checkpoint_metadata.actions,
            state_dict,
        )
    if model.actor.out_features != len(ACTION_ORDER):
        raise ValueError("Checkpoint actor does not match the canonical action order.")

    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if isinstance(model, (FixedGraphPolicy, PopulationFixedGraphPolicy, NestedPopulationFixedGraphPolicy, GatedNestedPopulationFixedGraphPolicy)):
        graph = model.graph
        source: dict[str, Any] = {
            "kind": "predicted",
            "name": "Reduced MaleCNS fixed-graph PPO controller",
            "normalization": (
                "ObservationV2 uses the declared 370-value encoding with explicit "
                "blocker cells and hazard-speed scaling; raw tanh node activity is "
                "mapped as (activity + 1) / 2 for atlas display"
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
        if isinstance(
            model,
            (
                PopulationFixedGraphPolicy,
                NestedPopulationFixedGraphPolicy,
                GatedNestedPopulationFixedGraphPolicy,
                FeedbackNestedPopulationFixedGraphPolicy,
            ),
        ):
            sensory_weights = torch.zeros(
                graph.node_count,
                model.sensory.in_features,
                dtype=model.sensory.weight.dtype,
            )
            sensory_weights.index_copy_(
                0,
                model.sensory_indices.detach().cpu(),
                model.sensory.weight.detach().cpu(),
            )
            actor_weights = torch.zeros(
                model.actor.out_features,
                graph.node_count,
                dtype=model.actor.weight.dtype,
            )
            actor_weights.index_copy_(
                1,
                model.readout_indices.detach().cpu(),
                model.actor.weight.detach().cpu(),
            )
            if isinstance(model, FeedbackNestedPopulationFixedGraphPolicy):
                interface_mode = "nested-feedback"
            elif isinstance(model, GatedNestedPopulationFixedGraphPolicy):
                interface_mode = "nested-gated"
            elif isinstance(model, NestedPopulationFixedGraphPolicy):
                interface_mode = "nested"
            else:
                interface_mode = "population"
        else:
            sensory_weights = model.sensory.weight
            actor_weights = model.actor.weight
            interface_mode = "legacy"

        if isinstance(
            model,
            (
                NestedPopulationFixedGraphPolicy,
                GatedNestedPopulationFixedGraphPolicy,
                FeedbackNestedPopulationFixedGraphPolicy,
            ),
        ):
            recurrent_indices, recurrent_values = model.effective_recurrent_edges()
            recurrent_source = [
                int(item) for item in recurrent_indices[0].detach().cpu().tolist()
            ]
            recurrent_target = [
                int(item) for item in recurrent_indices[1].detach().cpu().tolist()
            ]
            recurrent_weights = [
                float(item) for item in recurrent_values.detach().cpu().tolist()
            ]
            recurrent_gain = 1.0
        else:
            recurrent_source = [int(item) for item in graph.edge_index[0]]
            recurrent_target = [int(item) for item in graph.edge_index[1]]
            recurrent_weights = [float(item) for item in graph.edge_weight]

        network: dict[str, Any] = {
            "kind": "fixed-graph",
            "interfaceMode": interface_mode,
            "inputSize": model.sensory.in_features,
            "bodyIds": [int(item) for item in graph.body_ids],
            "activation": "tanh",
            "sensoryWeights": _finite_row_major(
                sensory_weights, "sensory weights"
            ),
            "recurrentSource": recurrent_source,
            "recurrentTarget": recurrent_target,
            "recurrentWeights": recurrent_weights,
            "recurrentGain": recurrent_gain,
            "timeConstant": time_constant,
            "actorWeights": _finite_row_major(actor_weights, "actor weights"),
            "actorBias": _finite_row_major(model.actor.bias, "actor bias"),
        }
        activity_body_ids = [int(item) for item in graph.body_ids]
    else:
        source = {
            "kind": "predicted",
            "name": "Conventional PPO baseline",
            "normalization": (
                "ObservationV2 cells divided by 8; motion uses hazard-specific "
                "speed scaling; support, previous-action one-hot, and edge distance unchanged"
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
        "observationVersion": OBSERVATION_VERSION,
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
