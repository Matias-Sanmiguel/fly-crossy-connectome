from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fly_crossy.checkpoint import validate_checkpoint
from fly_crossy.connectome import load_reduced_graph_variant
from fly_crossy.env import WORLD_VERSION
from fly_crossy.models import (
    NestedPopulationFixedGraphPolicy,
    PopulationFixedGraphPolicy,
)
from fly_crossy.schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE
from fly_crossy.train import TrainingConfig, train


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _graph_edge_map(graph) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for edge, weight in zip(
        graph.edge_index.T.tolist(),
        graph.edge_weight.tolist(),
        strict=True,
    ):
        source_id = int(graph.body_ids[int(edge[0])])
        target_id = int(graph.body_ids[int(edge[1])])
        result[(source_id, target_id)] = float(weight)
    return result


def _core_edge_map(
    model: NestedPopulationFixedGraphPolicy,
) -> dict[tuple[int, int], float]:
    adjacency = model.core_adjacency.coalesce()
    indices = adjacency.indices()
    values = adjacency.values()
    result: dict[tuple[int, int], float] = {}
    for edge in range(values.numel()):
        target = int(indices[0, edge])
        source = int(indices[1, edge])
        source_id = int(model.graph.body_ids[source])
        target_id = int(model.graph.body_ids[target])
        result[(source_id, target_id)] = float(values[edge])
    return result


def test_nested_1k_preserves_every_80_core_weight_exactly() -> None:
    core = load_reduced_graph_variant("80")
    capacity = load_reduced_graph_variant("1k")
    model = NestedPopulationFixedGraphPolicy(
        capacity, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )

    expected = _graph_edge_map(core)
    actual = _core_edge_map(model)

    assert actual.keys() == expected.keys()
    for edge in expected:
        assert actual[edge] == pytest.approx(expected[edge], abs=1e-8)


def test_nested_expansion_is_empty_for_80_and_present_for_1k() -> None:
    core = load_reduced_graph_variant("80")
    capacity = load_reduced_graph_variant("1k")

    model_80 = NestedPopulationFixedGraphPolicy(
        core, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    model_1k = NestedPopulationFixedGraphPolicy(
        capacity, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )

    assert model_80.core_edge_count == core.edge_count
    assert model_80.expansion_edge_count == 0
    assert model_1k.core_edge_count == core.edge_count
    assert model_1k.expansion_edge_count > 0


def test_nested_parameter_count_is_capacity_invariant() -> None:
    core = load_reduced_graph_variant("80")
    capacity = load_reduced_graph_variant("1k")

    model_80 = NestedPopulationFixedGraphPolicy(
        core, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    model_1k = NestedPopulationFixedGraphPolicy(
        capacity, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )

    assert _parameter_count(model_80) == _parameter_count(model_1k) == 15849
    assert float(model_80.expansion_gain.detach()) == pytest.approx(0.1)
    assert float(model_1k.expansion_gain.detach()) == pytest.approx(0.1)


def test_nested_80_matches_population_80_with_empty_expansion() -> None:
    core = load_reduced_graph_variant("80")

    torch.manual_seed(20260920)
    population = PopulationFixedGraphPolicy(
        core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    torch.manual_seed(20260920)
    nested = NestedPopulationFixedGraphPolicy(
        core, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )

    observation = torch.randn(4, OBSERVATION_INPUT_SIZE)
    hidden = torch.randn(4, core.node_count)

    population_logits, population_value, population_activity = population(
        observation, hidden
    )
    nested_logits, nested_value, nested_activity = nested(
        observation, hidden
    )

    assert torch.allclose(nested_activity, population_activity, atol=1e-6)
    assert torch.allclose(nested_logits, population_logits, atol=1e-6)
    assert torch.allclose(nested_value, population_value, atol=1e-6)


def test_nested_tiny_runs_round_trip_for_80_and_1k(tmp_path: Path) -> None:
    for variant, expected_nodes in (("80", 80), ("1k", 1000)):
        output = tmp_path / f"nested-{variant}"
        train(
            TrainingConfig(
                controller="connectome",
                seed=f"nested-{variant}-tiny",
                steps=4,
                envs=1,
                learning_rate=3e-4,
                output=output,
                device="cpu",
                connectome_graph=variant,
                connectome_interface="nested",
            )
        )

        checkpoint = torch.load(
            output / "checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        validated = validate_checkpoint(checkpoint)
        metadata = json.loads(
            (output / "metadata.json").read_text(encoding="utf-8")
        )
        policy = json.loads(
            (output / "policy.json").read_text(encoding="utf-8")
        )

        assert validated.connectome_interface == "nested"
        assert validated.graph is not None
        assert validated.core_graph is not None
        assert validated.graph.node_count == expected_nodes
        assert validated.core_graph.node_count == 80
        assert metadata["environmentVersion"] == WORLD_VERSION
        assert metadata["configuration"]["connectome_interface"] == "nested"
        assert metadata["configuration"]["graphNodes"] == expected_nodes
        assert policy["network"]["kind"] == "fixed-graph"
        assert policy["network"]["interfaceMode"] == "nested"
