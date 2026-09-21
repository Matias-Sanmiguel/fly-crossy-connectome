from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fly_crossy.checkpoint import validate_checkpoint
from fly_crossy.connectome import load_reduced_graph_variant
from fly_crossy.models import (
    GatedNestedPopulationFixedGraphPolicy,
    NestedPopulationFixedGraphPolicy,
)
from fly_crossy.schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE
from fly_crossy.train import TrainingConfig, train


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_gated_nested_parameter_count_and_initial_strength_are_capacity_invariant() -> None:
    core = load_reduced_graph_variant("80")
    capacity = load_reduced_graph_variant("1k")

    model_80 = GatedNestedPopulationFixedGraphPolicy(
        core, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    model_1k = GatedNestedPopulationFixedGraphPolicy(
        capacity, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )

    assert _parameter_count(model_80) == _parameter_count(model_1k) == 11945
    assert float(model_80.expansion_strength().detach()) == pytest.approx(0.1)
    assert float(model_1k.expansion_strength().detach()) == pytest.approx(0.1)


def test_gated_nested_gate_has_no_hard_zero_dead_zone() -> None:
    core = load_reduced_graph_variant("80")
    capacity = load_reduced_graph_variant("1k")
    model = GatedNestedPopulationFixedGraphPolicy(
        capacity, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )

    with torch.no_grad():
        model.expansion_logit.fill_(-10.0)

    strength = model.expansion_strength()
    assert 0.0 < float(strength.detach()) < 0.001

    strength.backward()
    assert model.expansion_logit.grad is not None
    assert float(model.expansion_logit.grad) > 0.0


def test_gated_nested_80_matches_old_nested_80() -> None:
    core = load_reduced_graph_variant("80")

    torch.manual_seed(20260920)
    old = NestedPopulationFixedGraphPolicy(
        core, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    torch.manual_seed(20260920)
    gated = GatedNestedPopulationFixedGraphPolicy(
        core, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )

    observation = torch.randn(4, OBSERVATION_INPUT_SIZE)
    hidden = torch.randn(4, core.node_count)

    old_logits, old_value, old_activity = old(observation, hidden)
    gated_logits, gated_value, gated_activity = gated(observation, hidden)

    assert torch.allclose(gated_activity, old_activity, atol=1e-6)
    assert torch.allclose(gated_logits, old_logits, atol=1e-6)
    assert torch.allclose(gated_value, old_value, atol=1e-6)


def test_gated_nested_tiny_runs_round_trip_for_80_and_1k(tmp_path: Path) -> None:
    for variant, expected_nodes in (("80", 80), ("1k", 1000)):
        output = tmp_path / f"nested-gated-{variant}"
        train(
            TrainingConfig(
                controller="connectome",
                seed=f"nested-gated-{variant}-tiny",
                steps=4,
                envs=1,
                learning_rate=3e-4,
                output=output,
                device="cpu",
                connectome_graph=variant,
                connectome_interface="nested-gated",
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

        assert validated.connectome_interface == "nested-gated"
        assert validated.graph is not None
        assert validated.core_graph is not None
        assert validated.graph.node_count == expected_nodes
        assert validated.core_graph.node_count == 80
        assert "expansion_logit" in validated.state_dict
        assert "expansion_gain" not in validated.state_dict
        assert metadata["environmentVersion"] == 7
        assert metadata["configuration"]["connectome_interface"] == "nested-gated"
        assert policy["network"]["kind"] == "fixed-graph"
        assert policy["network"]["interfaceMode"] == "nested-gated"
