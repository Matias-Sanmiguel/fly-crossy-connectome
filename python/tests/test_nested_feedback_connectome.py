from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fly_crossy.checkpoint import validate_checkpoint
from fly_crossy.connectome import load_reduced_graph_variant
from fly_crossy.env import WORLD_VERSION
from fly_crossy.models import (
    FeedbackNestedPopulationFixedGraphPolicy,
    NestedPopulationFixedGraphPolicy,
)
from fly_crossy.schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE
from fly_crossy.train import TrainingConfig, train


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_feedback_nested_parameter_count_and_initial_gate_are_capacity_invariant() -> None:
    core = load_reduced_graph_variant("80")
    capacity = load_reduced_graph_variant("1k")
    model_80 = FeedbackNestedPopulationFixedGraphPolicy(
        core, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    model_1k = FeedbackNestedPopulationFixedGraphPolicy(
        capacity, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    assert _parameter_count(model_80) == _parameter_count(model_1k) == 16649
    assert float(model_80.feedback_strength().detach()) == pytest.approx(0.1)
    assert float(model_1k.feedback_strength().detach()) == pytest.approx(0.1)


def test_feedback_nested_splits_branch_and_feedback_edges() -> None:
    core = load_reduced_graph_variant("80")
    capacity = load_reduced_graph_variant("1k")
    model_80 = FeedbackNestedPopulationFixedGraphPolicy(
        core, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    model_1k = FeedbackNestedPopulationFixedGraphPolicy(
        capacity, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    assert model_80.branch_edge_count == 0
    assert model_80.feedback_edge_count == 0
    assert model_1k.branch_edge_count > 0
    assert model_1k.feedback_edge_count > 0
    assert model_1k.branch_edge_count + model_1k.feedback_edge_count == model_1k.expansion_edge_count


def test_feedback_nested_80_matches_old_nested_80() -> None:
    core = load_reduced_graph_variant("80")
    torch.manual_seed(20260921)
    old = NestedPopulationFixedGraphPolicy(
        core, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    torch.manual_seed(20260921)
    feedback = FeedbackNestedPopulationFixedGraphPolicy(
        core, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    observation = torch.randn(4, OBSERVATION_INPUT_SIZE)
    hidden = torch.randn(4, core.node_count)
    old_logits, old_value, old_activity = old(observation, hidden)
    new_logits, new_value, new_activity = feedback(observation, hidden)
    assert torch.allclose(new_activity, old_activity, atol=1e-6)
    assert torch.allclose(new_logits, old_logits, atol=1e-6)
    assert torch.allclose(new_value, old_value, atol=1e-6)


def test_feedback_nested_branch_active_with_near_zero_feedback() -> None:
    core = load_reduced_graph_variant("80")
    capacity = load_reduced_graph_variant("1k")
    model = FeedbackNestedPopulationFixedGraphPolicy(
        capacity, core, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    with torch.no_grad():
        model.feedback_logit.fill_(-20.0)
        model.sensory.weight.fill_(0.01)
        model.recurrent_gain.fill_(1.0)
        model.time_constant.fill_(1.0)
    observation = torch.ones(1, OBSERVATION_INPUT_SIZE)
    hidden = torch.zeros(1, capacity.node_count)
    _, _, first = model(observation, hidden)
    _, _, second = model(observation, first)
    core_ids = set(int(x) for x in core.body_ids)
    added_indices = [
        index for index, body_id in enumerate(capacity.body_ids)
        if int(body_id) not in core_ids
    ]
    assert torch.count_nonzero(second[:, added_indices]).item() > 0
    assert 0.0 < float(model.feedback_strength().detach()) < 1e-6


def test_feedback_nested_tiny_runs_round_trip_for_80_and_1k(tmp_path: Path) -> None:
    for variant, expected_nodes in (("80", 80), ("1k", 1000)):
        output = tmp_path / f"nested-feedback-{variant}"
        train(
            TrainingConfig(
                controller="connectome",
                seed=f"nested-feedback-{variant}-tiny",
                steps=4,
                envs=1,
                learning_rate=3e-4,
                output=output,
                device="cpu",
                connectome_graph=variant,
                connectome_interface="nested-feedback",
            )
        )
        checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        validated = validate_checkpoint(checkpoint)
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        policy = json.loads((output / "policy.json").read_text(encoding="utf-8"))
        assert validated.connectome_interface == "nested-feedback"
        assert validated.graph is not None
        assert validated.core_graph is not None
        assert validated.graph.node_count == expected_nodes
        assert validated.core_graph.node_count == 80
        assert "feedback_logit" in validated.state_dict
        assert metadata["environmentVersion"] == WORLD_VERSION
        assert metadata["configuration"]["connectome_interface"] == "nested-feedback"
        assert policy["network"]["interfaceMode"] == "nested-feedback"
