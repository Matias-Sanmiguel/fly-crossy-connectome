from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fly_crossy.checkpoint import validate_checkpoint
from fly_crossy.connectome import load_reduced_graph_variant
from fly_crossy.env import PROGRESS_REWARD, STEP_COST, WAIT_COST, create_game, step_game
from fly_crossy.models import TrafficAwarePopulationPolicy
from fly_crossy.schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE, Action
from fly_crossy.train import (
    CONTROLLER_V2_READOUT_COUNT,
    CONTROLLER_V2_SENSORY_COUNT,
    TrainingConfig,
    _build_controller_v2_graph,
    train,
)


def test_controller_v2_interface_is_128_by_128_and_topology_deterministic() -> None:
    base = load_reduced_graph_variant("1k")
    first = _build_controller_v2_graph(base)
    second = _build_controller_v2_graph(base)
    assert len(first.sensory_body_ids) == CONTROLLER_V2_SENSORY_COUNT == 128
    assert len(first.readout_body_ids) == CONTROLLER_V2_READOUT_COUNT == 128
    assert first.sensory_body_ids.tolist() == second.sensory_body_ids.tolist()
    assert first.readout_body_ids.tolist() == second.readout_body_ids.tolist()
    assert set(map(int, first.sensory_body_ids)).isdisjoint(set(map(int, first.readout_body_ids)))
    assert first.edge_index.tolist() == base.edge_index.tolist()


def test_controller_v2_learned_risk_head_can_override_forward_at_inference() -> None:
    graph = _build_controller_v2_graph(load_reduced_graph_variant("1k"))
    model = TrafficAwarePopulationPolicy(graph, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER))
    with torch.no_grad():
        model.sensory.weight.zero_()
        model.actor.weight.zero_()
        model.actor.bias.zero_()
        model.risk_head.weight.zero_()
        model.risk_head.bias.zero_()
        model.route_head.weight.zero_()
        model.route_head.bias.zero_()
        model.risk_head.bias[0] = 10.0
        model.risk_head.bias[2] = -10.0
        model.route_head.bias[2] = 10.0
    observation = torch.zeros(1, OBSERVATION_INPUT_SIZE)
    hidden = torch.zeros(1, graph.node_count)
    logits, _, _ = model(observation, hidden)
    assert int(torch.argmax(logits, dim=1).item()) == 2
    assert logits[0, 2] > logits[0, 0]


def test_reward_v5_makes_lateral_navigation_cheaper_than_waiting() -> None:
    state = create_game("reward-v5-controller-v2")
    forward = step_game(state, Action.FORWARD)
    lateral = step_game(state, Action.LEFT)
    waiting = step_game(state, Action.WAIT)
    assert forward.reward == pytest.approx(PROGRESS_REWARD + STEP_COST)
    assert lateral.reward == pytest.approx(STEP_COST)
    assert waiting.reward == pytest.approx(STEP_COST + WAIT_COST)
    assert lateral.reward > waiting.reward


def test_controller_v2_tiny_training_exports_inference_safety_heads(tmp_path: Path) -> None:
    output = tmp_path / "controller-v2"
    train(TrainingConfig(
        controller="connectome",
        seed="controller-v2-tiny",
        steps=8,
        envs=1,
        learning_rate=3e-4,
        output=output,
        device="cpu",
        connectome_graph="1k",
        connectome_interface="controller-v2",
    ))
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    validated = validate_checkpoint(checkpoint)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    policy = json.loads((output / "policy.json").read_text(encoding="utf-8"))
    assert validated.connectome_interface == "controller-v2"
    assert validated.graph is not None
    assert len(validated.graph.sensory_body_ids) == 128
    assert len(validated.graph.readout_body_ids) == 128
    assert "risk_head.weight" in validated.state_dict
    assert "route_head.weight" in validated.state_dict
    first = metrics["trainingCurve"][0]
    assert "teacherRiskLoss" in first
    assert "teacherRouteLoss" in first
    assert "teacherRiskAccuracy" in first
    network = policy["network"]
    assert network["interfaceMode"] == "controller-v2"
    assert len(network["riskWeights"]) == 5 * 1000
    assert len(network["routeWeights"]) == 5 * 1000
    assert network["safetyGain"] == pytest.approx(4.0)
    assert network["routeGain"] == pytest.approx(0.75)
