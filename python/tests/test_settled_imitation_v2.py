from __future__ import annotations

import json
from pathlib import Path

import torch

from fly_crossy.checkpoint import validate_checkpoint
from fly_crossy.connectome import load_reduced_graph_variant
from fly_crossy.export import export_policy
from fly_crossy.models import PopulationFixedGraphPolicy, SettledPopulationFixedGraphPolicy
from fly_crossy.schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE
from fly_crossy.train import _build_controller_v2_graph


def test_two_updates_create_current_observation_gradient_to_readout() -> None:
    graph = _build_controller_v2_graph(load_reduced_graph_variant("1k"))
    baseline = PopulationFixedGraphPolicy(graph, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER))
    settled = SettledPopulationFixedGraphPolicy(graph, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER))
    settled.load_state_dict(baseline.state_dict())

    observation = torch.randn(1, OBSERVATION_INPUT_SIZE, requires_grad=True)
    hidden = torch.zeros(1, graph.node_count)
    baseline_logits, _, _ = baseline(observation, hidden)
    baseline_gradient = torch.autograd.grad(baseline_logits.sum(), observation, retain_graph=True)[0]
    settled_logits, _, _ = settled(observation, hidden)
    settled_gradient = torch.autograd.grad(settled_logits.sum(), observation)[0]

    assert torch.allclose(baseline_gradient, torch.zeros_like(baseline_gradient), atol=1e-9)
    assert float(settled_gradient.abs().sum()) > 0.0


def test_settled_checkpoint_exports_internal_steps(tmp_path: Path) -> None:
    graph = _build_controller_v2_graph(load_reduced_graph_variant("1k"))
    model = SettledPopulationFixedGraphPolicy(graph, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER))
    checkpoint = {
        "format_version": 1,
        "environment_version": 11,
        "controller": "connectome",
        "model": {
            "observation_size": OBSERVATION_INPUT_SIZE,
            "actions": len(ACTION_ORDER),
            "graph": graph.to_checkpoint(),
            "interface": "population-settled",
        },
        "model_state_dict": model.state_dict(),
        "training": {
            "seed": "settled-test",
            "steps": 4,
            "envs": 1,
            "learning_rate": 3e-4,
            "world_seeds": ["settled-test:0000", "settled-test:0001", "settled-test:0002", "settled-test:0003"],
        },
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    policy_path = tmp_path / "policy.json"
    torch.save(checkpoint, checkpoint_path)
    assert validate_checkpoint(checkpoint).connectome_interface == "population-settled"
    export_policy(checkpoint_path, policy_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert policy["network"]["interfaceMode"] == "population-settled"
    assert policy["network"]["internalSteps"] == 2
