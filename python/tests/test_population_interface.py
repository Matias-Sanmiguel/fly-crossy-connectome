from __future__ import annotations

import json
from pathlib import Path

import torch

from fly_crossy.checkpoint import validate_checkpoint
from fly_crossy.connectome import load_reduced_graph_variant
from fly_crossy.env import WORLD_VERSION
from fly_crossy.models import PopulationFixedGraphPolicy
from fly_crossy.schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE
from fly_crossy.train import TrainingConfig, train


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_population_interface_parameter_count_is_capacity_invariant() -> None:
    graph_80 = load_reduced_graph_variant("80")
    graph_1k = load_reduced_graph_variant("1k")

    model_80 = PopulationFixedGraphPolicy(
        graph_80, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    model_1k = PopulationFixedGraphPolicy(
        graph_1k, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )

    assert model_80.sensory.out_features == 32
    assert model_1k.sensory.out_features == 32
    assert model_80.actor.in_features == 16
    assert model_1k.actor.in_features == 16
    assert _parameter_count(model_80) == _parameter_count(model_1k) == 11944


def test_population_interface_injects_only_declared_sensory_cells() -> None:
    graph = load_reduced_graph_variant("80")
    model = PopulationFixedGraphPolicy(
        graph, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    with torch.no_grad():
        model.recurrent_gain.zero_()
        model.time_constant.fill_(1.0)
        model.sensory.weight.fill_(0.01)

    observation = torch.ones(1, OBSERVATION_INPUT_SIZE)
    hidden = torch.zeros(1, graph.node_count)
    _, _, activity = model(observation, hidden)

    active = set(torch.nonzero(activity[0], as_tuple=False).flatten().tolist())
    sensory = set(model.sensory_indices.tolist())
    assert active
    assert active.issubset(sensory)


def test_population_actor_reads_only_declared_readout_cells() -> None:
    graph = load_reduced_graph_variant("80")
    model = PopulationFixedGraphPolicy(
        graph, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)
    )
    baseline = torch.zeros(1, graph.node_count)
    altered = baseline.clone()

    non_readout = next(
        index
        for index in range(graph.node_count)
        if index not in set(model.readout_indices.tolist())
    )
    altered[0, non_readout] = 100.0

    assert torch.equal(
        model.actor_from_activity(baseline),
        model.actor_from_activity(altered),
    )


def test_population_tiny_runs_round_trip_for_80_and_1k(tmp_path: Path) -> None:
    for variant, expected_nodes in (("80", 80), ("1k", 1000)):
        output = tmp_path / f"population-{variant}"
        train(
            TrainingConfig(
                controller="connectome",
                seed=f"population-{variant}-tiny",
                steps=4,
                envs=1,
                learning_rate=3e-4,
                output=output,
                device="cpu",
                connectome_graph=variant,
                connectome_interface="population",
            )
        )

        checkpoint = torch.load(
            output / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        validated = validate_checkpoint(checkpoint)
        metadata = json.loads(
            (output / "metadata.json").read_text(encoding="utf-8")
        )
        policy = json.loads(
            (output / "policy.json").read_text(encoding="utf-8")
        )

        assert validated.connectome_interface == "population"
        assert validated.graph is not None
        assert validated.graph.node_count == expected_nodes
        assert metadata["environmentVersion"] == WORLD_VERSION
        assert metadata["configuration"]["connectome_interface"] == "population"
        assert metadata["configuration"]["graphNodes"] == expected_nodes

        # Browser export remains the existing dense fixed-graph wire format,
        # with zeros outside the declared biological interface populations.
        assert policy["network"]["kind"] == "fixed-graph"
        assert policy["network"]["interfaceMode"] == "population"
        assert len(policy["network"]["sensoryWeights"]) == (
            expected_nodes * OBSERVATION_INPUT_SIZE
        )
        assert len(policy["network"]["actorWeights"]) == (
            len(ACTION_ORDER) * expected_nodes
        )
