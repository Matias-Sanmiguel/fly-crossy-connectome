from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fly_crossy.checkpoint import validate_checkpoint
from fly_crossy.connectome import load_reduced_graph_variant
from fly_crossy.schema import OBSERVATION_INPUT_SIZE
from fly_crossy.train import (
    PREDICTIVE_TARGET_SIZE,
    WIDE_SENSORY_COUNT,
    TrainingConfig,
    _build_wide_sensory_graph,
    _predictive_traffic_target,
    train,
)


def test_wide_sensory_graph_is_deterministic_and_keeps_readouts_disjoint() -> None:
    base = load_reduced_graph_variant("1k")
    first = _build_wide_sensory_graph(base)
    second = _build_wide_sensory_graph(base)

    assert len(first.sensory_body_ids) == WIDE_SENSORY_COUNT == 128
    assert first.sensory_body_ids.tolist() == second.sensory_body_ids.tolist()
    assert set(int(x) for x in base.sensory_body_ids).issubset(
        set(int(x) for x in first.sensory_body_ids)
    )
    assert set(int(x) for x in first.sensory_body_ids).isdisjoint(
        set(int(x) for x in first.readout_body_ids)
    )
    assert first.edge_index.tolist() == base.edge_index.tolist()
    assert np.allclose(first.edge_weight, base.edge_weight)


def test_predictive_target_uses_three_rows_ahead_from_observation_v3() -> None:
    observations = torch.zeros(2, 3, OBSERVATION_INPUT_SIZE)
    target = _predictive_traffic_target(observations)

    assert target.shape == (2, 3, PREDICTIVE_TARGET_SIZE)
    assert PREDICTIVE_TARGET_SIZE == 132


def test_population_wide_predictive_tiny_training_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "population-wide-predictive"
    train(
        TrainingConfig(
            controller="connectome",
            seed="population-wide-predictive-tiny",
            steps=8,
            envs=1,
            learning_rate=3e-4,
            output=output,
            device="cpu",
            connectome_graph="1k",
            connectome_interface="population-wide-predictive",
        )
    )

    checkpoint = torch.load(
        output / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    validated = validate_checkpoint(checkpoint)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    policy = json.loads((output / "policy.json").read_text(encoding="utf-8"))

    assert validated.connectome_interface == "population-wide-predictive"
    assert validated.graph is not None
    assert len(validated.graph.sensory_body_ids) == 128
    assert len(validated.graph.readout_body_ids) == 16
    assert validated.state_dict["sensory.weight"].shape == (
        128,
        OBSERVATION_INPUT_SIZE,
    )
    assert "trafficPredictionLoss" in metrics["trainingCurve"][0]
    assert np.isfinite(metrics["trainingCurve"][0]["trafficPredictionLoss"])

    assert not any("predict" in key for key in validated.state_dict)
    assert policy["network"]["kind"] == "fixed-graph"
    assert policy["network"]["interfaceMode"] == "population"
    assert policy["network"]["inputSize"] == OBSERVATION_INPUT_SIZE
