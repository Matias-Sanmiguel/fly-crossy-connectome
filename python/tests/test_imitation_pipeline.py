from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fly_crossy.checkpoint import validate_checkpoint
from fly_crossy.expert_dataset import generate_dataset
from fly_crossy.imitation_train import train_imitation
from fly_crossy.schema import OBSERVATION_INPUT_SIZE


def test_expert_dataset_preserves_recurrent_episode_boundaries(
    tmp_path: Path,
) -> None:
    output = tmp_path / "expert"
    manifest = generate_dataset(
        output=output,
        episodes=3,
        depth=2,
        max_steps=5,
        seed_prefix="imitation-test",
        workers=1,
    )

    with np.load(output / "dataset.npz", allow_pickle=False) as data:
        observations = data["observations"]
        actions = data["actions"]
        offsets = data["episode_offsets"]
        seeds = data["seeds"]

    assert manifest["episodes"] == 3
    assert observations.shape[1] == OBSERVATION_INPUT_SIZE
    assert len(actions) == offsets[-1]
    assert offsets.tolist()[0] == 0
    assert len(offsets) == 4
    assert len(seeds) == 3
    assert all(str(seed).startswith("imitation-test:") for seed in seeds)


def test_tiny_imitation_training_exports_plain_neural_population_policy(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "expert"
    generate_dataset(
        output=dataset,
        episodes=4,
        depth=2,
        max_steps=4,
        seed_prefix="imitation-tiny",
        workers=1,
    )

    output = tmp_path / "run"
    result = train_imitation(
        dataset=dataset,
        output=output,
        epochs=1,
        batch_episodes=2,
        learning_rate=3e-4,
        validation_fraction=0.25,
        seed=7,
        patience=1,
        device_name="cpu",
    )

    saved = torch.load(
        output / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = validate_checkpoint(saved)
    policy = json.loads((output / "policy.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))

    assert checkpoint.connectome_interface == "population"
    assert checkpoint.graph is not None
    assert len(checkpoint.graph.sensory_body_ids) == 128
    assert len(checkpoint.graph.readout_body_ids) == 128
    assert saved["training"]["training_mode"] == "expert-imitation-bptt-v1"
    assert policy["network"]["kind"] == "fixed-graph"
    assert policy["network"].get("interfaceMode") != "controller-v2"
    assert "riskWeights" not in policy["network"]
    assert "routeWeights" not in policy["network"]
    assert result["validationAccuracy"] >= 0.0
    assert "validationPerActionAccuracy" in result
    assert metrics["trainingMode"] == "expert-imitation-bptt-v1"
