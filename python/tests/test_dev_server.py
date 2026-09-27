from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from fly_crossy.dev_server import (
    CHECKPOINT_PATH,
    LEGACY_CHECKPOINT_PATH,
    build_controller_runtime,
)
from fly_crossy.protocol import Observation
from fly_crossy.env import WORLD_VERSION


ROOT = Path(__file__).resolve().parents[2]


def test_dev_server_uses_released_v6_controller_through_the_v11_adapter() -> None:
    assert WORLD_VERSION == 11

    assert LEGACY_CHECKPOINT_PATH == (
        ROOT / "release/eval-v1/training/connectome/checkpoint.pt"
    )
    assert LEGACY_CHECKPOINT_PATH.is_file()

    v6 = ROOT / "release/eval-v6/training/connectome/checkpoint.pt"
    assert v6.is_file()
    saved = torch.load(v6, map_location="cpu", weights_only=True)
    assert saved["environment_version"] == 6
    assert saved["training"]["seed"] == "final-80n-v6-train-1m-01"

    v7 = ROOT / "release/eval-v7-80n-baseline/training/connectome/checkpoint.pt"
    assert v7.is_file()
    saved_v7 = torch.load(v7, map_location="cpu", weights_only=True)
    assert saved_v7["environment_version"] == 7

    assert CHECKPOINT_PATH == (
        ROOT / "release/eval-v6/training/connectome/checkpoint.pt"
    )
    assert CHECKPOINT_PATH.is_file()
    first = build_controller_runtime()
    second = build_controller_runtime()
    assert first is not second
    assert first.select_action is not second.select_action
    assert first.neural_activity is not second.neural_activity

    sample = Observation.model_validate({
        "type": "observation",
        "version": 2,
        "sessionId": "s-00000001",
        "episodeId": "e-00000001",
        "sequence": 2,
        "simulationTime": 1.0,
        "gameStep": 0,
        "observation": [0.0] * 517,
        "reward": 0.0,
    })
    first.select_action(sample)
    first_activity = first.neural_activity()
    second_activity = second.neural_activity()
    assert len(first_activity) == 80
    assert len(second_activity) == 80
    assert len({neuron_id for neuron_id, _ in first_activity}) == 80
    assert all(0.0 <= value <= 1.0 for _, value in first_activity)
    first.reset()
