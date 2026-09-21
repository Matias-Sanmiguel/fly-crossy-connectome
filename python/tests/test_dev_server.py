from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from fly_crossy.dev_server import (
    CHECKPOINT_PATH,
    LEGACY_CHECKPOINT_PATH,
    controller,
)
from fly_crossy.env import WORLD_VERSION


ROOT = Path(__file__).resolve().parents[2]


def test_dev_server_is_fail_closed_until_v8_release_and_preserves_history() -> None:
    assert WORLD_VERSION == 8

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
        ROOT / "release/eval-v8/training/connectome/checkpoint.pt"
    )
    assert not CHECKPOINT_PATH.exists()
    assert controller is None
