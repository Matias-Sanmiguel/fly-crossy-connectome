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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dev_server_uses_released_v6_checkpoint_and_preserves_v3_history() -> None:
    assert WORLD_VERSION == 6

    assert LEGACY_CHECKPOINT_PATH == (
        ROOT / "release/eval-v1/training/connectome/checkpoint.pt"
    )
    assert LEGACY_CHECKPOINT_PATH.is_file()
    historical = torch.load(
        LEGACY_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    assert historical["environment_version"] == 3

    assert CHECKPOINT_PATH == (
        ROOT / "release/eval-v6/training/connectome/checkpoint.pt"
    )
    released = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    assert released["environment_version"] == 6
    assert released["controller"] == "connectome"
    assert released["training"]["seed"] == "final-80n-v6-train-1m-01"
    assert released["training"]["steps"] == 1_000_000

    # The runtime loader remains lazy; app construction already verifies the
    # artifact registry bytes during module import.
    assert controller is None

    manifest = json.loads(
        (ROOT / "runtime-artifacts-biomechanics.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest == {
        "artifacts": [
            {
                "population": 80,
                "graph": {
                    "path": "public/data/connectome/graph.json",
                    "sha256": _sha256(
                        ROOT / "public/data/connectome/graph.json"
                    ),
                },
                "checkpoint": {
                    "path": "release/eval-v6/training/connectome/checkpoint.pt",
                    "sha256": _sha256(CHECKPOINT_PATH),
                },
            }
        ]
    }
