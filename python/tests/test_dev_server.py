from __future__ import annotations

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


def test_dev_server_does_not_activate_historical_v3_checkpoint_for_v5() -> None:
    assert WORLD_VERSION == 5

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
        ROOT / "release/eval-v5/training/connectome/checkpoint.pt"
    )

    # Loading stays lazy. During the freeze->training transition there is no
    # valid v5 runtime checkpoint to instantiate yet.
    assert controller is None

    manifest = json.loads(
        (ROOT / "runtime-artifacts-biomechanics.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(manifest["artifacts"], list)
    assert all(
        artifact["checkpoint"]["path"]
        != "release/eval-v1/training/connectome/checkpoint.pt"
        for artifact in manifest["artifacts"]
    )
