from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fly_crossy.export import export_policy
from fly_crossy.models import DensePolicy
from fly_crossy.schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE


def _checkpoint(path: Path, hidden_size: int = 4) -> tuple[Path, DensePolicy]:
    torch.manual_seed(7)
    policy = DensePolicy(
        observation_size=OBSERVATION_INPUT_SIZE,
        hidden_size=hidden_size,
        actions=len(ACTION_ORDER),
    )
    torch.save(
        {
            "format_version": 1,
            "controller": "conventional",
            "model": {
                "observation_size": OBSERVATION_INPUT_SIZE,
                "hidden_size": hidden_size,
                "actions": len(ACTION_ORDER),
            },
            "model_state_dict": policy.state_dict(),
        },
        path,
    )
    return path, policy


def _run_exported(payload: dict[str, object], inputs: np.ndarray) -> np.ndarray:
    values = inputs.astype(np.float64)
    network = payload["network"]
    assert isinstance(network, dict)
    for layer in network["layers"]:
        weights = np.asarray(layer["weights"], dtype=np.float64).reshape(
            layer["outputSize"], layer["inputSize"]
        )
        values = weights @ values + np.asarray(layer["bias"], dtype=np.float64)
        if layer["activation"] == "tanh":
            values = np.tanh(values)
    return values


def test_export_contains_provenance_and_finite_row_major_weights(tmp_path: Path) -> None:
    checkpoint, _ = _checkpoint(tmp_path / "checkpoint.pt")
    output = tmp_path / "policy.json"

    payload = export_policy(checkpoint, output)

    assert payload["version"] == 1
    assert payload["observationVersion"] == 1
    assert payload["actions"] == [action.value for action in ACTION_ORDER]
    assert payload["source"]["kind"] == "predicted"
    assert payload["source"]["checkpointHash"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert payload["activityBodyIds"] == []
    assert [layer["name"] for layer in payload["network"]["layers"]] == [
        "hidden_1",
        "hidden_2",
        "actor",
    ]
    assert [layer["activation"] for layer in payload["network"]["layers"]] == [
        "tanh",
        "tanh",
        "linear",
    ]
    for layer in payload["network"]["layers"]:
        assert len(layer["weights"]) == layer["inputSize"] * layer["outputSize"]
        assert np.isfinite(layer["weights"]).all()
        assert np.isfinite(layer["bias"]).all()
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_exported_logits_match_pytorch_on_fixed_input(tmp_path: Path) -> None:
    checkpoint, policy = _checkpoint(tmp_path / "checkpoint.pt")
    payload = export_policy(checkpoint, tmp_path / "policy.json")
    fixed_input = np.linspace(-1.0, 1.0, OBSERVATION_INPUT_SIZE, dtype=np.float32)

    with torch.no_grad():
        expected, _ = policy(torch.from_numpy(fixed_input).unsqueeze(0))
    actual = _run_exported(payload, fixed_input)

    assert actual.tolist() == pytest.approx(expected.squeeze(0).tolist(), abs=1e-5)

