from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fly_crossy.export import export_policy
from fly_crossy.connectome import build_reduced_graph
from fly_crossy.models import DensePolicy, FixedGraphPolicy
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


def _fixed_checkpoint(path: Path) -> tuple[Path, FixedGraphPolicy]:
    source = path.with_name("graph.json")
    source.write_text(
        json.dumps(
            {
                "version": "fixture-v1",
                "nodes": [
                    {"id": 101, "sign": 1, "role": "input"},
                    {"id": 102, "sign": -1, "role": "output"},
                ],
                "edges": [[0, 1, 2], [1, 0, 1]],
                "inputs": [[0, 0]],
                "outputs": [1],
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source.with_name("manifest.json").write_text(
        json.dumps(
            {
                "dataset": "Fixture v1",
                "source": "https://example.test/fixture",
                "license": "CC BY 4.0",
                "selection": "Two-cell fixture.",
                "graphSha256": digest,
            }
        ),
        encoding="utf-8",
    )
    graph = build_reduced_graph(
        source, selected_ids=(101, 102), atlas_visible_ids=(101, 102)
    )
    torch.manual_seed(9)
    policy = FixedGraphPolicy(graph, observation_size=3, actions=len(ACTION_ORDER))
    torch.save(
        {
            "format_version": 1,
            "controller": "connectome",
            "model": {
                "observation_size": 3,
                "actions": len(ACTION_ORDER),
                "graph": graph.to_checkpoint(),
            },
            "model_state_dict": policy.state_dict(),
        },
        path,
    )
    return path, policy


def test_fixed_graph_export_preserves_topology_provenance_and_trainable_weights(
    tmp_path: Path,
) -> None:
    checkpoint, policy = _fixed_checkpoint(tmp_path / "fixed.pt")

    payload = export_policy(checkpoint, tmp_path / "fixed.json")
    network = payload["network"]

    assert network["kind"] == "fixed-graph"
    assert network["bodyIds"] == [101, 102]
    assert network["recurrentSource"] == [0, 1]
    assert network["recurrentTarget"] == [1, 0]
    assert network["recurrentWeights"] == pytest.approx([1.0, -1.0])
    assert network["recurrentGain"] == pytest.approx(
        float(policy.recurrent_gain.detach())
    )
    assert network["timeConstant"] == pytest.approx(
        float(policy.time_constant.detach())
    )
    assert len(network["sensoryWeights"]) == 6
    assert len(network["actorWeights"]) == 10
    assert payload["activityBodyIds"] == [101, 102]
    assert payload["source"]["graphSourceHash"] == policy.graph.source_sha256
    assert payload["source"]["graphArtifactHash"] == policy.graph.artifact_sha256


def test_exported_fixed_graph_one_step_matches_pytorch(tmp_path: Path) -> None:
    checkpoint, policy = _fixed_checkpoint(tmp_path / "fixed.pt")
    payload = export_policy(checkpoint, tmp_path / "fixed.json")
    network = payload["network"]
    fixed_input = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)
    hidden = np.asarray([0.2, -0.3], dtype=np.float32)

    with torch.no_grad():
        expected_logits, _, expected_activity = policy(
            torch.from_numpy(fixed_input).unsqueeze(0),
            torch.from_numpy(hidden).unsqueeze(0),
        )

    sensory = np.asarray(network["sensoryWeights"]).reshape(2, 3) @ fixed_input
    recurrent = np.zeros(2)
    for source, target, weight in zip(
        network["recurrentSource"],
        network["recurrentTarget"],
        network["recurrentWeights"],
        strict=True,
    ):
        recurrent[target] += weight * hidden[source]
    candidate = np.tanh(sensory + network["recurrentGain"] * recurrent)
    actual_activity = (
        (1 - network["timeConstant"]) * hidden
        + network["timeConstant"] * candidate
    )
    actual_logits = (
        np.asarray(network["actorWeights"]).reshape(5, 2) @ actual_activity
        + np.asarray(network["actorBias"])
    )

    assert actual_activity.tolist() == pytest.approx(
        expected_activity.squeeze(0).tolist(), abs=1e-5
    )
    assert actual_logits.tolist() == pytest.approx(
        expected_logits.squeeze(0).tolist(), abs=1e-5
    )
