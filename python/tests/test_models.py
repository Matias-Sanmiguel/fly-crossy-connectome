from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fly_crossy.connectome import ReducedGraphArtifact, build_reduced_graph
from fly_crossy.env import WORLD_VERSION
from fly_crossy.models import DensePolicy, FixedGraphPolicy
from fly_crossy.schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE
from fly_crossy.train import TrainingConfig, train


def test_dense_policy_shapes() -> None:
    policy = DensePolicy(
        observation_size=OBSERVATION_INPUT_SIZE,
        hidden_size=64,
        actions=len(ACTION_ORDER),
    )

    logits, value = policy(torch.zeros(3, OBSERVATION_INPUT_SIZE))

    assert logits.shape == (3, 5)
    assert value.shape == (3,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_dense_policy_uses_two_shared_tanh_hidden_layers() -> None:
    policy = DensePolicy(observation_size=7, hidden_size=11, actions=5)

    assert isinstance(policy.hidden_1, torch.nn.Linear)
    assert isinstance(policy.hidden_2, torch.nn.Linear)
    assert policy.hidden_1.in_features == 7
    assert policy.hidden_1.out_features == 11
    assert policy.hidden_2.in_features == 11
    assert policy.hidden_2.out_features == 11
    assert policy.actor.out_features == 5
    assert policy.critic.out_features == 1


def _fixed_graph(tmp_path: Path):
    source = tmp_path / "graph.json"
    source.write_text(
        json.dumps(
            {
                "version": "fixture-v1",
                "nodes": [
                    {"id": 101, "sign": 1, "role": "input"},
                    {"id": 102, "sign": -1, "role": "output"},
                ],
                "edges": [[0, 1, 2]],
                "inputs": [[0, 0]],
                "outputs": [1],
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
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
    return build_reduced_graph(
        source, selected_ids=(101, 102), atlas_visible_ids=(101, 102)
    )


def test_fixed_graph_policy_keeps_recurrence_fixed_and_trains_only_declared_interfaces(
    tmp_path: Path,
) -> None:
    policy = FixedGraphPolicy(_fixed_graph(tmp_path), observation_size=3, actions=5)

    assert policy.adjacency.layout == torch.sparse_coo
    assert "adjacency" in dict(policy.named_buffers())
    assert "adjacency" not in dict(policy.named_parameters())
    assert "adjacency" not in policy.state_dict()
    assert set(dict(policy.named_parameters())) == {
        "recurrent_gain",
        "time_constant",
        "sensory.weight",
        "actor.weight",
        "actor.bias",
        "critic.weight",
        "critic.bias",
    }


def test_fixed_graph_policy_one_step_matches_hand_derived_values(tmp_path: Path) -> None:
    policy = FixedGraphPolicy(_fixed_graph(tmp_path), observation_size=2, actions=2)
    with torch.no_grad():
        policy.sensory.weight.copy_(torch.tensor([[0.5, -0.25], [-0.4, 0.2]]))
        policy.actor.weight.copy_(torch.tensor([[1.0, -1.0], [0.5, 0.25]]))
        policy.actor.bias.copy_(torch.tensor([0.1, -0.2]))
        policy.critic.weight.copy_(torch.tensor([[0.75, -0.5]]))
        policy.critic.bias.copy_(torch.tensor([0.05]))
        policy.recurrent_gain.fill_(1.25)
        policy.time_constant.fill_(0.8)

    logits, value, activity = policy(
        torch.tensor([[0.1, -0.2]]), torch.tensor([[0.2, -0.3]])
    )

    assert activity.tolist()[0] == pytest.approx(
        [0.1197343957, 0.0747048367], abs=1e-7
    )
    assert logits.tolist()[0] == pytest.approx(
        [0.1450295590, -0.1214565930], abs=1e-7
    )
    assert value.tolist() == pytest.approx([0.1024483784], abs=1e-7)
    assert policy.normalized_activity(activity).tolist()[0] == pytest.approx(
        [0.5598671978, 0.5373524183], abs=1e-7
    )


def test_python_fixed_graph_one_step_matches_the_shared_browser_fixture() -> None:
    fixture_path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "fixed-graph-policy-v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    network = fixture["network"]
    source = fixture["source"]
    graph = ReducedGraphArtifact(
        dataset_version=source["datasetVersion"],
        source_url=source["sourceUrl"],
        license=source["license"],
        source_sha256=source["graphSourceHash"],
        selection_rule=source["selectionRule"],
        minimum_edge_threshold=1,
        body_ids=np.asarray(network["bodyIds"], dtype=np.int64),
        edge_index=np.asarray(
            [network["recurrentSource"], network["recurrentTarget"]], dtype=np.int64
        ),
        edge_weight=np.asarray(network["recurrentWeights"], dtype=np.float32),
        sensory_body_ids=np.asarray([101], dtype=np.int64),
        readout_body_ids=np.asarray([102], dtype=np.int64),
    )
    policy = FixedGraphPolicy(graph, observation_size=network["inputSize"], actions=5)
    with torch.no_grad():
        policy.sensory.weight.copy_(
            torch.tensor(network["sensoryWeights"]).reshape(2, network["inputSize"])
        )
        policy.actor.weight.copy_(torch.tensor(network["actorWeights"]).reshape(5, 2))
        policy.actor.bias.copy_(torch.tensor(network["actorBias"]))
        policy.recurrent_gain.fill_(network["recurrentGain"])
        policy.time_constant.fill_(network["timeConstant"])

    logits, _, activity = policy(
        torch.tensor([fixture["parity"]["input"]]),
        torch.tensor([fixture["parity"]["hidden"]]),
    )

    assert activity.tolist()[0] == pytest.approx(fixture["parity"]["activity"], abs=1e-5)
    assert logits.tolist()[0] == pytest.approx(fixture["parity"]["logits"], abs=1e-5)


def _assert_finite_numbers(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite_numbers(item)
    elif isinstance(value, float):
        assert torch.isfinite(torch.tensor(value))


def test_tiny_ppo_run_writes_a_consistent_artifact_bundle(tmp_path: Path) -> None:
    output = tmp_path / "tiny-run"

    result = train(
        TrainingConfig(
            controller="conventional",
            seed="tiny-test",
            steps=16,
            envs=2,
            learning_rate=3e-4,
            output=output,
            device="cpu",
        )
    )

    checkpoint = output / "checkpoint.pt"
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    policy = json.loads((output / "policy.json").read_text(encoding="utf-8"))
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert result["checkpointHash"] == digest
    assert metadata["checkpoint"]["sha256"] == digest
    assert metadata["configuration"]["steps"] == 16
    assert metadata["environmentVersion"] == WORLD_VERSION
    assert checkpoint_payload["environment_version"] == WORLD_VERSION
    assert checkpoint_payload["training"]["world_seeds"][:2] == [
        "tiny-test:0:0",
        "tiny-test:1:0",
    ]
    assert metadata["configuration"]["trainingWorldSeeds"] == checkpoint_payload[
        "training"
    ]["world_seeds"]
    assert metadata["software"]["torch"] == torch.__version__
    assert metrics["totalSteps"] == 16
    assert metrics["trainingCurve"]
    assert policy["source"]["checkpointHash"] == digest
    _assert_finite_numbers(metrics)


def test_tiny_connectome_ppo_run_exports_the_pinned_fixed_graph(tmp_path: Path) -> None:
    output = tmp_path / "tiny-connectome"

    train(
        TrainingConfig(
            controller="connectome",
            seed="tiny-connectome-test",
            steps=4,
            envs=1,
            learning_rate=3e-4,
            output=output,
            device="cpu",
        )
    )

    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    policy = json.loads((output / "policy.json").read_text(encoding="utf-8"))
    assert checkpoint["controller"] == "connectome"
    assert checkpoint["model"]["observation_size"] == OBSERVATION_INPUT_SIZE
    assert checkpoint["model"]["graph"]["node_count"] == 80
    assert policy["network"]["kind"] == "fixed-graph"
    assert len(policy["network"]["bodyIds"]) == 80
    assert len(policy["network"]["recurrentWeights"]) == 1296
    assert policy["source"]["graphSourceHash"] == (
        "2424c9dd2e44534e600aeda1a9058039b1f22a4bd983284a27adc10b22130719"
    )


@pytest.mark.parametrize("device", ["auto", "cuda"])
def test_cuda_capable_modes_configure_deterministic_cublas_before_cuda_probe(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    workspace_at_probe: list[str | None] = []

    def unavailable_cuda() -> bool:
        workspace_at_probe.append(os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        return False

    monkeypatch.setattr(torch.cuda, "is_available", unavailable_cuda)
    config = TrainingConfig(
        controller="conventional",
        seed="deterministic-runtime-test",
        steps=1,
        envs=1,
        learning_rate=3e-4,
        output=tmp_path / device,
        device=device,
    )

    if device == "cuda":
        with pytest.raises(ValueError, match="CUDA was requested"):
            train(config)
    else:
        train(config)

    assert workspace_at_probe
    assert workspace_at_probe[0] == ":4096:8"
