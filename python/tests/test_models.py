from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from fly_crossy.models import DensePolicy
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
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert result["checkpointHash"] == digest
    assert metadata["checkpoint"]["sha256"] == digest
    assert metadata["configuration"]["steps"] == 16
    assert metadata["software"]["torch"] == torch.__version__
    assert metrics["totalSteps"] == 16
    assert metrics["trainingCurve"]
    assert policy["source"]["checkpointHash"] == digest
    _assert_finite_numbers(metrics)
