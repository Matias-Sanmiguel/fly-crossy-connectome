from __future__ import annotations

from dataclasses import asdict, replace
import os
from pathlib import Path
import random
import shutil

import pytest
import torch

import fly_crossy.v7.artifacts as artifacts_module
from fly_crossy.v7.artifacts import (
    CheckpointProvenance,
    load_verified_checkpoint,
    preflight_resources,
    retain_run_artifacts,
    save_verified_checkpoint,
    sha256_file,
)
from fly_crossy.v7.contracts import ACTION_NAMES, profile_budget


def _provenance() -> CheckpointProvenance:
    return CheckpointProvenance(
        version="crossy-v7-test-1",
        profile="80",
        stage="student",
        action_order=ACTION_NAMES,
        graph_sha256="1" * 64,
        teacher_sha256="2" * 64,
        dataset_manifest_sha256="3" * 64,
        seed_manifest_sha256="4" * 64,
        completed_round=2,
        completed_epoch=3,
    )


def _payload(value: float = 1.0) -> dict[str, object]:
    return {
        "provenance": asdict(_provenance()),
        "model": {"weight": torch.tensor([value])},
        "optimizer": {"step": torch.tensor(4)},
        "torchRngState": torch.get_rng_state(),
        "pythonRngState": random.getstate(),
    }


@pytest.mark.parametrize("profile", ["smoke", "80", "1k", "full"])
def test_preflight_uses_exact_profile_disk_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    threshold = profile_budget(profile).minimum_free_bytes
    usage_type = type(shutil.disk_usage(tmp_path))
    monkeypatch.setattr(
        artifacts_module.shutil,
        "disk_usage",
        lambda path: usage_type(threshold * 2, threshold, threshold),
    )
    snapshot = preflight_resources(profile, tmp_path, torch.device("cpu"))
    assert snapshot.free_disk_bytes == threshold
    assert snapshot.available_ram_bytes > 0
    assert snapshot.device == "cpu"
    assert snapshot.torch_version == torch.__version__
    assert snapshot.thread_count == torch.get_num_threads()

    monkeypatch.setattr(
        artifacts_module.shutil,
        "disk_usage",
        lambda path: usage_type(threshold * 2, threshold + 1, threshold - 1),
    )
    with pytest.raises(RuntimeError, match="free disk"):
        preflight_resources(profile, tmp_path, torch.device("cpu"))


def test_checkpoint_round_trip_is_safe_hashed_and_restores_rng(tmp_path: Path) -> None:
    torch.manual_seed(77)
    random.seed(77)
    payload = _payload()
    expected_torch = torch.rand(3)
    expected_python = random.random()
    path = tmp_path / "best.pt"

    digest = save_verified_checkpoint(path, payload=payload, verify=lambda value: None)
    loaded = load_verified_checkpoint(path, expected=_provenance())
    assert digest == sha256_file(path)
    assert torch.equal(loaded["model"]["weight"], torch.tensor([1.0]))
    torch.set_rng_state(loaded["torchRngState"])
    random.setstate(loaded["pythonRngState"])
    assert torch.equal(torch.rand(3), expected_torch)
    assert random.random() == expected_python


def test_rng_and_optimizer_round_trip_reproduce_next_update(tmp_path: Path) -> None:
    torch.manual_seed(19)
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    inputs = torch.ones(2, 3)

    optimizer.zero_grad(set_to_none=True)
    model(inputs).square().mean().backward()
    optimizer.step()
    payload = {
        "provenance": asdict(_provenance()),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torchRngState": torch.get_rng_state(),
    }
    path = tmp_path / "latest.pt"
    save_verified_checkpoint(path, payload=payload, verify=lambda value: None)

    optimizer.zero_grad(set_to_none=True)
    model(inputs).square().mean().backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    loaded = load_verified_checkpoint(path, expected=_provenance())
    resumed = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.5))
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-2)
    resumed.load_state_dict(loaded["model"])
    resumed_optimizer.load_state_dict(loaded["optimizer"])
    torch.set_rng_state(loaded["torchRngState"])
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed(inputs).square().mean().backward()
    resumed_optimizer.step()

    assert all(
        torch.equal(expected[name], value)
        for name, value in resumed.state_dict().items()
    )


@pytest.mark.parametrize(
    "field",
    [
        "version",
        "profile",
        "stage",
        "action_order",
        "graph_sha256",
        "teacher_sha256",
        "dataset_manifest_sha256",
        "seed_manifest_sha256",
        "completed_round",
        "completed_epoch",
    ],
)
def test_loader_rejects_every_provenance_mismatch(
    tmp_path: Path, field: str
) -> None:
    path = tmp_path / "best.pt"
    save_verified_checkpoint(path, payload=_payload(), verify=lambda value: None)
    value = getattr(_provenance(), field)
    if field == "action_order":
        replacement = tuple(reversed(ACTION_NAMES))
    elif isinstance(value, str):
        replacement = "different"
    elif value is None:
        replacement = "f" * 64
    else:
        replacement = value + 1
    with pytest.raises(ValueError, match="provenance|action order"):
        load_verified_checkpoint(
            path, expected=replace(_provenance(), **{field: replacement})
        )


def test_loader_rejects_noncanonical_action_order_even_if_expected_matches(
    tmp_path: Path,
) -> None:
    provenance = replace(_provenance(), action_order=tuple(reversed(ACTION_NAMES)))
    payload = _payload()
    payload["provenance"] = asdict(provenance)
    path = tmp_path / "bad.pt"
    save_verified_checkpoint(path, payload=payload, verify=lambda value: None)
    with pytest.raises(ValueError, match="action order"):
        load_verified_checkpoint(path, expected=provenance)


def test_failed_verification_preserves_previous_checkpoint_and_ordering(
    tmp_path: Path,
) -> None:
    path = tmp_path / "best.pt"
    save_verified_checkpoint(path, payload=_payload(1.0), verify=lambda value: None)
    old_hash = sha256_file(path)

    def reject_while_old_is_still_visible(candidate) -> None:
        current = torch.load(path, map_location="cpu", weights_only=True)
        assert current["model"]["weight"].item() == 1.0
        raise ValueError("candidate rejected")

    with pytest.raises(ValueError, match="candidate rejected"):
        save_verified_checkpoint(path, payload=_payload(2.0), verify=reject_while_old_is_still_visible)
    assert sha256_file(path) == old_hash
    assert not list(tmp_path.glob(".v7-checkpoint-*"))


def test_corrupt_temporary_write_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "best.pt"
    save_verified_checkpoint(path, payload=_payload(1.0), verify=lambda value: None)
    old = path.read_bytes()

    def corrupt_save(payload, target) -> None:
        Path(target).write_bytes(b"not a checkpoint")

    monkeypatch.setattr(artifacts_module.torch, "save", corrupt_save)
    with pytest.raises(Exception):
        save_verified_checkpoint(path, payload=_payload(2.0), verify=lambda value: None)
    assert path.read_bytes() == old


def test_retention_keeps_only_durable_run_contract(tmp_path: Path) -> None:
    names = (
        "best.pt",
        "latest.pt",
        "report.json",
        "dataset-manifest.json",
        "old.log",
        "training.log",
        "scratch.pt",
        "dataset-shard.npz",
    )
    for index, name in enumerate(names):
        path = tmp_path / name
        path.write_text(name)
        os.utime(path, (index + 1, index + 1))
    extra = tmp_path / "cache"
    extra.mkdir()
    (extra / "junk").write_text("junk")

    retained = retain_run_artifacts(tmp_path)

    assert {path.name for path in retained} == {
        "best.pt",
        "latest.pt",
        "report.json",
        "dataset-manifest.json",
        "training.log",
    }
    assert {path.name for path in tmp_path.iterdir()} == {
        path.name for path in retained
    }
