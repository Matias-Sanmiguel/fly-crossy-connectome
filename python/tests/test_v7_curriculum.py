from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import fly_crossy.v7.curriculum as curriculum_module
from fly_crossy.v7.curriculum import CurriculumConfig, run_curriculum
from fly_crossy.v7.evaluation import EpisodeMetrics


def _write_predecessor(path: Path, *, profile: str, decision: str) -> None:
    payload = {
        "schemaVersion": "crossy-v7-report-1",
        "profile": profile,
        "decision": {"nextAction": decision},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["reportSha256"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


@pytest.mark.parametrize(
    "profile,predecessor_profile",
    [("1k", "80"), ("full", "1k")],
)
def test_large_profiles_validate_promoting_predecessor_before_graph_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    predecessor_profile: str,
) -> None:
    report = tmp_path / "predecessor.json"
    _write_predecessor(report, profile=predecessor_profile, decision="continue")
    calls = []
    monkeypatch.setattr(
        curriculum_module,
        "build_visual_student",
        lambda *args, **kwargs: calls.append("graph") or None,
    )

    result = run_curriculum(
        CurriculumConfig(
            profile=profile,
            output=tmp_path / "run",
            predecessor_report=report,
        )
    )

    assert calls == []
    assert result["decision"]["nextAction"] == "reject"
    assert "predecessor" in result["decision"]["reasons"][0].lower()
    assert (tmp_path / "run" / "report.json").is_file()


def test_hash_invalid_predecessor_is_rejected_before_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "predecessor.json"
    _write_predecessor(report, profile="80", decision="promote")
    payload = json.loads(report.read_text())
    payload["reportSha256"] = "0" * 64
    report.write_text(json.dumps(payload))
    monkeypatch.setattr(
        curriculum_module,
        "preflight_resources",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("too late")),
    )

    result = run_curriculum(
        CurriculumConfig(
            profile="1k", output=tmp_path / "run", predecessor_report=report
        )
    )
    assert result["decision"]["nextAction"] == "reject"


def test_zero_time_budget_writes_resumable_latest_and_report(tmp_path: Path) -> None:
    result = run_curriculum(
        CurriculumConfig(
            profile="smoke",
            output=tmp_path / "run",
            time_budget_seconds=0.0,
        )
    )
    assert result["decision"]["nextAction"] == "continue"
    assert "time budget" in " ".join(result["decision"]["reasons"]).lower()
    assert (tmp_path / "run" / "latest.pt").is_file()
    assert (tmp_path / "run" / "report.json").is_file()


def test_smoke_runs_full_state_machine_and_emits_complete_contract(tmp_path: Path) -> None:
    output = tmp_path / "run"
    result = run_curriculum(
        CurriculumConfig(profile="smoke", output=output, time_budget_seconds=180.0)
    )

    assert result["events"] == [
        "preflight",
        "baseline",
        "dataset",
        "teacher",
        "student",
        "validation",
        "selected-final-test",
        "report",
    ]
    assert result["contractOnly"] is True
    assert result["decision"]["nextAction"] != "promote"
    assert result["rounds"] and result["rounds"][0]["teacherLoss"]
    assert result["rounds"][0]["studentLoss"]
    assert result["trainableParameters"]
    assert result["labels"] == {
        "environment": "simulated",
        "teacher": "privileged-observation-v4",
        "labels": "planner",
    }
    assert result["checkpointSha256"]
    assert result["seedManifestSha256"]
    assert result["datasetManifestSha256"]
    assert result["resources"]["device"] == "cpu"
    assert result["validation"]["candidate"]
    assert result["finalTest"]
    assert (output / "best.pt").is_file()
    assert (output / "latest.pt").is_file()
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    latest = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
    assert "student" in best
    assert "teacher" in latest and "student" in latest


def test_resume_preserves_seed_manifest_and_skips_completed_round(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    first = run_curriculum(
        CurriculumConfig(profile="smoke", output=output, time_budget_seconds=180.0)
    )
    resumed = run_curriculum(
        CurriculumConfig(
            profile="smoke", output=output, resume=True, time_budget_seconds=180.0
        )
    )
    assert resumed["seedManifestSha256"] == first["seedManifestSha256"]
    assert resumed["resume"]["startedAtRound"] == 1
    assert len(resumed["rounds"]) == 1


def test_nonfinite_update_writes_latest_and_never_runs_final_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_collect = curriculum_module._collect_partition
    monkeypatch.setattr(
        curriculum_module,
        "_collect_partition",
        lambda seeds, **kwargs: original_collect(
            (seeds[0],),
            max_steps=1,
            source=kwargs["source"],
            behavior=kwargs.get("behavior"),
        ),
    )
    evaluation_calls = []

    def fake_evaluate(policy, seeds, **kwargs):
        evaluation_calls.append(tuple(seeds))
        return tuple(
            EpisodeMetrics(seed, 1, 1.0, True, True, None, 0, 1, (i % 5,))
            for i, seed in enumerate(seeds)
        )

    monkeypatch.setattr(curriculum_module, "evaluate_policy", fake_evaluate)
    monkeypatch.setattr(
        curriculum_module,
        "fit_teacher_epoch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FloatingPointError("non-finite update")
        ),
    )
    output = tmp_path / "run"
    result = run_curriculum(CurriculumConfig(profile="smoke", output=output))
    assert result["decision"]["nextAction"] == "continue"
    assert result["finalTest"] == []
    assert "selected-final-test" not in result["events"]
    assert len(evaluation_calls) == 2
    assert (output / "latest.pt").is_file()
