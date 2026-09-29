from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from fly_crossy.runtime_controller import ConnectomeActionSelector
from fly_crossy.schema import ACTION_ORDER

from .artifacts import (
    CheckpointProvenance,
    preflight_resources,
    save_verified_checkpoint,
    sha256_file,
)
from .contracts import ACTION_NAMES, ProfileName, build_seed_manifest, profile_budget
from .dataset import TransitionDataset, collect_labeled_episode
from .evaluation import (
    EpisodeMetrics,
    aggregate_metrics,
    decide_promotion,
    evaluate_policy,
    robust_key,
)
from .student import build_visual_student
from .teacher import PrivilegedTeacher, fit_teacher_epoch, teacher_agreement
from .training import attach_teacher_logits, fit_student_epoch, student_behavior


REPORT_VERSION = "crossy-v7-report-1"
CHECKPOINT_VERSION = "crossy-v7-checkpoint-1"


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    profile: ProfileName
    output: Path
    device: str = "cpu"
    resume: bool = False
    namespace: str = "crossy-v7"
    time_budget_seconds: float | None = None
    flyhard_root: Path | None = None
    predecessor_report: Path | None = None


class _TimeBudgetExpired(RuntimeError):
    pass


def _canonical_hash(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "reportSha256"}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".v7-report-")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _early_rejection(config: CurriculumConfig, reason: str) -> dict[str, Any]:
    report = {
        "schemaVersion": REPORT_VERSION,
        "profile": config.profile,
        "contractOnly": config.profile == "smoke",
        "events": ["report"],
        "decision": {"nextAction": "reject", "predicates": {}, "reasons": [reason]},
        "elapsedSeconds": 0.0,
    }
    report["reportSha256"] = _canonical_hash(report)
    _write_json(config.output / "report.json", report)
    return report


def _validate_predecessor(config: CurriculumConfig) -> str | None:
    required = {"1k": "80", "full": "1k"}.get(config.profile)
    if required is None:
        return None
    if config.predecessor_report is None or not config.predecessor_report.is_file():
        return f"predecessor {required} promotion report is required"
    try:
        report = json.loads(config.predecessor_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return f"predecessor report is unreadable: {error}"
    if report.get("reportSha256") != _canonical_hash(report):
        return "predecessor report hash is invalid"
    if report.get("schemaVersion") != REPORT_VERSION or report.get("profile") != required:
        return f"predecessor report must be a V7 {required} report"
    if report.get("decision", {}).get("nextAction") != "promote":
        return "predecessor report does not authorize promotion"
    return None


class _ReleasedBaseline:
    def __init__(self, device: str) -> None:
        root = Path(__file__).resolve().parents[3]
        checkpoint = root / "release/eval-v6/training/connectome/checkpoint.pt"
        self.selector = ConnectomeActionSelector(
            checkpoint, device=device, expected_environment_version=6
        )
        self.episode = 0

    def reset(self) -> None:
        self.selector.reset()
        self.episode += 1

    def act(self, state, frame, observation) -> int:
        del frame
        message = SimpleNamespace(
            episode_id=f"e-{self.episode:08d}",
            observation=observation.tolist(),
            game_step=state.step,
        )
        action = self.selector(message)
        return ACTION_NAMES.index(action)


def _collect_partition(
    seeds: tuple[str, ...],
    *,
    max_steps: int,
    source: str,
    behavior=None,
) -> TransitionDataset:
    combined = None
    for seed in seeds:
        episode = collect_labeled_episode(
            seed,
            max_steps=max_steps,
            planner_depth=1,
            source=source,
            behavior=behavior,
        )
        combined = episode if combined is None else combined.concatenate(episode)
    assert combined is not None
    return combined


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _graph_hash(student: torch.nn.Module) -> str | None:
    graph = getattr(getattr(student, "dynamics", None), "graph", None)
    value = getattr(graph, "artifact_sha256", None)
    return str(value) if value is not None else None


def _episodes_json(episodes: tuple[EpisodeMetrics, ...]) -> list[dict[str, Any]]:
    return [asdict(episode) for episode in episodes]


@torch.no_grad()
def _student_best_action_agreement(
    student: torch.nn.Module,
    dataset: TransitionDataset,
    device: torch.device,
) -> float:
    correct = 0
    total = 0
    student.eval()
    for seed in np.unique(dataset.episode_seed):
        rows = np.flatnonzero(dataset.episode_seed == seed)
        rows = rows[np.argsort(dataset.step_index[rows])]
        recurrent = student.zero_state(1, device=device)
        for row in rows:
            frame = torch.as_tensor(
                dataset.frames[row : row + 1], dtype=torch.float32, device=device
            ) / 255.0
            output = student(frame, recurrent)
            recurrent = output.next_recurrent_state
            action = int(output.logits[0].argmax().item())
            correct += int(action == int(dataset.best_action[row]))
            total += 1
    return correct / total


@torch.no_grad()
def _student_agreement(
    student: torch.nn.Module,
    dataset: TransitionDataset,
    device: torch.device,
) -> float:
    correct = 0
    total = 0
    student.eval()
    for seed in np.unique(dataset.episode_seed):
        rows = np.flatnonzero(dataset.episode_seed == seed)
        rows = rows[np.argsort(dataset.step_index[rows])]
        recurrent = student.zero_state(1, device=device)
        for row in rows:
            frame = torch.as_tensor(
                dataset.frames[row : row + 1], dtype=torch.float32, device=device
            ) / 255.0
            output = student(frame, recurrent)
            recurrent = output.next_recurrent_state
            action = int(output.logits[0].argmax().item())
            correct += int(dataset.acceptable[row, action])
            total += 1
    return correct / total


def _checkpoint_payload(
    *,
    provenance: CheckpointProvenance,
    teacher: PrivilegedTeacher | None,
    student: torch.nn.Module | None,
    teacher_optimizer=None,
    student_optimizer=None,
) -> dict[str, Any]:
    return {
        "provenance": asdict(provenance),
        "teacher": {} if teacher is None else teacher.state_dict(),
        "student": {} if student is None else student.state_dict(),
        "teacherOptimizer": {} if teacher_optimizer is None else teacher_optimizer.state_dict(),
        "studentOptimizer": {} if student_optimizer is None else student_optimizer.state_dict(),
        "torchRngState": torch.get_rng_state(),
        "pythonRngState": random.getstate(),
        "numpyRngState": torch.from_numpy(
            np.random.get_state()[1].astype(np.uint32).copy()
        ),
    }


def run_curriculum(config: CurriculumConfig) -> dict[str, Any]:
    started = time.monotonic()
    predecessor_error = _validate_predecessor(config)
    if predecessor_error is not None:
        return _early_rejection(config, predecessor_error)
    budget = profile_budget(config.profile)
    device = torch.device(config.device)
    limit = config.time_budget_seconds
    if limit is None and config.profile == "80" and device.type == "cpu":
        limit = 3600.0
    events: list[str] = []
    rounds: list[dict[str, Any]] = []
    config.output.mkdir(parents=True, exist_ok=True)
    if config.resume and (config.output / "report.json").is_file():
        previous_report = json.loads(
            (config.output / "report.json").read_text(encoding="utf-8")
        )
        rounds = list(previous_report.get("rounds", []))

    def expired() -> bool:
        return limit is not None and time.monotonic() - started >= limit

    resources = preflight_resources(config.profile, config.output, device)
    events.append("preflight")
    manifest = build_seed_manifest(config.profile, config.namespace)
    seed_hash = manifest.sha256()
    minimal_provenance = CheckpointProvenance(
        CHECKPOINT_VERSION, config.profile, "curriculum", ACTION_NAMES,
        None, None, "0" * 64, seed_hash, -1, -1,
    )
    if expired():
        save_verified_checkpoint(
            config.output / "latest.pt",
            payload=_checkpoint_payload(
                provenance=minimal_provenance, teacher=None, student=None
            ),
            verify=lambda payload: None,
        )
        report = {
            "schemaVersion": REPORT_VERSION,
            "profile": config.profile,
            "contractOnly": config.profile == "smoke",
            "events": events + ["report"],
            "resources": asdict(resources),
            "seedManifestSha256": seed_hash,
            "decision": {
                "nextAction": "continue",
                "predicates": {},
                "reasons": ["time budget expired before baseline"],
            },
            "elapsedSeconds": time.monotonic() - started,
        }
        report["reportSha256"] = _canonical_hash(report)
        _write_json(config.output / "report.json", report)
        return report

    baseline = _ReleasedBaseline(config.device)
    baseline_validation = evaluate_policy(
        baseline,
        manifest.validation,
        max_steps=budget.max_steps,
        planner_depth=1,
    )
    events.append("baseline")
    dataset = _collect_partition(
        manifest.training, max_steps=budget.max_steps, source="planner"
    )
    validation_dataset = _collect_partition(
        manifest.validation, max_steps=budget.max_steps, source="planner"
    )
    dataset_manifest = {
        "seedManifestSha256": seed_hash,
        "rows": len(dataset.frames),
        "actionOrder": ACTION_NAMES,
    }
    dataset_hash = _canonical_hash(dataset_manifest)
    dataset_manifest["sha256"] = dataset_hash
    _write_json(config.output / "dataset-manifest.json", dataset_manifest)
    events.append("dataset")

    teacher = PrivilegedTeacher().to(device)
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
    student_profile = "80" if config.profile == "smoke" else config.profile
    student = build_visual_student(
        student_profile, device=device, flyhard_root=config.flyhard_root
    )
    student_optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    start_round = 0
    if config.resume and (config.output / "latest.pt").is_file():
        saved = torch.load(config.output / "latest.pt", map_location=device, weights_only=True)
        saved_provenance = saved.get("provenance", {})
        if saved_provenance.get("profile") != config.profile or saved_provenance.get("seed_manifest_sha256") != seed_hash:
            raise ValueError("resume checkpoint profile or seed manifest does not match")
        if saved.get("teacher"):
            teacher.load_state_dict(saved["teacher"])
            student.load_state_dict(saved["student"])
            teacher_optimizer.load_state_dict(saved["teacherOptimizer"])
            student_optimizer.load_state_dict(saved["studentOptimizer"])
            torch.set_rng_state(saved["torchRngState"].cpu())
            random.setstate(saved["pythonRngState"])
        start_round = int(saved_provenance.get("completed_round", -1)) + 1

    initial_provenance = CheckpointProvenance(
        CHECKPOINT_VERSION, config.profile, "student", ACTION_NAMES,
        _graph_hash(student), _state_hash(teacher), dataset_hash, seed_hash,
        start_round - 1, 0,
    )
    initial_payload = _checkpoint_payload(
        provenance=initial_provenance,
        teacher=teacher,
        student=student,
        teacher_optimizer=teacher_optimizer,
        student_optimizer=student_optimizer,
    )
    save_verified_checkpoint(
        config.output / "best.pt", payload=initial_payload, verify=lambda payload: None
    )
    save_verified_checkpoint(
        config.output / "latest.pt", payload=initial_payload, verify=lambda payload: None
    )
    best_key = None
    best_validation = None
    interrupted_reason = None
    try:
        for round_index in range(start_round, budget.dagger_rounds):
            if expired():
                raise _TimeBudgetExpired("time budget expired between DAgger rounds")
            teacher_metrics = fit_teacher_epoch(
                teacher, dataset, teacher_optimizer, batch_size=64, seed=1000 + round_index
            )
            if "teacher" not in events:
                events.append("teacher")
            logits = attach_teacher_logits(dataset, teacher, device)
            student_metrics = fit_student_epoch(
                student,
                dataset,
                logits,
                student_optimizer,
                batch_size=16,
                window=8,
                seed=2000 + round_index,
            )
            if "student" not in events:
                events.append("student")
            candidate_validation = evaluate_policy(
                student_behavior(student, device),
                manifest.validation,
                max_steps=budget.max_steps,
                planner_depth=1,
            )
            if "validation" not in events:
                events.append("validation")
            candidate_key = robust_key(aggregate_metrics(candidate_validation))
            selected = best_key is None or candidate_key > best_key
            teacher_sha = _state_hash(teacher)
            provenance = CheckpointProvenance(
                CHECKPOINT_VERSION, config.profile, "student", ACTION_NAMES,
                _graph_hash(student), teacher_sha, dataset_hash, seed_hash,
                round_index, 0,
            )
            payload = _checkpoint_payload(
                provenance=provenance,
                teacher=teacher,
                student=student,
                teacher_optimizer=teacher_optimizer,
                student_optimizer=student_optimizer,
            )
            latest_hash = save_verified_checkpoint(
                config.output / "latest.pt", payload=payload, verify=lambda value: None
            )
            if selected:
                save_verified_checkpoint(
                    config.output / "best.pt", payload=payload, verify=lambda value: None
                )
                best_key = candidate_key
                best_validation = candidate_validation
            else:
                best = torch.load(
                    config.output / "best.pt", map_location=device, weights_only=True
                )
                teacher.load_state_dict(best["teacher"])
                student.load_state_dict(best["student"])
            rounds.append(
                {
                    "round": round_index,
                    "teacherLoss": teacher_metrics,
                    "studentLoss": student_metrics,
                    "selected": selected,
                    "latestCheckpointSha256": latest_hash,
                }
            )
            if round_index + 1 < budget.dagger_rounds:
                dagger = _collect_partition(
                    manifest.training,
                    max_steps=budget.max_steps,
                    source="student",
                    behavior=student_behavior(student, device),
                )
                dataset = dataset.concatenate(dagger)
    except (KeyboardInterrupt, _TimeBudgetExpired, FloatingPointError) as error:
        interrupted_reason = str(error) or type(error).__name__
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        interrupted_reason = f"out of memory: {error}"

    if best_validation is None:
        best_validation = evaluate_policy(
            student_behavior(student, device),
            manifest.validation,
            max_steps=budget.max_steps,
            planner_depth=1,
        )
    best = torch.load(config.output / "best.pt", map_location=device, weights_only=True)
    if best.get("teacher"):
        teacher.load_state_dict(best["teacher"])
        student.load_state_dict(best["student"])
    if interrupted_reason is None:
        final_test = evaluate_policy(
            student_behavior(student, device),
            manifest.final_test,
            max_steps=budget.max_steps,
            planner_depth=1,
        )
        events.append("selected-final-test")
    else:
        final_test = ()
    teacher_validation_agreement = teacher_agreement(
        teacher, validation_dataset, device
    )
    agreement = _student_agreement(student, validation_dataset, device)
    best_action_agreement = _student_best_action_agreement(
        student, validation_dataset, device
    )
    decision = decide_promotion(
        profile=config.profile,
        predecessor=baseline_validation,
        candidate=best_validation,
        planner_agreement=agreement,
        agreement_threshold=0.8,
        final_test_passed=interrupted_reason is None,
    )
    predicates = dict(decision.predicates)
    predicates["teacherAgreementPassed"] = teacher_validation_agreement >= 0.9
    reasons = list(decision.reasons)
    if not predicates["teacherAgreementPassed"]:
        reasons.append("teacherAgreementPassed")
    next_action = decision.next_action
    if next_action == "promote" and not predicates["teacherAgreementPassed"]:
        next_action = "continue"
    if interrupted_reason is not None:
        next_action = "continue" if isinstance(interrupted_reason, str) else "reject"
        reasons.append(interrupted_reason)
    checkpoint_hash = sha256_file(config.output / "best.pt")
    report = {
        "schemaVersion": REPORT_VERSION,
        "profile": config.profile,
        "contractOnly": config.profile == "smoke",
        "events": events + ["report"],
        "resources": asdict(resources),
        "labels": {
            "environment": "simulated",
            "teacher": "privileged-observation-v4",
            "labels": "planner",
        },
        "actionExecution": {
            "curriculum": "logical-step-game",
            "runtimeAuthority": "biomechanical-gate",
        },
        "seedManifestSha256": seed_hash,
        "datasetManifestSha256": dataset_hash,
        "trainableParameters": [
            name for name, parameter in student.named_parameters() if parameter.requires_grad
        ],
        "rounds": rounds,
        "resume": {"requested": config.resume, "startedAtRound": start_round},
        "validation": {
            "baseline": _episodes_json(baseline_validation),
            "candidate": _episodes_json(best_validation),
            "plannerAgreement": agreement,
            "plannerBestActionAgreement": best_action_agreement,
            "teacherAgreement": teacher_validation_agreement,
        },
        "aggregateMetrics": {
            "validationBaseline": asdict(aggregate_metrics(baseline_validation)),
            "validationCandidate": asdict(aggregate_metrics(best_validation)),
            "finalTest": (
                asdict(aggregate_metrics(final_test)) if final_test else None
            ),
        },
        "finalTest": _episodes_json(final_test),
        "decision": {
            "nextAction": next_action,
            "predicates": predicates,
            "reasons": reasons,
        },
        "checkpointSha256": checkpoint_hash,
        "elapsedSeconds": time.monotonic() - started,
    }
    report["reportSha256"] = _canonical_hash(report)
    _write_json(config.output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the V7 visual connectome curriculum")
    parser.add_argument("--profile", choices=("smoke", "80", "1k", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--namespace", default="crossy-v7")
    parser.add_argument("--time-budget-seconds", type=float)
    parser.add_argument("--flyhard-root", type=Path)
    parser.add_argument("--predecessor-report", type=Path)
    args = parser.parse_args()
    report = run_curriculum(
        CurriculumConfig(
            profile=args.profile,
            output=args.out,
            device=args.device,
            resume=args.resume,
            namespace=args.namespace,
            time_budget_seconds=args.time_budget_seconds,
            flyhard_root=args.flyhard_root,
            predecessor_report=args.predecessor_report,
        )
    )
    print(json.dumps({"report": str(args.out / "report.json"), "decision": report["decision"]}))


if __name__ == "__main__":
    main()
