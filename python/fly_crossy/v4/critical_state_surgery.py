from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from fly_crossy.v2.preference_distill import preference_loss
from fly_crossy.v4.exact_seed_branch_curriculum import (
    build_exact_teacher_trace,
    load_decoder_checkpoint,
)
from fly_crossy.v4.recovery_dagger_v2 import (
    ACTION_NAMES,
    RecoveryDecoderV2,
    clone_decoder,
    evaluate_suite,
    evaluate_variant,
    load_frozen_policy,
    repo_root,
    save_decoder,
)
from fly_crossy.v4.student_state_frontier_curriculum import (
    StudentSnapshot,
    collect_student_trace,
)
from fly_crossy.v4.train_expo_specialist import EXPO_SEED, default_flyhard_root


VERSION = "crossy-v4-critical-state-surgery-1"


@dataclass(frozen=True)
class SurgeryRecipe:
    name: str
    learning_rate: float
    epochs: int
    margin: float
    margin_weight: float
    distill_weight: float
    parameter_weight: float


DEFAULT_RECIPES = (
    SurgeryRecipe("conservative", 0.0010, 140, 1.0, 2.0, 20.0, 0.25),
    SurgeryRecipe("balanced", 0.0015, 180, 1.5, 3.0, 10.0, 0.15),
    SurgeryRecipe("focused", 0.0020, 220, 2.0, 4.0, 5.0, 0.10),
)


def _safe_float(value: float | np.floating) -> float:
    return float(value)


def _first_unsafe_index(snapshots: list[StudentSnapshot]) -> int | None:
    for i, snap in enumerate(snapshots):
        if not bool(snap.safe[snap.student_action]):
            return i
    return None


def _first_unacceptable_index(snapshots: list[StudentSnapshot]) -> int | None:
    for i, snap in enumerate(snapshots):
        if not bool(snap.acceptable[snap.student_action]):
            return i
    return None


def select_primary_critical(snapshots: list[StudentSnapshot]) -> tuple[int, str]:
    """Return the first unsafe state, otherwise first unacceptable, otherwise frontier."""
    if not snapshots:
        raise ValueError("Cannot select a critical state from an empty trace.")
    unsafe = _first_unsafe_index(snapshots)
    if unsafe is not None:
        return unsafe, "first-unsafe"
    unacceptable = _first_unacceptable_index(snapshots)
    if unacceptable is not None:
        return unacceptable, "first-unacceptable"
    return len(snapshots) - 1, "frontier-last-state"


def select_correction_indices(
    snapshots: list[StudentSnapshot],
    *,
    primary: int,
    tail: int,
    max_states: int,
) -> list[int]:
    """Small correction set: primary failure plus nearby actual student mistakes."""
    if max_states < 1:
        return []
    start = max(0, primary - max(0, int(tail)))
    candidates: list[tuple[int, int]] = []
    for i in range(start, primary + 1):
        snap = snapshots[i]
        action = snap.student_action
        if not bool(snap.safe[action]):
            priority = 0
        elif not bool(snap.acceptable[action]):
            priority = 1
        elif i == primary:
            priority = 2
        else:
            continue
        candidates.append((priority, i))
    candidates.sort(key=lambda item: (item[0], -item[1]))
    selected: list[int] = []
    for _, i in candidates:
        if i not in selected:
            selected.append(i)
        if len(selected) >= max_states:
            break
    if primary not in selected:
        selected.insert(0, primary)
    return sorted(selected[:max_states])


def _normalized_features(
    decoder: RecoveryDecoderV2,
    motors: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = torch.as_tensor(motors, dtype=torch.float32, device=device)
    z = (x - decoder.mean) / decoder.std
    return z.detach().cpu().numpy()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(1, -1)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a, axis=1)[0] * np.linalg.norm(b, axis=1)
    denom = np.maximum(denom, 1e-12)
    return (b @ a[0]) / denom


def nearest_rows(
    query: np.ndarray,
    matrix: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    k: int,
) -> list[dict[str, Any]]:
    if len(matrix) == 0:
        return []
    cos = _cosine_similarity(query, matrix)
    l2 = np.linalg.norm(matrix.astype(np.float64) - query.astype(np.float64), axis=1)
    order = np.lexsort((l2, -cos))[: max(1, min(k, len(matrix)))]
    rows: list[dict[str, Any]] = []
    for index in order:
        row = dict(metadata[int(index)])
        row.update({"cosine": float(cos[index]), "l2": float(l2[index])})
        rows.append(row)
    return rows


def _snapshot_meta(snapshot: Any, index: int, source: str) -> dict[str, Any]:
    acceptable = [ACTION_NAMES[i] for i in np.flatnonzero(snapshot.acceptable)]
    safe = [ACTION_NAMES[i] for i in np.flatnonzero(snapshot.safe)]
    student_action = getattr(snapshot, "student_action", None)
    return {
        "source": source,
        "index": int(index),
        "score": float(snapshot.state.score),
        "row": int(snapshot.state.fly.row),
        "column": float(snapshot.state.fly.column),
        "bestAction": ACTION_NAMES[int(snapshot.best_action)],
        "studentAction": None if student_action is None else ACTION_NAMES[int(student_action)],
        "acceptableActions": acceptable,
        "safeActions": safe,
    }


def separability_report(
    *,
    decoder: RecoveryDecoderV2,
    student: list[StudentSnapshot],
    teacher_snapshots: list[Any],
    critical_index: int,
    device: torch.device,
    k: int,
) -> dict[str, Any]:
    critical = student[critical_index]
    target_action = int(critical.best_action)
    bad_action = int(critical.student_action)

    student_motors = np.stack([s.motor for s in student]).astype(np.float32)
    teacher_motors = np.stack([s.motor for s in teacher_snapshots]).astype(np.float32)
    student_z = _normalized_features(decoder, student_motors, device)
    teacher_z = _normalized_features(decoder, teacher_motors, device)
    query = student_z[critical_index]

    previous = student_z[:critical_index]
    previous_meta = [_snapshot_meta(s, i, "student-prefix") for i, s in enumerate(student[:critical_index])]
    teacher_meta = [_snapshot_meta(s, i, "teacher") for i, s in enumerate(teacher_snapshots)]

    with torch.no_grad():
        logits = decoder(
            torch.as_tensor(critical.motor[None], dtype=torch.float32, device=device)
        )[0]
        margin = float((logits[target_action] - logits[bad_action]).item())
        logits_dict = {ACTION_NAMES[i]: float(logits[i].item()) for i in range(len(ACTION_NAMES))}

    conflict_indices: list[int] = []
    target_indices: list[int] = []
    for i, snap in enumerate(student[:critical_index]):
        if bool(snap.acceptable[bad_action]) and not bool(snap.acceptable[target_action]):
            conflict_indices.append(i)
        if bool(snap.acceptable[target_action]):
            target_indices.append(i)

    def best_similarity(indices: list[int]) -> dict[str, Any] | None:
        if not indices:
            return None
        subset = student_z[np.asarray(indices, dtype=np.int64)]
        cos = _cosine_similarity(query, subset)
        local = int(np.argmax(cos))
        global_index = indices[local]
        l2 = float(np.linalg.norm(student_z[global_index] - query))
        return {
            **_snapshot_meta(student[global_index], global_index, "student-prefix"),
            "cosine": float(cos[local]),
            "l2": l2,
        }

    return {
        "criticalIndex": int(critical_index),
        "score": float(critical.state.score),
        "row": int(critical.state.fly.row),
        "column": float(critical.state.fly.column),
        "targetAction": ACTION_NAMES[target_action],
        "badAction": ACTION_NAMES[bad_action],
        "decoderTargetMinusBadMargin": margin,
        "decoderLogits": logits_dict,
        "nearestPreviousStudent": nearest_rows(query, previous, previous_meta, k=k),
        "nearestTeacher": nearest_rows(query, teacher_z, teacher_meta, k=k),
        "nearestBadActionConflict": best_similarity(conflict_indices),
        "nearestTargetSupport": best_similarity(target_indices),
    }


def _motor_tensor(snapshots: Iterable[Any], device: torch.device) -> torch.Tensor:
    arr = np.stack([snap.motor for snap in snapshots]).astype(np.float32)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _surgery_key(exact: dict[str, Any], first_unsafe: int | None) -> tuple[float, ...]:
    # Exact-seed specialist: a real success wins; otherwise score, then steps,
    # then delaying the first unsafe decision wins.
    return (
        float(exact["success"]),
        float(exact["score"]),
        float(exact["steps"]),
        float(10_000 if first_unsafe is None else first_unsafe),
    )


def train_surgical_candidate(
    *,
    base: RecoveryDecoderV2,
    student: list[StudentSnapshot],
    teacher_snapshots: list[Any],
    correction_indices: list[int],
    primary_index: int,
    recipe: SurgeryRecipe,
    teacher_prefix_end: int,
    device: torch.device,
    seed: int,
) -> tuple[RecoveryDecoderV2, dict[str, Any]]:
    if base.architecture != "mlp64" or base.residual is None:
        raise ValueError("Critical-state surgery requires the mlp64 decoder.")

    candidate = clone_decoder(base, device)
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    assert candidate.residual is not None
    for parameter in candidate.residual.parameters():
        parameter.requires_grad_(True)

    correction = [student[i] for i in correction_indices]
    corr_motor = _motor_tensor(correction, device)
    acceptable = torch.as_tensor(
        np.stack([s.acceptable for s in correction]), dtype=torch.bool, device=device
    )
    safe = torch.as_tensor(np.stack([s.safe for s in correction]), dtype=torch.bool, device=device)
    pair_sign = torch.as_tensor(
        np.stack([s.pair_sign for s in correction]), dtype=torch.float32, device=device
    )
    pair_weight = torch.as_tensor(
        np.stack([s.pair_weight for s in correction]), dtype=torch.float32, device=device
    )
    best = torch.as_tensor([s.best_action for s in correction], dtype=torch.long, device=device)
    bad = torch.as_tensor([s.student_action for s in correction], dtype=torch.long, device=device)

    corr_weights_np = np.ones(len(correction), dtype=np.float32)
    for j, index in enumerate(correction_indices):
        snap = student[index]
        if index == primary_index:
            corr_weights_np[j] = 20.0
        elif not bool(snap.safe[snap.student_action]):
            corr_weights_np[j] = 12.0
        elif not bool(snap.acceptable[snap.student_action]):
            corr_weights_np[j] = 6.0
        else:
            corr_weights_np[j] = 2.0
    corr_weights = torch.as_tensor(corr_weights_np, dtype=torch.float32, device=device)

    # Preserve everything before the primary failure plus a teacher anchor prefix.
    protected_student = [s for i, s in enumerate(student[:primary_index]) if i not in correction_indices]
    teacher_end = max(1, min(int(teacher_prefix_end), len(teacher_snapshots)))
    protected_teacher = list(teacher_snapshots[:teacher_end])
    protected = protected_student + protected_teacher
    protected_motor = _motor_tensor(protected, device)
    with torch.no_grad():
        protected_logits = base(protected_motor).detach()

    original_weight = base.residual.weight.detach().clone()
    original_bias = base.residual.bias.detach().clone()

    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        candidate.residual.parameters(), lr=recipe.learning_rate, weight_decay=0.0
    )
    history: list[dict[str, float]] = []

    for epoch in range(1, recipe.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = candidate(corr_motor)
        pref, _ = preference_loss(
            logits, acceptable, safe, pair_sign, pair_weight, corr_weights
        )
        rows = torch.arange(len(correction), device=device)
        target_margin = logits[rows, best] - logits[rows, bad]
        margin_each = F.relu(recipe.margin - target_margin)
        margin_loss = (margin_each * (corr_weights / corr_weights.mean())).mean()

        preserve_logits = candidate(protected_motor)
        distill = F.mse_loss(preserve_logits, protected_logits)
        parameter_delta = (
            F.mse_loss(candidate.residual.weight, original_weight)
            + F.mse_loss(candidate.residual.bias, original_bias)
        )
        loss = (
            pref
            + recipe.margin_weight * margin_loss
            + recipe.distill_weight * distill
            + recipe.parameter_weight * parameter_delta
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite critical-state surgery loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.residual.parameters(), 2.0)
        optimizer.step()

        if epoch == 1 or epoch == recipe.epochs or epoch % 40 == 0:
            history.append(
                {
                    "epoch": int(epoch),
                    "total": float(loss.detach()),
                    "preference": float(pref.detach()),
                    "margin": float(margin_loss.detach()),
                    "distill": float(distill.detach()),
                    "parameterDelta": float(parameter_delta.detach()),
                }
            )

    candidate.eval()
    with torch.no_grad():
        base_prefix_actions = base(_motor_tensor(student[:primary_index], device)).argmax(dim=1)
        cand_prefix_actions = candidate(_motor_tensor(student[:primary_index], device)).argmax(dim=1)
        prefix_agreement = (
            float((base_prefix_actions == cand_prefix_actions).float().mean().item())
            if primary_index > 0
            else 1.0
        )
        corr_logits = candidate(corr_motor)
        rows = torch.arange(len(correction), device=device)
        final_margins = (corr_logits[rows, best] - corr_logits[rows, bad]).cpu().numpy()

    return candidate, {
        "recipe": recipe.__dict__,
        "trainableParameters": int(sum(p.numel() for p in candidate.residual.parameters())),
        "correctionIndices": [int(i) for i in correction_indices],
        "protectedStudentFrames": int(len(protected_student)),
        "protectedTeacherFrames": int(len(protected_teacher)),
        "prefixActionAgreementWithBase": prefix_agreement,
        "criticalMarginsAfter": [float(x) for x in final_margins],
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Final decoder-side experiment for Crossy V4: measure the first critical "
            "student state and immediately try a trust-region surgical correction by "
            "training only the 64->5 residual layer. If it cannot improve the exact "
            "expo seed without breaking the prefix, the report explicitly escalates "
            "to sensor/core work instead of requesting another readout audit."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=EXPO_SEED)
    parser.add_argument("--rng-seed", type=int, default=909)
    parser.add_argument("--planner-depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--correction-tail", type=int, default=12)
    parser.add_argument("--max-correction-states", type=int, default=8)
    parser.add_argument("--teacher-margin", type=int, default=16)
    parser.add_argument("--nearest-k", type=int, default=8)
    parser.add_argument("--stall-steps", type=int, default=12)
    parser.add_argument("--min-teacher-fraction", type=float, default=0.80)
    parser.add_argument("--min-prefix-agreement", type=float, default=0.90)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root() / "runs" / "crossy-v4-expo-specialist" / "best.pt",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        default=repo_root()
        / "runs"
        / "crossy-v4-exact-seed-branch-curriculum"
        / "best_decoder.pt",
    )
    parser.add_argument("--flyhard-root", type=Path, default=default_flyhard_root())
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root() / "runs" / "crossy-v4-critical-state-surgery",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_steps = min(args.max_steps, 40)
        args.rounds = min(args.rounds, 2)
        args.correction_tail = min(args.correction_tail, 6)
        args.max_correction_states = min(args.max_correction_states, 4)
        args.teacher_margin = min(args.teacher_margin, 8)
        args.nearest_k = min(args.nearest_k, 4)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.rng_seed)
    started = time.perf_counter()

    policy, metadata = load_frozen_policy(
        checkpoint_path=args.checkpoint,
        flyhard_root=args.flyhard_root,
        device=device,
    )
    decoder, decoder_metadata = load_decoder_checkpoint(args.decoder_checkpoint, device=device)
    if decoder.architecture != "mlp64":
        raise SystemExit("This experiment requires the selected mlp64 decoder.")

    print("=== CROSSY V4 / CRITICAL-STATE SURGERY ===", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(
        f"MaleCNS frozen: {metadata['neurons']:,} neurons / {metadata['edges']:,} edges",
        flush=True,
    )
    print(
        "hard stop: this is the final decoder-side diagnostic/intervention; "
        "failure escalates to sensor/core work.",
        flush=True,
    )

    teacher, teacher_snapshots, teacher_report = build_exact_teacher_trace(
        policy=policy,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        planner_depth=args.planner_depth,
    )
    teacher_score = float(teacher_report["score"])
    target_score = teacher_score * float(args.min_teacher_fraction)
    effective_stall = max(int(args.stall_steps), int(teacher_report["maxNoProgressStreak"]) + 2)

    current = clone_decoder(decoder, device)
    baseline_suite = evaluate_suite(
        policy=policy,
        decoder=current,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        target_score=target_score,
        stall_steps=effective_stall,
        smoke=args.smoke,
    )

    config = {
        "seed": args.seed,
        "rngSeed": args.rng_seed,
        "plannerDepth": args.planner_depth,
        "maxSteps": args.max_steps,
        "rounds": args.rounds,
        "correctionTail": args.correction_tail,
        "maxCorrectionStates": args.max_correction_states,
        "teacherMargin": args.teacher_margin,
        "nearestK": args.nearest_k,
        "minimumTeacherFraction": args.min_teacher_fraction,
        "teacherScore": teacher_score,
        "minimumProgressScore": target_score,
        "effectiveStallSteps": effective_stall,
        "minimumPrefixAgreement": args.min_prefix_agreement,
        "MaleCNSRetrained": False,
        "decoderLayersTrainable": ["residual.weight", "residual.bias"],
        "decoderTrainableParameters": 325,
        "hardStopRule": "success => keep decoder; no exact-seed improvement => escalate to sensor/core; max rounds without success => escalate to sensor/core",
    }

    best_suite = baseline_suite
    best_stage = "critical-surgery-baseline"
    rounds: list[dict[str, Any]] = []
    initial_separability: dict[str, Any] | None = None
    reason = "max-rounds"

    save_decoder(
        args.out / "best_decoder.pt",
        decoder=current,
        source_checkpoint=args.checkpoint,
        stage=best_stage,
        metrics=best_suite,
        config=config,
    )

    recipes = list(DEFAULT_RECIPES)
    if args.smoke:
        recipes = [
            SurgeryRecipe(r.name, r.learning_rate, min(r.epochs, 20), r.margin, r.margin_weight, r.distill_weight, r.parameter_weight)
            for r in recipes[:2]
        ]

    for round_index in range(1, args.rounds + 1):
        _, snapshots, trace = collect_student_trace(
            policy=policy,
            decoder=current,
            device=device,
            seed=args.seed,
            max_steps=args.max_steps,
            planner_depth=args.planner_depth,
            stall_steps=effective_stall,
        )
        exact_before = best_suite["exactExpo"]
        if bool(exact_before["success"]):
            reason = "target-achieved"
            break

        primary, critical_kind = select_primary_critical(snapshots)
        correction_indices = select_correction_indices(
            snapshots,
            primary=primary,
            tail=args.correction_tail,
            max_states=args.max_correction_states,
        )
        sep = separability_report(
            decoder=current,
            student=snapshots,
            teacher_snapshots=teacher_snapshots,
            critical_index=primary,
            device=device,
            k=args.nearest_k,
        )
        if initial_separability is None:
            initial_separability = sep

        print(
            f"\n[round {round_index}] score={trace['score']:.1f} steps={trace['steps']} "
            f"critical={primary} ({critical_kind}) "
            f"{sep['badAction']}->{sep['targetAction']} margin={sep['decoderTargetMinusBadMargin']:.3f}",
            flush=True,
        )

        base_first_unsafe = _first_unsafe_index(snapshots)
        candidate_rows: list[dict[str, Any]] = []
        best_candidate: RecoveryDecoderV2 | None = None
        best_candidate_exact: dict[str, Any] | None = None
        best_candidate_trace: dict[str, Any] | None = None
        best_candidate_training: dict[str, Any] | None = None
        best_candidate_key = _surgery_key(exact_before, base_first_unsafe)

        for recipe_index, recipe in enumerate(recipes):
            candidate, training = train_surgical_candidate(
                base=current,
                student=snapshots,
                teacher_snapshots=teacher_snapshots,
                correction_indices=correction_indices,
                primary_index=primary,
                recipe=recipe,
                teacher_prefix_end=primary + args.teacher_margin,
                device=device,
                seed=args.rng_seed + round_index * 100 + recipe_index,
            )
            exact = evaluate_variant(
                policy=policy,
                decoder=candidate,
                device=device,
                seed=args.seed,
                phase_offset=0.0,
                initial_column=0,
                max_steps=args.max_steps,
                target_score=target_score,
                stall_steps=effective_stall,
            )
            _, cand_snapshots, cand_trace = collect_student_trace(
                policy=policy,
                decoder=candidate,
                device=device,
                seed=args.seed,
                max_steps=args.max_steps,
                planner_depth=args.planner_depth,
                stall_steps=effective_stall,
            )
            cand_first_unsafe = _first_unsafe_index(cand_snapshots)
            key = _surgery_key(exact, cand_first_unsafe)
            prefix_ok = training["prefixActionAgreementWithBase"] >= args.min_prefix_agreement
            improves = key > best_candidate_key and prefix_ok
            row = {
                "recipe": recipe.__dict__,
                "training": training,
                "exactExpo": exact,
                "studentTrace": cand_trace,
                "firstUnsafeDecision": cand_first_unsafe,
                "prefixPreserved": bool(prefix_ok),
                "improvesCurrent": bool(improves),
            }
            candidate_rows.append(row)
            print(
                f"  {recipe.name:12s} score={exact['score']:.1f} steps={exact['steps']} "
                f"firstUnsafe={cand_first_unsafe} prefixAgree={training['prefixActionAgreementWithBase']:.3f} "
                f"{'PROMISING' if improves else 'reject'}",
                flush=True,
            )
            if improves:
                best_candidate_key = key
                best_candidate = candidate
                best_candidate_exact = exact
                best_candidate_trace = cand_trace
                best_candidate_training = training

        round_report: dict[str, Any] = {
            "round": int(round_index),
            "before": trace,
            "criticalKind": critical_kind,
            "criticalIndex": int(primary),
            "correctionIndices": [int(i) for i in correction_indices],
            "separability": sep,
            "candidates": candidate_rows,
            "accepted": False,
        }

        if best_candidate is None:
            reason = "no-surgical-improvement"
            rounds.append(round_report)
            print("  no surgical candidate improved the exact seed without prefix regression.", flush=True)
            break

        suite = evaluate_suite(
            policy=policy,
            decoder=best_candidate,
            device=device,
            seed=args.seed,
            max_steps=args.max_steps,
            target_score=target_score,
            stall_steps=effective_stall,
            smoke=args.smoke,
        )
        current = clone_decoder(best_candidate, device)
        best_suite = suite
        best_stage = f"critical-surgery-round-{round_index}"
        round_report.update(
            {
                "accepted": True,
                "acceptedTraining": best_candidate_training,
                "acceptedStudentTrace": best_candidate_trace,
                "acceptedClosedLoop": suite,
            }
        )
        rounds.append(round_report)
        save_decoder(
            args.out / "best_decoder.pt",
            decoder=current,
            source_checkpoint=args.checkpoint,
            stage=best_stage,
            metrics=best_suite,
            config=config,
        )
        exact_after = suite["exactExpo"]
        robust = suite["robust"]
        print(
            f"  ACCEPT: exact score={exact_after['score']:.1f} steps={exact_after['steps']} "
            f"robustMean={robust['meanScore']:.1f}",
            flush=True,
        )
        if bool(exact_after["success"]):
            reason = "target-achieved"
            break

    final_exact = best_suite["exactExpo"]
    if bool(final_exact["success"]):
        next_action = "use-surgical-decoder-and-validate-expo"
    else:
        next_action = "escalate-to-sensory-or-core-training"

    report = {
        "version": VERSION,
        "sourceCheckpoint": str(args.checkpoint.resolve()),
        "sourceDecoder": str(args.decoder_checkpoint.resolve()),
        "metadata": metadata,
        "decoderMetadata": decoder_metadata,
        "config": config,
        "frozenMaleCNS": True,
        "teacherReference": teacher_report,
        "baseline": baseline_suite,
        "initialSeparability": initial_separability,
        "rounds": rounds,
        "selectedStage": best_stage,
        "selectedClosedLoop": best_suite,
        "bestDecoder": str((args.out / "best_decoder.pt").resolve()),
        "stopReason": reason,
        "nextAction": next_action,
        "elapsedSeconds": time.perf_counter() - started,
        "interpretation": (
            "This is intentionally the final decoder-side diagnostic/intervention. "
            "MaleCNS remains frozen. Only 325 parameters in the existing MLP64 residual "
            "output layer may change, under prefix-logit distillation. If this cannot "
            "improve the deterministic expo seed, the report escalates directly to "
            "sensory/core work rather than asking for another readout audit."
        ),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== FINAL ===", flush=True)
    print(
        f"stage={best_stage} score={final_exact['score']:.1f} steps={final_exact['steps']} "
        f"success={int(final_exact['success'])}",
        flush=True,
    )
    print(f"stopReason={reason}", flush=True)
    print(f"nextAction={next_action}", flush=True)
    print(f"report: {(args.out / 'report.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
