from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from fly_crossy.env import GameState, step_game
from fly_crossy.schema import ACTION_ORDER
from fly_crossy.v2.crossy_camera import render_crossy_neural_frame
from fly_crossy.v2.preference_distill import planner_action_preferences, primary_acceptable_mask
from fly_crossy.v4.exact_seed_branch_curriculum import load_decoder_checkpoint
from fly_crossy.v4.recovery_dagger_v2 import (
    ACTION_NAMES,
    RecoveryDecoderV2,
    _immediate_progress_mask,
    load_frozen_policy,
    repo_root,
)
from fly_crossy.v4.train_expo_specialist import (
    EXPO_SEED,
    default_flyhard_root,
    resize_rgb,
    variant_state,
)


VERSION = "crossy-v4-recurrent-state-drift-audit-1"
DEFAULT_MODES = ("stateful", "reset", "short4", "short8", "short16")


@dataclass(frozen=True)
class ModeSpec:
    name: str
    history: int | None


@dataclass
class ReferenceFrame:
    decision: int
    state: GameState
    image: np.ndarray
    stateful_motor: np.ndarray
    best_action: int
    acceptable: np.ndarray
    safe: np.ndarray
    progress: np.ndarray


def parse_modes(value: str) -> list[ModeSpec]:
    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("modes must contain at least one mode")
    result: list[ModeSpec] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        if name == "stateful":
            spec = ModeSpec(name="stateful", history=None)
        elif name == "reset":
            spec = ModeSpec(name="reset", history=1)
        elif name.startswith("short"):
            suffix = name.removeprefix("short")
            if not suffix.isdigit() or int(suffix) < 1:
                raise ValueError(f"Invalid short-history mode: {name}")
            spec = ModeSpec(name=name, history=int(suffix))
        else:
            raise ValueError(f"Unknown mode: {name}")
        seen.add(name)
        result.append(spec)
    return result


def _frame_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(frame[None], dtype=torch.float32, device=device)


@torch.no_grad()
def motor_from_recent_history(
    *,
    policy,
    frames: list[np.ndarray],
    history: int,
    device: torch.device,
) -> np.ndarray:
    if history < 1:
        raise ValueError("history must be >= 1")
    neural = policy.zero_state(1, device=device)
    for frame in frames[-history:]:
        neural = policy.step_state(neural, _frame_tensor(frame, device))
    return policy.motor_state(neural)[0].detach().cpu().numpy().astype(np.float32, copy=True)


def _labels(state: GameState, planner_depth: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    values, best_action, safe = planner_action_preferences(state, depth=planner_depth)
    acceptable = primary_acceptable_mask(values)
    progress = _immediate_progress_mask(state)
    return int(best_action), acceptable, safe, progress


@torch.no_grad()
def collect_reference(
    *,
    policy,
    decoder: RecoveryDecoderV2,
    device: torch.device,
    seed: str,
    max_steps: int,
    planner_depth: int,
    behavior: str,
    stall_steps: int,
) -> tuple[list[ReferenceFrame], dict[str, Any]]:
    if behavior not in {"teacher", "stateful-student"}:
        raise ValueError(f"Unknown behavior: {behavior}")

    state = variant_state(seed, phase_offset=0.0, initial_column=0)
    neural = policy.zero_state(1, device=device)
    frames: list[ReferenceFrame] = []
    actions: Counter[str] = Counter()
    no_progress = 0
    max_no_progress = 0
    stalled = False

    for decision in range(max_steps):
        if state.terminal is not None:
            break
        image = resize_rgb(render_crossy_neural_frame(state)).astype(np.float32, copy=False)
        neural = policy.step_state(neural, _frame_tensor(image, device))
        motor_t = policy.motor_state(neural)[0]
        motor = motor_t.detach().cpu().numpy().astype(np.float32, copy=True)
        best_action, acceptable, safe, progress = _labels(state, planner_depth)
        frames.append(
            ReferenceFrame(
                decision=decision,
                state=state,
                image=image.copy(),
                stateful_motor=motor,
                best_action=best_action,
                acceptable=acceptable.copy(),
                safe=safe.copy(),
                progress=progress.copy(),
            )
        )

        if behavior == "teacher":
            action_index = best_action
        else:
            action_index = int(decoder(motor_t[None])[0].argmax().item())
        action = ACTION_ORDER[action_index]
        actions[action.value] += 1
        before = float(state.score)
        state = step_game(state, action).state
        if float(state.score) > before:
            no_progress = 0
        else:
            no_progress += 1
            max_no_progress = max(max_no_progress, no_progress)
        if behavior == "stateful-student" and state.terminal is None and no_progress >= stall_steps:
            stalled = True
            break

    return frames, {
        "behavior": behavior,
        "frames": len(frames),
        "steps": len(frames),
        "score": float(state.score),
        "terminalReason": "stagnation" if stalled else state.terminal,
        "stalled": bool(stalled),
        "maxNoProgressStreak": int(max_no_progress),
        "actions": dict(sorted(actions.items())),
    }


def _segment_ranges(length: int) -> list[tuple[str, int, int]]:
    if length <= 0:
        return []
    cut1 = max(1, length // 3)
    cut2 = max(cut1 + 1, (2 * length) // 3)
    cut2 = min(cut2, length)
    return [
        ("early", 0, cut1),
        ("middle", cut1, cut2),
        ("late", cut2, length),
    ]


def summarize_predictions(
    *,
    reference: list[ReferenceFrame],
    predictions: np.ndarray,
) -> dict[str, Any]:
    n = len(reference)
    if predictions.shape != (n,):
        raise ValueError(f"predictions shape {predictions.shape} != {(n,)}")
    if n == 0:
        return {
            "frames": 0,
            "exactActionRate": 0.0,
            "acceptableActionRate": 0.0,
            "immediateSafeActionRate": 0.0,
            "progressOpportunityFrames": 0,
            "progressChoiceRate": 1.0,
            "actions": {},
            "firstUnacceptable": None,
            "firstUnsafe": None,
            "segments": {},
        }

    best = np.asarray([row.best_action for row in reference], dtype=np.int64)
    acceptable = np.stack([row.acceptable for row in reference])
    safe = np.stack([row.safe for row in reference])
    progress = np.stack([row.progress for row in reference])
    rows = np.arange(n)
    exact = predictions == best
    acceptable_choice = acceptable[rows, predictions]
    safe_choice = safe[rows, predictions]
    preferred_progress = acceptable & progress
    progress_eligible = preferred_progress.any(axis=1)
    progress_choice = np.zeros(n, dtype=np.bool_)
    progress_choice[progress_eligible] = preferred_progress[
        rows[progress_eligible], predictions[progress_eligible]
    ]

    def compact(index: int | None) -> dict[str, Any] | None:
        if index is None:
            return None
        row = reference[index]
        return {
            "decision": int(row.decision),
            "gameStep": int(row.state.step),
            "score": float(row.state.score),
            "row": int(row.state.fly.row),
            "column": float(row.state.fly.column),
            "teacherAction": ACTION_NAMES[row.best_action],
            "predictedAction": ACTION_NAMES[int(predictions[index])],
            "acceptableActions": [ACTION_NAMES[i] for i in np.flatnonzero(row.acceptable)],
            "safeActions": [ACTION_NAMES[i] for i in np.flatnonzero(row.safe)],
        }

    bad_acc = np.flatnonzero(~acceptable_choice)
    bad_safe = np.flatnonzero(~safe_choice)
    segments: dict[str, Any] = {}
    for name, start, end in _segment_ranges(n):
        if end <= start:
            continue
        idx = np.arange(start, end)
        seg_eligible = progress_eligible[idx]
        segments[name] = {
            "start": int(start),
            "endExclusive": int(end),
            "frames": int(len(idx)),
            "exactActionRate": float(exact[idx].mean()),
            "acceptableActionRate": float(acceptable_choice[idx].mean()),
            "immediateSafeActionRate": float(safe_choice[idx].mean()),
            "progressChoiceRate": (
                float(progress_choice[idx][seg_eligible].mean()) if seg_eligible.any() else 1.0
            ),
        }

    return {
        "frames": n,
        "exactActionRate": float(exact.mean()),
        "acceptableActionRate": float(acceptable_choice.mean()),
        "immediateSafeActionRate": float(safe_choice.mean()),
        "progressOpportunityFrames": int(progress_eligible.sum()),
        "progressChoiceRate": (
            float(progress_choice[progress_eligible].mean()) if progress_eligible.any() else 1.0
        ),
        "actions": dict(sorted(Counter(ACTION_NAMES[int(i)] for i in predictions).items())),
        "firstUnacceptable": compact(int(bad_acc[0])) if len(bad_acc) else None,
        "firstUnsafe": compact(int(bad_safe[0])) if len(bad_safe) else None,
        "segments": segments,
    }


def representation_similarity(stateful: np.ndarray, other: np.ndarray) -> dict[str, float]:
    if stateful.shape != other.shape or stateful.ndim != 2:
        raise ValueError("Expected matching [frames,motor] arrays")
    if len(stateful) == 0:
        return {
            "meanCosineToStateful": 1.0,
            "p10CosineToStateful": 1.0,
            "meanRelativeL2ToStateful": 0.0,
        }
    dot = np.sum(stateful * other, axis=1)
    denom = np.linalg.norm(stateful, axis=1) * np.linalg.norm(other, axis=1)
    cosine = dot / np.maximum(denom, 1e-8)
    rel_l2 = np.linalg.norm(other - stateful, axis=1) / np.maximum(
        np.linalg.norm(stateful, axis=1), 1e-8
    )
    return {
        "meanCosineToStateful": float(np.mean(cosine)),
        "p10CosineToStateful": float(np.quantile(cosine, 0.10)),
        "meanRelativeL2ToStateful": float(np.mean(rel_l2)),
    }


@torch.no_grad()
def evaluate_modes_on_same_frames(
    *,
    policy,
    decoder: RecoveryDecoderV2,
    device: torch.device,
    reference: list[ReferenceFrame],
    modes: list[ModeSpec],
) -> dict[str, Any]:
    stateful_motor = np.asarray([row.stateful_motor for row in reference], dtype=np.float32)
    stateful_pred: np.ndarray | None = None
    result: dict[str, Any] = {}

    for spec in modes:
        motors: list[np.ndarray] = []
        if spec.history is None:
            motors_np = stateful_motor
        else:
            history_frames: list[np.ndarray] = []
            for row in reference:
                history_frames.append(row.image)
                motors.append(
                    motor_from_recent_history(
                        policy=policy,
                        frames=history_frames,
                        history=spec.history,
                        device=device,
                    )
                )
            motors_np = np.asarray(motors, dtype=np.float32)

        motor_t = torch.as_tensor(motors_np, dtype=torch.float32, device=device)
        pred = decoder(motor_t).argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        if spec.name == "stateful":
            stateful_pred = pred.copy()
        metrics = summarize_predictions(reference=reference, predictions=pred)
        metrics["representation"] = representation_similarity(stateful_motor, motors_np)
        result[spec.name] = {
            "historyFrames": spec.history,
            "metrics": metrics,
            "predictions": pred.tolist(),
        }

    if stateful_pred is None:
        stateful_pred = decoder(
            torch.as_tensor(stateful_motor, dtype=torch.float32, device=device)
        ).argmax(dim=1).detach().cpu().numpy().astype(np.int64)
    for spec in modes:
        pred = np.asarray(result[spec.name]["predictions"], dtype=np.int64)
        result[spec.name]["metrics"]["actionAgreementWithStateful"] = float(
            np.mean(pred == stateful_pred)
        ) if len(pred) else 1.0
        del result[spec.name]["predictions"]
    return result


@torch.no_grad()
def evaluate_closed_loop_mode(
    *,
    policy,
    decoder: RecoveryDecoderV2,
    device: torch.device,
    seed: str,
    max_steps: int,
    planner_depth: int,
    target_score: float,
    stall_steps: int,
    mode: ModeSpec,
) -> dict[str, Any]:
    state = variant_state(seed, phase_offset=0.0, initial_column=0)
    neural = policy.zero_state(1, device=device)
    frame_history: list[np.ndarray] = []
    actions: Counter[str] = Counter()
    predictions: list[int] = []
    refs: list[ReferenceFrame] = []
    no_progress = 0
    max_no_progress = 0
    stalled = False

    for decision in range(max_steps):
        if state.terminal is not None:
            break
        image = resize_rgb(render_crossy_neural_frame(state)).astype(np.float32, copy=False)
        frame_history.append(image.copy())

        if mode.history is None:
            neural = policy.step_state(neural, _frame_tensor(image, device))
            motor_t = policy.motor_state(neural)[0]
            motor = motor_t.detach().cpu().numpy().astype(np.float32, copy=True)
        else:
            motor = motor_from_recent_history(
                policy=policy,
                frames=frame_history,
                history=mode.history,
                device=device,
            )
            motor_t = torch.as_tensor(motor, dtype=torch.float32, device=device)

        best_action, acceptable, safe, progress = _labels(state, planner_depth)
        action_index = int(decoder(motor_t[None])[0].argmax().item())
        predictions.append(action_index)
        refs.append(
            ReferenceFrame(
                decision=decision,
                state=state,
                image=image.copy(),
                stateful_motor=motor,
                best_action=best_action,
                acceptable=acceptable.copy(),
                safe=safe.copy(),
                progress=progress.copy(),
            )
        )
        action = ACTION_ORDER[action_index]
        actions[action.value] += 1
        before = float(state.score)
        state = step_game(state, action).state
        if float(state.score) > before:
            no_progress = 0
        else:
            no_progress += 1
            max_no_progress = max(max_no_progress, no_progress)
        if state.terminal is None and no_progress >= stall_steps:
            stalled = True
            break

    reached_limit = bool(state.terminal is None and len(refs) >= max_steps)
    progress_qualified = float(state.score) >= float(target_score)
    decision_metrics = summarize_predictions(
        reference=refs,
        predictions=np.asarray(predictions, dtype=np.int64),
    )
    return {
        "mode": mode.name,
        "historyFrames": mode.history,
        "steps": len(refs),
        "score": float(state.score),
        "targetScore": float(target_score),
        "reachedStepLimit": reached_limit,
        "progressQualified": bool(progress_qualified),
        "success": bool(reached_limit and progress_qualified and not stalled),
        "stalled": bool(stalled),
        "maxNoProgressStreak": int(max_no_progress),
        "terminalReason": "stagnation" if stalled else state.terminal,
        "actions": dict(sorted(actions.items())),
        "decisionQuality": decision_metrics,
    }


def interpretation_hints(
    same_student: dict[str, Any],
    closed_loop: dict[str, Any],
) -> list[str]:
    hints: list[str] = []
    stateful = closed_loop.get("stateful")
    candidates = [row for name, row in closed_loop.items() if name != "stateful"]
    if stateful is not None and candidates:
        best = max(candidates, key=lambda row: (float(row["score"]), int(row["steps"])))
        stateful_acc = float(
            same_student.get("stateful", {}).get("metrics", {}).get("acceptableActionRate", 0.0)
        )
        best_acc = float(
            same_student.get(best["mode"], {}).get("metrics", {}).get("acceptableActionRate", 0.0)
        )
        if float(best["score"]) >= float(stateful["score"]) + 10 or best_acc >= stateful_acc + 0.10:
            hints.append(
                f"{best['mode']} materially beats stateful on score and/or same-frame acceptable actions; long recurrent history is likely hurting the controller."
            )
        elif float(stateful["score"]) >= max(float(row["score"]) for row in candidates):
            hints.append(
                "Stateful remains best in closed loop; the audit does not support simple history truncation as the main fix."
            )
        else:
            hints.append(
                "A truncated-history mode helps somewhat, but not decisively; recurrent drift may contribute without being the only bottleneck."
            )
    reset = closed_loop.get("reset")
    if stateful is not None and reset is not None and float(reset["score"]) > float(stateful["score"]):
        hints.append("Even one-frame reset beats persistent state on score, which is strong evidence of harmful accumulated state.")
    return hints


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether long recurrent MaleCNS history helps or hurts V4 closed-loop control. "
            "Compares persistent state with reset and finite-history reconstructions using the same frozen brain and decoder."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=EXPO_SEED)
    parser.add_argument("--planner-depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--stall-steps", type=int, default=12)
    parser.add_argument("--min-teacher-fraction", type=float, default=0.80)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root() / "runs" / "crossy-v4-expo-specialist" / "best.pt",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        default=repo_root() / "runs" / "crossy-v4-exact-seed-branch-curriculum" / "best_decoder.pt",
    )
    parser.add_argument("--flyhard-root", type=Path, default=default_flyhard_root())
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root() / "runs" / "crossy-v4-recurrent-state-drift-audit",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_steps = min(args.max_steps, 30)
        args.modes = "stateful,reset,short4"
        args.stall_steps = min(args.stall_steps, 8)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    modes = parse_modes(args.modes)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    policy, metadata = load_frozen_policy(
        checkpoint_path=args.checkpoint,
        flyhard_root=args.flyhard_root,
        device=device,
    )
    decoder, decoder_metadata = load_decoder_checkpoint(
        args.decoder_checkpoint,
        device=device,
    )

    print("=== CROSSY V4 / RECURRENT STATE DRIFT AUDIT ===", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(f"modes: {[mode.name for mode in modes]}", flush=True)
    print(
        f"MaleCNS frozen: {metadata['neurons']:,} neurons / {metadata['edges']:,} edges",
        flush=True,
    )
    print(
        f"decoder: {decoder_metadata.get('stage')} / {decoder.architecture} / "
        f"{sum(p.numel() for p in decoder.parameters()):,} params",
        flush=True,
    )

    print("\n[1] teacher reference trajectory...", flush=True)
    teacher_ref, teacher_rollout = collect_reference(
        policy=policy,
        decoder=decoder,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        planner_depth=args.planner_depth,
        behavior="teacher",
        stall_steps=args.stall_steps,
    )
    teacher_score = float(teacher_rollout["score"])
    target_score = float(args.min_teacher_fraction) * teacher_score
    print(
        f"  teacher score={teacher_score:.1f} frames={len(teacher_ref)} target={target_score:.1f}",
        flush=True,
    )

    print("\n[2] same teacher frames under each state mode...", flush=True)
    teacher_same = evaluate_modes_on_same_frames(
        policy=policy,
        decoder=decoder,
        device=device,
        reference=teacher_ref,
        modes=modes,
    )
    for mode in modes:
        m = teacher_same[mode.name]["metrics"]
        print(
            f"  {mode.name:8s} acceptable={m['acceptableActionRate']:.3f} "
            f"safe={m['immediateSafeActionRate']:.3f} exact={m['exactActionRate']:.3f}",
            flush=True,
        )

    print("\n[3] real stateful student trajectory...", flush=True)
    student_ref, student_rollout = collect_reference(
        policy=policy,
        decoder=decoder,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        planner_depth=args.planner_depth,
        behavior="stateful-student",
        stall_steps=args.stall_steps,
    )
    print(
        f"  stateful reference score={student_rollout['score']:.1f} "
        f"steps={student_rollout['steps']} terminal={student_rollout['terminalReason']}",
        flush=True,
    )

    print("\n[4] SAME student frames, reconstructed with each state mode...", flush=True)
    student_same = evaluate_modes_on_same_frames(
        policy=policy,
        decoder=decoder,
        device=device,
        reference=student_ref,
        modes=modes,
    )
    for mode in modes:
        m = student_same[mode.name]["metrics"]
        r = m["representation"]
        print(
            f"  {mode.name:8s} acceptable={m['acceptableActionRate']:.3f} "
            f"safe={m['immediateSafeActionRate']:.3f} exact={m['exactActionRate']:.3f} "
            f"cos={r['meanCosineToStateful']:.3f}",
            flush=True,
        )

    print("\n[5] independent closed-loop rollout per mode...", flush=True)
    closed_loop: dict[str, Any] = {}
    for mode in modes:
        row = evaluate_closed_loop_mode(
            policy=policy,
            decoder=decoder,
            device=device,
            seed=args.seed,
            max_steps=args.max_steps,
            planner_depth=args.planner_depth,
            target_score=target_score,
            stall_steps=args.stall_steps,
            mode=mode,
        )
        closed_loop[mode.name] = row
        q = row["decisionQuality"]
        print(
            f"  {mode.name:8s} score={row['score']:.1f} steps={row['steps']:3d} "
            f"terminal={row['terminalReason']} acceptable={q['acceptableActionRate']:.3f} "
            f"safe={q['immediateSafeActionRate']:.3f}",
            flush=True,
        )

    report = {
        "version": VERSION,
        "sourceCheckpoint": str(args.checkpoint),
        "sourceDecoder": str(args.decoder_checkpoint),
        "metadata": metadata,
        "decoderMetadata": decoder_metadata,
        "config": {
            "seed": args.seed,
            "plannerDepth": int(args.planner_depth),
            "maxSteps": int(args.max_steps),
            "stallSteps": int(args.stall_steps),
            "minimumTeacherFraction": float(args.min_teacher_fraction),
            "teacherScore": teacher_score,
            "minimumProgressScore": target_score,
            "modes": [{"name": mode.name, "historyFrames": mode.history} for mode in modes],
            "MaleCNSRetrained": False,
            "decoderRetrained": False,
        },
        "teacherReference": teacher_rollout,
        "teacherSameFrames": teacher_same,
        "studentStatefulReference": student_rollout,
        "studentSameFrames": student_same,
        "closedLoopByMode": closed_loop,
        "interpretationHints": interpretation_hints(student_same, closed_loop),
        "elapsedSeconds": float(time.perf_counter() - started),
    }
    out_path = args.out / "report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out_path}", flush=True)
    print(f"elapsed: {report['elapsedSeconds']:.1f}s", flush=True)
    for hint in report["interpretationHints"]:
        print(f"hint: {hint}", flush=True)


if __name__ == "__main__":
    main()
