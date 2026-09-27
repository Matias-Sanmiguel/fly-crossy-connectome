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
from fly_crossy.v2.preference_distill import (
    pairwise_targets,
    planner_action_preferences,
    primary_acceptable_mask,
)
from fly_crossy.v4.exact_seed_branch_curriculum import (
    build_exact_teacher_trace,
    finetune_decoder,
    load_decoder_checkpoint,
)
from fly_crossy.v4.recovery_dagger_v2 import (
    ACTION_COUNT,
    ACTION_NAMES,
    RecoveryDataV2,
    RecoveryDecoderV2,
    _immediate_progress_mask,
    _to_data,
    action_metrics,
    clone_decoder,
    concat_data,
    evaluate_suite,
    load_frozen_policy,
    repo_root,
    save_decoder,
)
from fly_crossy.v4.train_expo_specialist import (
    EXPO_SEED,
    default_flyhard_root,
    resize_rgb,
    semantic_target,
    variant_state,
)


VERSION = "crossy-v4-student-state-frontier-curriculum-1"


@dataclass
class StudentSnapshot:
    index: int
    state: GameState
    neural_after_frame: torch.Tensor
    motor: np.ndarray
    values: np.ndarray
    acceptable: np.ndarray
    safe: np.ndarray
    progress: np.ndarray
    pair_sign: np.ndarray
    pair_weight: np.ndarray
    best_action: int
    student_action: int
    semantic: np.ndarray


def _subset_data(data: RecoveryDataV2, end: int) -> RecoveryDataV2:
    end = max(1, min(int(end), data.length))
    sl = slice(0, end)
    return RecoveryDataV2(
        motor=data.motor[sl].copy(),
        acceptable=data.acceptable[sl].copy(),
        safe=data.safe[sl].copy(),
        progress=data.progress[sl].copy(),
        pair_sign=data.pair_sign[sl].copy(),
        pair_weight=data.pair_weight[sl].copy(),
        best_action=data.best_action[sl].copy(),
        semantic=data.semantic[sl].copy(),
        behavior_action=data.behavior_action[sl].copy(),
        recovery=data.recovery[sl].copy(),
        source=data.source[sl].copy(),
    )


def _snapshot_row(snapshot: StudentSnapshot, *, source: str, recovery: bool) -> dict[str, Any]:
    return {
        "motor": snapshot.motor.copy(),
        "acceptable": snapshot.acceptable.copy(),
        "safe": snapshot.safe.copy(),
        "progress": snapshot.progress.copy(),
        "pair_sign": snapshot.pair_sign.copy(),
        "pair_weight": snapshot.pair_weight.copy(),
        "best_action": int(snapshot.best_action),
        "semantic": snapshot.semantic.copy(),
        "behavior_action": int(snapshot.student_action),
        "recovery": bool(recovery),
        "source": source,
    }


def _quality_from_snapshots(snapshots: list[StudentSnapshot]) -> dict[str, Any]:
    if not snapshots:
        return {
            "frames": 0,
            "exactActionRate": 0.0,
            "acceptableActionRate": 0.0,
            "immediateSafeActionRate": 0.0,
            "progressOpportunityFrames": 0,
            "progressChoiceRate": 1.0,
            "firstUnacceptable": None,
            "firstUnsafe": None,
        }
    student = np.asarray([row.student_action for row in snapshots], dtype=np.int64)
    best = np.asarray([row.best_action for row in snapshots], dtype=np.int64)
    acceptable = np.stack([row.acceptable for row in snapshots])
    safe = np.stack([row.safe for row in snapshots])
    progress = np.stack([row.progress for row in snapshots])
    rows = np.arange(len(snapshots))
    acc = acceptable[rows, student]
    safe_choice = safe[rows, student]
    preferred_progress = acceptable & progress
    eligible = preferred_progress.any(axis=1)
    chosen_progress = np.zeros(len(snapshots), dtype=np.bool_)
    chosen_progress[eligible] = preferred_progress[rows[eligible], student[eligible]]

    def compact(index: int | None) -> dict[str, Any] | None:
        if index is None:
            return None
        snap = snapshots[index]
        return {
            "decision": int(snap.index),
            "gameStep": int(snap.state.step),
            "score": float(snap.state.score),
            "row": int(snap.state.fly.row),
            "column": float(snap.state.fly.column),
            "teacherAction": ACTION_NAMES[snap.best_action],
            "studentAction": ACTION_NAMES[snap.student_action],
            "acceptableActions": [ACTION_NAMES[i] for i in np.flatnonzero(snap.acceptable)],
            "safeActions": [ACTION_NAMES[i] for i in np.flatnonzero(snap.safe)],
        }

    bad_acc = np.flatnonzero(~acc)
    bad_safe = np.flatnonzero(~safe_choice)
    return {
        "frames": len(snapshots),
        "exactActionRate": float(np.mean(student == best)),
        "acceptableActionRate": float(np.mean(acc)),
        "immediateSafeActionRate": float(np.mean(safe_choice)),
        "progressOpportunityFrames": int(eligible.sum()),
        "progressChoiceRate": float(chosen_progress[eligible].mean()) if eligible.any() else 1.0,
        "firstUnacceptable": compact(int(bad_acc[0])) if len(bad_acc) else None,
        "firstUnsafe": compact(int(bad_safe[0])) if len(bad_safe) else None,
        "actions": dict(sorted(Counter(ACTION_NAMES[int(i)] for i in student).items())),
    }


@torch.no_grad()
def collect_student_trace(
    *,
    policy,
    decoder: RecoveryDecoderV2,
    device: torch.device,
    seed: str,
    max_steps: int,
    planner_depth: int,
    stall_steps: int,
) -> tuple[RecoveryDataV2, list[StudentSnapshot], dict[str, Any]]:
    """Collect the exact hidden states visited by the current student.

    Unlike teacher-prefix branching, every saved recurrent state below is the
    state actually produced by the student's own preceding trajectory.
    """
    state = variant_state(seed, phase_offset=0.0, initial_column=0)
    neural = policy.zero_state(1, device=device)
    rows: list[dict[str, Any]] = []
    snapshots: list[StudentSnapshot] = []
    no_progress = 0
    max_no_progress = 0
    stalled = False

    for decision in range(max_steps):
        if state.terminal is not None:
            break
        image = torch.as_tensor(
            resize_rgb(render_crossy_neural_frame(state))[None],
            dtype=torch.float32,
            device=device,
        )
        neural = policy.step_state(neural, image)
        motor_t = policy.motor_state(neural)[0]
        values, best_action, safe = planner_action_preferences(state, depth=planner_depth)
        acceptable = primary_acceptable_mask(values)
        pair_sign, pair_weight = pairwise_targets(values)
        progress = _immediate_progress_mask(state)
        student_action = int(decoder(motor_t[None])[0].argmax().item())
        semantic = semantic_target(state)

        snapshot = StudentSnapshot(
            index=decision,
            state=state,
            neural_after_frame=neural.detach().to("cpu", dtype=torch.float32).clone(),
            motor=motor_t.detach().to("cpu", dtype=torch.float32).numpy().copy(),
            values=np.asarray(values).copy(),
            acceptable=acceptable.copy(),
            safe=safe.copy(),
            progress=progress.copy(),
            pair_sign=pair_sign.copy(),
            pair_weight=pair_weight.copy(),
            best_action=int(best_action),
            student_action=student_action,
            semantic=semantic.copy(),
        )
        snapshots.append(snapshot)
        rows.append(
            _snapshot_row(
                snapshot,
                source="student-onpolicy",
                recovery=not bool(acceptable[student_action]),
            )
        )

        before = float(state.score)
        state = step_game(state, ACTION_ORDER[student_action]).state
        if float(state.score) > before:
            no_progress = 0
        else:
            no_progress += 1
            max_no_progress = max(max_no_progress, no_progress)
        if state.terminal is None and no_progress >= stall_steps:
            stalled = True
            break

    data = _to_data(rows)
    quality = _quality_from_snapshots(snapshots)
    return data, snapshots, {
        "steps": len(snapshots),
        "score": float(state.score),
        "terminalReason": "stagnation" if stalled else state.terminal,
        "stalled": bool(stalled),
        "maxNoProgressStreak": int(max_no_progress),
        "decisionQuality": quality,
    }


def select_frontier_roots(
    snapshots: list[StudentSnapshot],
    *,
    tail: int,
    max_roots: int,
) -> list[int]:
    """Prioritize actual student states that matter for the current frontier."""
    if not snapshots or max_roots <= 0:
        return []

    unsafe: list[int] = []
    unacceptable: list[int] = []
    singleton: list[int] = []
    for i, snap in enumerate(snapshots):
        action = snap.student_action
        if not bool(snap.safe[action]):
            unsafe.append(i)
        if not bool(snap.acceptable[action]):
            unacceptable.append(i)
        if int(snap.acceptable.sum()) == 1:
            singleton.append(i)

    tail_start = max(0, len(snapshots) - max(0, int(tail)))
    tail_indices = list(range(tail_start, len(snapshots)))

    ordered: list[int] = []
    seen: set[int] = set()
    # Fatal mistakes and the frontier tail matter most. Then include other
    # wrong/one-choice states and finally sparse coverage of the prefix.
    groups = [unsafe, tail_indices, unacceptable, singleton]
    for group in groups:
        for index in group:
            if index not in seen:
                seen.add(index)
                ordered.append(index)
                if len(ordered) >= max_roots:
                    return sorted(ordered)

    stride = max(1, len(snapshots) // max(1, max_roots))
    for index in range(0, len(snapshots), stride):
        if index not in seen:
            seen.add(index)
            ordered.append(index)
            if len(ordered) >= max_roots:
                break
    return sorted(ordered)


def choose_frontier_branch_actions(snapshot: StudentSnapshot, *, limit: int) -> list[int]:
    """Branch from a real student hidden state using recoverable actions.

    The planner-best action is always tried first. If the student's own action
    was immediately safe, replay it too, because that is the most realistic
    recovery branch. Remaining slots use safe planner-unacceptable actions,
    followed by other safe alternatives.
    """
    if limit <= 0:
        return []
    ordered: list[int] = []

    def add(index: int) -> None:
        if bool(snapshot.safe[index]) and index not in ordered:
            ordered.append(index)

    add(snapshot.best_action)
    add(snapshot.student_action)

    safe_unacceptable = [
        i
        for i in range(ACTION_COUNT)
        if bool(snapshot.safe[i]) and not bool(snapshot.acceptable[i])
    ]
    safe_unacceptable.sort(key=lambda i: (tuple(float(x) for x in snapshot.values[i]), i))
    for index in safe_unacceptable:
        add(index)
    for index in range(ACTION_COUNT):
        add(index)
    return ordered[:limit]


@torch.no_grad()
def collect_frontier_branches(
    *,
    policy,
    snapshots: list[StudentSnapshot],
    root_indices: list[int],
    device: torch.device,
    planner_depth: int,
    branches_per_root: int,
    branch_horizon: int,
) -> tuple[RecoveryDataV2 | None, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    terminals: Counter[str] = Counter()
    branch_actions: Counter[str] = Counter()
    branch_lengths: list[int] = []
    end_scores: list[float] = []
    unsafe_roots = 0
    unacceptable_roots = 0
    branches = 0

    for root_index in root_indices:
        snap = snapshots[root_index]
        if not bool(snap.safe[snap.student_action]):
            unsafe_roots += 1
        if not bool(snap.acceptable[snap.student_action]):
            unacceptable_roots += 1

        # Duplicate the true student root at high priority. The behavior_action
        # lets the existing preference weights strongly punish bad decisions.
        rows.append(_snapshot_row(snap, source="student-frontier-root", recovery=True))

        choices = choose_frontier_branch_actions(snap, limit=branches_per_root)
        for branch_action in choices:
            branches += 1
            branch_actions[ACTION_NAMES[branch_action]] += 1
            state = step_game(snap.state, ACTION_ORDER[branch_action]).state
            neural = snap.neural_after_frame.to(device=device)
            length = 1

            for _ in range(branch_horizon):
                if state.terminal is not None:
                    break
                image = torch.as_tensor(
                    resize_rgb(render_crossy_neural_frame(state))[None],
                    dtype=torch.float32,
                    device=device,
                )
                neural = policy.step_state(neural, image)
                motor = policy.motor_state(neural)[0]
                values, best_action, safe = planner_action_preferences(state, depth=planner_depth)
                acceptable = primary_acceptable_mask(values)
                pair_sign, pair_weight = pairwise_targets(values)
                progress = _immediate_progress_mask(state)
                rows.append(
                    {
                        "motor": motor.detach().to("cpu", dtype=torch.float32).numpy().copy(),
                        "acceptable": acceptable,
                        "safe": safe,
                        "progress": progress,
                        "pair_sign": pair_sign,
                        "pair_weight": pair_weight,
                        "best_action": int(best_action),
                        "semantic": semantic_target(state),
                        "behavior_action": int(best_action),
                        "recovery": True,
                        "source": "student-frontier-recovery",
                    }
                )
                state = step_game(state, ACTION_ORDER[best_action]).state
                length += 1

            if state.terminal is not None:
                terminals[str(state.terminal)] += 1
            branch_lengths.append(length)
            end_scores.append(float(state.score))

    data = _to_data(rows) if rows else None
    return data, {
        "rootIndices": [int(i) for i in root_indices],
        "roots": len(root_indices),
        "unsafeRoots": int(unsafe_roots),
        "unacceptableRoots": int(unacceptable_roots),
        "branches": int(branches),
        "frames": 0 if data is None else data.length,
        "branchesPerRoot": int(branches_per_root),
        "branchHorizon": int(branch_horizon),
        "branchActions": dict(sorted(branch_actions.items())),
        "terminalReasons": dict(sorted(terminals.items())),
        "meanBranchLength": float(np.mean(branch_lengths)) if branch_lengths else 0.0,
        "meanEndScore": float(np.mean(end_scores)) if end_scores else 0.0,
    }


def frontier_selection_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    exact = metrics["exactExpo"]
    robust = metrics["robust"]
    return (
        float(exact["success"]),
        float(exact["score"]),
        float(exact["steps"]),
        float(robust["successes"]),
        float(robust["meanScore"]),
        -float(robust["stalled"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Student-state frontier curriculum: freeze MaleCNS, collect the recurrent "
            "states actually visited by the best student, label that frontier with the "
            "planner, branch recovery from those exact hidden states, and only accept "
            "decoder updates that improve the deterministic expo seed."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=EXPO_SEED)
    parser.add_argument("--rng-seed", type=int, default=808)
    parser.add_argument("--planner-depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--frontier-tail", type=int, default=18)
    parser.add_argument("--max-roots", type=int, default=32)
    parser.add_argument("--branches-per-root", type=int, default=3)
    parser.add_argument("--branch-horizon", type=int, default=10)
    parser.add_argument("--teacher-margin", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--decoder-lr", type=float, default=0.0015)
    parser.add_argument("--decoder-batch", type=int, default=1024)
    parser.add_argument("--progress-loss-weight", type=float, default=1.0)
    parser.add_argument("--stall-steps", type=int, default=12)
    parser.add_argument("--min-teacher-fraction", type=float, default=0.80)
    parser.add_argument("--plateau-rounds", type=int, default=2)
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
        default=repo_root() / "runs" / "crossy-v4-student-state-frontier-curriculum",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_steps = min(args.max_steps, 40)
        args.rounds = min(args.rounds, 2)
        args.frontier_tail = min(args.frontier_tail, 8)
        args.max_roots = min(args.max_roots, 10)
        args.branches_per_root = min(args.branches_per_root, 2)
        args.branch_horizon = min(args.branch_horizon, 4)
        args.teacher_margin = min(args.teacher_margin, 8)
        args.epochs = min(args.epochs, 8)
        args.plateau_rounds = 1
        args.stall_steps = min(args.stall_steps, 8)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    if args.rounds < 1 or args.max_roots < 1 or args.branches_per_root < 1:
        raise SystemExit("rounds, max-roots, and branches-per-root must be >= 1")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.rng_seed)
    started = time.perf_counter()

    policy, metadata = load_frozen_policy(
        checkpoint_path=args.checkpoint,
        flyhard_root=args.flyhard_root,
        device=device,
    )
    starting_decoder, decoder_metadata = load_decoder_checkpoint(
        args.decoder_checkpoint,
        device=device,
    )

    print("=== CROSSY V4 / STUDENT-STATE FRONTIER CURRICULUM ===", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(
        f"MaleCNS frozen: {metadata['neurons']:,} neurons / {metadata['edges']:,} edges",
        flush=True,
    )
    print(
        f"starting decoder: {decoder_metadata['stage']} / {starting_decoder.architecture}",
        flush=True,
    )

    print("\n[1] teacher reference...", flush=True)
    teacher, _, teacher_report = build_exact_teacher_trace(
        policy=policy,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        planner_depth=args.planner_depth,
    )
    teacher_score = float(teacher_report["score"])
    target_score = teacher_score * float(args.min_teacher_fraction)
    effective_stall_steps = max(int(args.stall_steps), int(teacher_report["maxNoProgressStreak"]) + 2)
    print(
        f"  teacher score={teacher_score:.1f} target={target_score:.1f} "
        f"stallCutoff={effective_stall_steps}",
        flush=True,
    )

    best_decoder = clone_decoder(starting_decoder, device)
    baseline = evaluate_suite(
        policy=policy,
        decoder=best_decoder,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        target_score=target_score,
        stall_steps=effective_stall_steps,
        smoke=args.smoke,
    )
    best_metrics = baseline
    best_stage = "branch-curriculum-baseline"
    accepted_replay_parts: list[RecoveryDataV2] = []
    rounds: list[dict[str, Any]] = []
    plateau = 0

    config = {
        "seed": args.seed,
        "rngSeed": args.rng_seed,
        "plannerDepth": args.planner_depth,
        "maxSteps": args.max_steps,
        "rounds": args.rounds,
        "frontierTail": args.frontier_tail,
        "maxRoots": args.max_roots,
        "branchesPerRoot": args.branches_per_root,
        "branchHorizon": args.branch_horizon,
        "teacherMargin": args.teacher_margin,
        "epochs": args.epochs,
        "decoderLearningRate": args.decoder_lr,
        "decoderBatch": args.decoder_batch,
        "progressLossWeight": args.progress_loss_weight,
        "minimumTeacherFraction": args.min_teacher_fraction,
        "teacherScore": teacher_score,
        "minimumProgressScore": target_score,
        "effectiveStallSteps": effective_stall_steps,
        "MaleCNSRetrained": False,
        "startingDecoder": str(args.decoder_checkpoint.resolve()),
        "selectionRule": "exact success -> exact score -> exact steps -> robust successes -> robust score",
    }

    save_decoder(
        args.out / "best_decoder.pt",
        decoder=best_decoder,
        source_checkpoint=args.checkpoint,
        stage=best_stage,
        metrics=best_metrics,
        config=config,
    )

    exact = baseline["exactExpo"]
    print(
        f"  baseline: score={exact['score']:.1f} steps={exact['steps']}/{args.max_steps} "
        f"robustMean={baseline['robust']['meanScore']:.1f}",
        flush=True,
    )

    for round_index in range(1, args.rounds + 1):
        print(f"\n[round {round_index}/{args.rounds}] collect REAL student hidden states...", flush=True)
        trace_data, snapshots, trace_report = collect_student_trace(
            policy=policy,
            decoder=best_decoder,
            device=device,
            seed=args.seed,
            max_steps=args.max_steps,
            planner_depth=args.planner_depth,
            stall_steps=effective_stall_steps,
        )
        roots = select_frontier_roots(
            snapshots,
            tail=args.frontier_tail,
            max_roots=args.max_roots,
        )
        frontier_data, frontier_report = collect_frontier_branches(
            policy=policy,
            snapshots=snapshots,
            root_indices=roots,
            device=device,
            planner_depth=args.planner_depth,
            branches_per_root=args.branches_per_root,
            branch_horizon=args.branch_horizon,
        )

        teacher_limit = min(teacher.length, max(1, trace_data.length + args.teacher_margin))
        teacher_prefix = _subset_data(teacher, teacher_limit)
        parts = [teacher_prefix, *accepted_replay_parts, trace_data]
        if frontier_data is not None:
            parts.append(frontier_data)
        replay = concat_data(parts)
        print(
            f"  trace score={trace_report['score']:.1f} steps={trace_report['steps']} "
            f"roots={len(roots)} branchFrames={0 if frontier_data is None else frontier_data.length:,} "
            f"replay={replay.length:,}",
            flush=True,
        )
        q = trace_report["decisionQuality"]
        print(
            f"  student quality: acceptable={q['acceptableActionRate']:.3f} "
            f"safe={q['immediateSafeActionRate']:.3f}",
            flush=True,
        )

        candidate, training = finetune_decoder(
            initial=best_decoder,
            data=replay,
            device=device,
            epochs=args.epochs,
            learning_rate=args.decoder_lr,
            batch_size=args.decoder_batch,
            seed=args.rng_seed + round_index * 137,
            progress_loss_weight=args.progress_loss_weight,
        )
        teacher_forced = action_metrics(candidate, teacher_prefix, device)
        closed_loop = evaluate_suite(
            policy=policy,
            decoder=candidate,
            device=device,
            seed=args.seed,
            max_steps=args.max_steps,
            target_score=target_score,
            stall_steps=effective_stall_steps,
            smoke=args.smoke,
        )
        candidate_exact = closed_loop["exactExpo"]
        improved = frontier_selection_key(closed_loop) > frontier_selection_key(best_metrics)
        print(
            f"  candidate: score={candidate_exact['score']:.1f} "
            f"steps={candidate_exact['steps']}/{args.max_steps} "
            f"success={int(candidate_exact['success'])} "
            f"robustMean={closed_loop['robust']['meanScore']:.1f} "
            f"{'NEW BEST' if improved else 'REJECT'}",
            flush=True,
        )

        stage_name = f"student-frontier-round-{round_index}"
        save_decoder(
            args.out / f"round-{round_index}.pt",
            decoder=candidate,
            source_checkpoint=args.checkpoint,
            stage=stage_name,
            metrics=closed_loop,
            config=config,
        )

        if improved:
            best_decoder = clone_decoder(candidate, device)
            best_metrics = closed_loop
            best_stage = stage_name
            accepted_replay_parts.extend([trace_data] + ([frontier_data] if frontier_data is not None else []))
            save_decoder(
                args.out / "best_decoder.pt",
                decoder=best_decoder,
                source_checkpoint=args.checkpoint,
                stage=best_stage,
                metrics=best_metrics,
                config=config,
            )
            plateau = 0
        else:
            plateau += 1

        rounds.append(
            {
                "round": round_index,
                "studentTrace": trace_report,
                "frontierRoots": roots,
                "frontierCollection": frontier_report,
                "teacherPrefixFrames": teacher_limit,
                "acceptedReplayPartsBeforeCandidate": len(accepted_replay_parts) - (2 if improved and frontier_data is not None else (1 if improved else 0)),
                "trainingSamples": replay.length,
                "training": training,
                "teacherForcedPrefix": teacher_forced,
                "closedLoop": closed_loop,
                "selectedGlobally": improved,
            }
        )

        if best_metrics["exactExpo"]["success"]:
            print("  early stop: exact seed reached real-progress target.", flush=True)
            break
        if plateau >= args.plateau_rounds:
            print(f"  early stop: no exact-seed improvement for {plateau} rounds.", flush=True)
            break

    elapsed = time.perf_counter() - started
    report = {
        "version": VERSION,
        "sourceCheckpoint": str(args.checkpoint.resolve()),
        "sourceDecoder": str(args.decoder_checkpoint.resolve()),
        "metadata": metadata,
        "decoderMetadata": decoder_metadata,
        "config": config,
        "frozenMaleCNS": True,
        "teacherTrace": teacher_report,
        "baseline": baseline,
        "rounds": rounds,
        "selectedStage": best_stage,
        "selectedClosedLoop": best_metrics,
        "bestDecoder": str((args.out / "best_decoder.pt").resolve()),
        "elapsedSeconds": elapsed,
        "interpretation": (
            "MaleCNS stayed frozen and fully stateful. Each round collected the actual "
            "student recurrent states on the exact expo seed, strongly supervised the "
            "current failure frontier, and rolled planner recovery branches from those "
            "same hidden states. Deep teacher-prefix branch data was intentionally not "
            "used; teacher anchors only extend a small margin beyond the current frontier."
        ),
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    exact = best_metrics["exactExpo"]
    robust = best_metrics["robust"]
    print("\n=== FINAL ===", flush=True)
    print(f"selected stage: {best_stage}", flush=True)
    print(
        f"exact: score={exact['score']:.1f}/{target_score:.1f} "
        f"steps={exact['steps']}/{args.max_steps} success={int(exact['success'])}",
        flush=True,
    )
    print(
        f"robust: {robust['successes']}/{robust['episodes']} success; "
        f"meanScore={robust['meanScore']:.1f}",
        flush=True,
    )
    print(f"report: {report_path.resolve()}", flush=True)
    print(f"elapsed: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
