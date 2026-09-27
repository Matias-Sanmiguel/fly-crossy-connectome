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
from torch import nn

from fly_crossy.env import GameState, step_game
from fly_crossy.schema import ACTION_ORDER
from fly_crossy.v2.crossy_camera import render_crossy_neural_frame
from fly_crossy.v2.preference_distill import (
    pairwise_targets,
    planner_action_preferences,
    preference_loss,
    primary_acceptable_mask,
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
    compute_training_weights,
    concat_data,
    evaluate_suite,
    load_frozen_policy,
    progress_preference_loss,
    repo_root,
    save_decoder,
    selection_key,
)
from fly_crossy.v4.train_expo_specialist import (
    EXPO_SEED,
    default_flyhard_root,
    resize_rgb,
    semantic_target,
    variant_state,
)


VERSION = "crossy-v4-exact-seed-branch-curriculum-1"


@dataclass
class PrefixSnapshot:
    index: int
    state: GameState
    neural_after_frame: torch.Tensor
    motor: np.ndarray
    acceptable: np.ndarray
    safe: np.ndarray
    progress: np.ndarray
    pair_sign: np.ndarray
    pair_weight: np.ndarray
    best_action: int
    semantic: np.ndarray


def _parse_stage_ends(value: str, max_steps: int) -> list[int]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        raise ValueError("stage-ends must contain at least one integer")
    result = sorted(set(int(item) for item in raw))
    if any(item <= 0 for item in result):
        raise ValueError("stage-ends must be positive")
    result = [min(item, max_steps) for item in result]
    result = sorted(set(result))
    if result[-1] != max_steps:
        result.append(max_steps)
    return result


def _stage_windows(stage_ends: list[int]) -> list[tuple[int, int]]:
    start = 0
    result: list[tuple[int, int]] = []
    for end in stage_ends:
        if end <= start:
            continue
        result.append((start, end))
        start = end
    return result


def choose_branch_actions(
    *,
    values: np.ndarray,
    acceptable: np.ndarray,
    safe: np.ndarray,
    best_index: int,
    limit: int,
) -> list[int]:
    """Choose deterministic, recoverable deviations from a teacher prefix.

    Safe but planner-unacceptable actions come first. Within each group the
    lower planner value comes first so the curriculum sees meaningful errors,
    not only near-ties. Immediate-fatal actions are never executed as branches.
    """
    if limit <= 0:
        return []
    candidates = [
        index
        for index in range(ACTION_COUNT)
        if index != best_index and bool(safe[index])
    ]

    def key(index: int) -> tuple[int, tuple[float, ...], int]:
        unacceptable_rank = 0 if not bool(acceptable[index]) else 1
        return unacceptable_rank, tuple(float(x) for x in values[index]), index

    candidates.sort(key=key)
    return candidates[:limit]


def _row_from_snapshot(snapshot: PrefixSnapshot, *, behavior_action: int, source: str) -> dict[str, Any]:
    return {
        "motor": snapshot.motor.copy(),
        "acceptable": snapshot.acceptable.copy(),
        "safe": snapshot.safe.copy(),
        "progress": snapshot.progress.copy(),
        "pair_sign": snapshot.pair_sign.copy(),
        "pair_weight": snapshot.pair_weight.copy(),
        "best_action": int(snapshot.best_action),
        "semantic": snapshot.semantic.copy(),
        "behavior_action": int(behavior_action),
        "recovery": True,
        "source": source,
    }


def _label_row(
    *,
    motor: torch.Tensor,
    state: GameState,
    planner_depth: int,
    behavior_action: int,
    source: str,
    recovery: bool,
) -> tuple[dict[str, Any], int, np.ndarray, np.ndarray]:
    values, best_index, safe = planner_action_preferences(state, depth=planner_depth)
    acceptable = primary_acceptable_mask(values)
    pair_sign, pair_weight = pairwise_targets(values)
    progress = _immediate_progress_mask(state)
    row = {
        "motor": motor.detach().to("cpu", dtype=torch.float32).numpy().copy(),
        "acceptable": acceptable,
        "safe": safe,
        "progress": progress,
        "pair_sign": pair_sign,
        "pair_weight": pair_weight,
        "best_action": int(best_index),
        "semantic": semantic_target(state),
        "behavior_action": int(behavior_action),
        "recovery": bool(recovery),
        "source": source,
    }
    return row, int(best_index), values, safe


@torch.no_grad()
def build_exact_teacher_trace(
    *,
    policy,
    device: torch.device,
    seed: str,
    max_steps: int,
    planner_depth: int,
) -> tuple[RecoveryDataV2, list[PrefixSnapshot], dict[str, Any]]:
    state = variant_state(seed, phase_offset=0.0, initial_column=0)
    neural = policy.zero_state(1, device=device)
    rows: list[dict[str, Any]] = []
    snapshots: list[PrefixSnapshot] = []
    actions: Counter[str] = Counter()
    no_progress = 0
    max_no_progress = 0

    for index in range(max_steps):
        if state.terminal is not None:
            break
        image = torch.as_tensor(
            resize_rgb(render_crossy_neural_frame(state))[None],
            dtype=torch.float32,
            device=device,
        )
        neural = policy.step_state(neural, image)
        motor = policy.motor_state(neural)[0]
        values, best_index, safe = planner_action_preferences(state, depth=planner_depth)
        acceptable = primary_acceptable_mask(values)
        pair_sign, pair_weight = pairwise_targets(values)
        progress = _immediate_progress_mask(state)
        semantic = semantic_target(state)

        snapshots.append(
            PrefixSnapshot(
                index=index,
                state=state,
                neural_after_frame=neural.detach().to("cpu", dtype=torch.float32).clone(),
                motor=motor.detach().to("cpu", dtype=torch.float32).numpy().copy(),
                acceptable=acceptable.copy(),
                safe=safe.copy(),
                progress=progress.copy(),
                pair_sign=pair_sign.copy(),
                pair_weight=pair_weight.copy(),
                best_action=int(best_index),
                semantic=semantic.copy(),
            )
        )
        rows.append(
            {
                "motor": motor.detach().to("cpu", dtype=torch.float32).numpy().copy(),
                "acceptable": acceptable,
                "safe": safe,
                "progress": progress,
                "pair_sign": pair_sign,
                "pair_weight": pair_weight,
                "best_action": int(best_index),
                "semantic": semantic,
                "behavior_action": int(best_index),
                "recovery": False,
                "source": "teacher",
            }
        )

        action = ACTION_ORDER[best_index]
        actions[action.value] += 1
        previous_score = float(state.score)
        state = step_game(state, action).state
        if float(state.score) > previous_score:
            no_progress = 0
        else:
            no_progress += 1
            max_no_progress = max(max_no_progress, no_progress)

    data = _to_data(rows)
    return data, snapshots, {
        "frames": data.length,
        "score": float(state.score),
        "terminalReason": state.terminal,
        "reachedStepLimit": bool(state.terminal is None and data.length >= max_steps),
        "maxNoProgressStreak": int(max_no_progress),
        "actions": dict(sorted(actions.items())),
    }


@torch.no_grad()
def collect_branch_window(
    *,
    policy,
    snapshots: list[PrefixSnapshot],
    device: torch.device,
    planner_depth: int,
    root_start: int,
    root_end: int,
    root_stride: int,
    branches_per_root: int,
    branch_horizon: int,
) -> tuple[RecoveryDataV2 | None, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root_actions: Counter[str] = Counter()
    terminals: Counter[str] = Counter()
    branch_lengths: list[int] = []
    end_scores: list[float] = []
    roots_considered = 0
    roots_used = 0
    branches = 0

    capped_end = min(root_end, len(snapshots))
    for root_index in range(root_start, capped_end, max(1, root_stride)):
        roots_considered += 1
        snapshot = snapshots[root_index]
        values, _, _ = planner_action_preferences(snapshot.state, depth=planner_depth)
        branch_actions = choose_branch_actions(
            values=values,
            acceptable=snapshot.acceptable,
            safe=snapshot.safe,
            best_index=snapshot.best_action,
            limit=branches_per_root,
        )
        if not branch_actions:
            continue
        roots_used += 1

        for branch_action in branch_actions:
            branches += 1
            root_actions[ACTION_NAMES[branch_action]] += 1
            rows.append(
                _row_from_snapshot(
                    snapshot,
                    behavior_action=branch_action,
                    source="branch-root",
                )
            )

            state = step_game(snapshot.state, ACTION_ORDER[branch_action]).state
            neural = snapshot.neural_after_frame.to(device=device)
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
                values_now, best_now, safe_now = planner_action_preferences(
                    state, depth=planner_depth
                )
                acceptable_now = primary_acceptable_mask(values_now)
                pair_sign, pair_weight = pairwise_targets(values_now)
                progress_now = _immediate_progress_mask(state)
                rows.append(
                    {
                        "motor": motor.detach().to("cpu", dtype=torch.float32).numpy().copy(),
                        "acceptable": acceptable_now,
                        "safe": safe_now,
                        "progress": progress_now,
                        "pair_sign": pair_sign,
                        "pair_weight": pair_weight,
                        "best_action": int(best_now),
                        "semantic": semantic_target(state),
                        "behavior_action": int(best_now),
                        "recovery": True,
                        "source": "branch-recovery",
                    }
                )
                state = step_game(state, ACTION_ORDER[best_now]).state
                length += 1

            if state.terminal is not None:
                terminals[str(state.terminal)] += 1
            branch_lengths.append(length)
            end_scores.append(float(state.score))

    data = _to_data(rows) if rows else None
    return data, {
        "rootStart": int(root_start),
        "rootEnd": int(capped_end),
        "rootStride": int(root_stride),
        "rootsConsidered": int(roots_considered),
        "rootsUsed": int(roots_used),
        "branches": int(branches),
        "frames": int(0 if data is None else data.length),
        "branchesPerRootLimit": int(branches_per_root),
        "branchHorizon": int(branch_horizon),
        "rootActions": dict(sorted(root_actions.items())),
        "terminalReasons": dict(sorted(terminals.items())),
        "meanBranchLength": float(np.mean(branch_lengths)) if branch_lengths else 0.0,
        "meanEndScore": float(np.mean(end_scores)) if end_scores else 0.0,
    }


def load_decoder_checkpoint(path: Path, *, device: torch.device) -> tuple[RecoveryDecoderV2, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Recovery V2 decoder not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    motor_dim = int(payload["motorDim"])
    architecture = str(payload["architecture"])
    decoder = RecoveryDecoderV2(
        motor_dim,
        mean=torch.zeros(motor_dim, dtype=torch.float32, device=device),
        std=torch.ones(motor_dim, dtype=torch.float32, device=device),
        architecture=architecture,
    ).to(device)
    decoder.load_state_dict(payload["decoder"], strict=True)
    decoder.eval()
    return decoder, {
        "stage": payload.get("stage"),
        "architecture": architecture,
        "motorDim": motor_dim,
        "metrics": payload.get("metrics"),
        "config": payload.get("config"),
    }


def curriculum_training_weights(data: RecoveryDataV2) -> tuple[np.ndarray, dict[str, Any]]:
    weights, report = compute_training_weights(data)
    source_as_text = data.source.astype(str)
    branch = np.char.startswith(source_as_text, "branch")
    weights = weights * np.where(branch, 1.5, 1.0).astype(np.float32)
    weights = np.clip(weights, 0.40, 20.0).astype(np.float32)
    report = dict(report)
    report.update(
        {
            "branchFrames": int(branch.sum()),
            "branchMultiplier": 1.5,
            "meanWeightAfterBranchBoost": float(weights.mean()),
            "maxWeightAfterBranchBoost": float(weights.max()),
        }
    )
    return weights, report


def finetune_decoder(
    *,
    initial: RecoveryDecoderV2,
    data: RecoveryDataV2,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    progress_loss_weight: float,
) -> tuple[RecoveryDecoderV2, dict[str, Any]]:
    decoder = clone_decoder(initial, device)
    decoder.train()
    weights_np, weight_report = curriculum_training_weights(data)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    x = torch.as_tensor(data.motor, dtype=torch.float32, device=device)
    acceptable = torch.as_tensor(data.acceptable, dtype=torch.bool, device=device)
    safe = torch.as_tensor(data.safe, dtype=torch.bool, device=device)
    progress = torch.as_tensor(data.progress, dtype=torch.bool, device=device)
    pair_sign = torch.as_tensor(data.pair_sign, dtype=torch.float32, device=device)
    pair_weight = torch.as_tensor(data.pair_weight, dtype=torch.float32, device=device)
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)

    history: list[dict[str, float]] = []
    indices = np.arange(data.length)
    for epoch in range(1, epochs + 1):
        shuffled = rng.permutation(indices)
        totals: list[float] = []
        prefs: list[float] = []
        progresses: list[float] = []
        for start in range(0, data.length, batch_size):
            batch_np = shuffled[start : start + batch_size]
            batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = decoder(x[batch])
            pref_loss, _ = preference_loss(
                logits,
                acceptable[batch],
                safe[batch],
                pair_sign[batch],
                pair_weight[batch],
                weights[batch],
            )
            prog_loss = progress_preference_loss(
                logits,
                acceptable[batch],
                progress[batch],
                weights[batch],
            )
            loss = pref_loss + progress_loss_weight * prog_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite exact-seed branch curriculum loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()
            totals.append(float(loss.detach()))
            prefs.append(float(pref_loss.detach()))
            progresses.append(float(prog_loss.detach()))

        record = {
            "total": float(np.mean(totals)),
            "preference": float(np.mean(prefs)),
            "progress": float(np.mean(progresses)),
        }
        history.append(record)
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"    epoch {epoch:3d}/{epochs} loss={record['total']:.4f} "
                f"pref={record['preference']:.4f} progress={record['progress']:.4f}",
                flush=True,
            )

    decoder.eval()
    return decoder, {
        "architecture": decoder.architecture,
        "trainableParameters": sum(p.numel() for p in decoder.parameters()),
        "epochs": int(epochs),
        "learningRate": float(learning_rate),
        "batchSize": int(batch_size),
        "progressLossWeight": float(progress_loss_weight),
        "first": history[0],
        "last": history[-1],
        "weights": weight_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exact-seed branch curriculum: freeze the trained MaleCNS, start from "
            "Recovery DAgger V2's MLP readout, and teach recovery branches from "
            "teacher prefixes throughout the deterministic expo seed."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=EXPO_SEED)
    parser.add_argument("--rng-seed", type=int, default=707)
    parser.add_argument("--planner-depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--stage-ends", default="50,100,150,200")
    parser.add_argument("--root-stride", type=int, default=1)
    parser.add_argument("--branches-per-root", type=int, default=2)
    parser.add_argument("--branch-horizon", type=int, default=12)
    parser.add_argument("--epochs-per-stage", type=int, default=80)
    parser.add_argument("--decoder-lr", type=float, default=0.003)
    parser.add_argument("--decoder-batch", type=int, default=1024)
    parser.add_argument("--progress-loss-weight", type=float, default=1.0)
    parser.add_argument("--stall-steps", type=int, default=12)
    parser.add_argument("--min-teacher-fraction", type=float, default=0.80)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root() / "runs" / "crossy-v4-expo-specialist" / "best.pt",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        default=repo_root() / "runs" / "crossy-v4-recovery-dagger-v2" / "best_decoder.pt",
    )
    parser.add_argument("--flyhard-root", type=Path, default=default_flyhard_root())
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root() / "runs" / "crossy-v4-exact-seed-branch-curriculum",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_steps = min(args.max_steps, 40)
        args.stage_ends = "20,40"
        args.root_stride = 2
        args.branches_per_root = 1
        args.branch_horizon = 4
        args.epochs_per_stage = 8
        args.stall_steps = min(args.stall_steps, 8)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    if args.branches_per_root < 1:
        raise SystemExit("branches-per-root must be >= 1")
    if args.branch_horizon < 1:
        raise SystemExit("branch-horizon must be >= 1")

    stage_ends = _parse_stage_ends(args.stage_ends, args.max_steps)
    windows = _stage_windows(stage_ends)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.rng_seed)
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
    if decoder.architecture != "mlp64":
        print(
            f"WARNING: starting decoder architecture is {decoder.architecture}, not mlp64.",
            flush=True,
        )

    print("=== CROSSY V4 / EXACT-SEED BRANCH CURRICULUM ===", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(
        f"MaleCNS frozen: {metadata['neurons']:,} neurons / {metadata['edges']:,} edges",
        flush=True,
    )
    print(
        f"starting decoder: {decoder_metadata['stage']} / {decoder.architecture} / "
        f"{sum(p.numel() for p in decoder.parameters()):,} params",
        flush=True,
    )
    print(f"curriculum windows: {windows}", flush=True)

    print("\n[1] exact teacher trace + recurrent prefix snapshots...", flush=True)
    teacher, snapshots, teacher_report = build_exact_teacher_trace(
        policy=policy,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        planner_depth=args.planner_depth,
    )
    teacher_score = float(teacher_report["score"])
    target_score = float(args.min_teacher_fraction) * teacher_score
    effective_stall_steps = max(
        int(args.stall_steps), int(teacher_report["maxNoProgressStreak"]) + 2
    )
    print(
        f"  teacher: score={teacher_score:.1f} frames={teacher.length} "
        f"target={target_score:.1f} stallCutoff={effective_stall_steps}",
        flush=True,
    )

    print("\n[2] starting V2 decoder baseline...", flush=True)
    baseline_teacher_forced = action_metrics(decoder, teacher, device)
    baseline_closed_loop = evaluate_suite(
        policy=policy,
        decoder=decoder,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        target_score=target_score,
        stall_steps=effective_stall_steps,
        smoke=args.smoke,
    )
    base_exact = baseline_closed_loop["exactExpo"]
    print(
        f"  exactScore={base_exact['score']:.1f} steps={base_exact['steps']}/{args.max_steps} "
        f"success={int(base_exact['success'])} "
        f"TFacc={baseline_teacher_forced['acceptableActionRate']:.3f}",
        flush=True,
    )

    best_decoder = clone_decoder(decoder, device)
    best_metrics = baseline_closed_loop
    best_stage = "recovery-v2-baseline"
    replay_parts: list[RecoveryDataV2] = [teacher]
    stages: list[dict[str, Any]] = []

    config = {
        "seed": args.seed,
        "rngSeed": args.rng_seed,
        "plannerDepth": args.planner_depth,
        "maxSteps": args.max_steps,
        "stageEnds": stage_ends,
        "rootStride": args.root_stride,
        "branchesPerRoot": args.branches_per_root,
        "branchHorizon": args.branch_horizon,
        "epochsPerStage": args.epochs_per_stage,
        "decoderLearningRate": args.decoder_lr,
        "decoderBatch": args.decoder_batch,
        "progressLossWeight": args.progress_loss_weight,
        "minimumTeacherFraction": args.min_teacher_fraction,
        "teacherScore": teacher_score,
        "minimumProgressScore": target_score,
        "effectiveStallSteps": effective_stall_steps,
        "MaleCNSRetrained": False,
        "startingDecoder": str(args.decoder_checkpoint.resolve()),
        "selectionRule": "exact real-progress success -> exact score -> robust successes -> robust score",
    }

    save_decoder(
        args.out / "best_decoder.pt",
        decoder=best_decoder,
        source_checkpoint=args.checkpoint,
        stage=best_stage,
        metrics=best_metrics,
        config=config,
    )

    for stage_index, (root_start, root_end) in enumerate(windows, start=1):
        print(
            f"\n[stage {stage_index}/{len(windows)}] branch roots {root_start}:{root_end}...",
            flush=True,
        )
        branch_data, branch_report = collect_branch_window(
            policy=policy,
            snapshots=snapshots,
            device=device,
            planner_depth=args.planner_depth,
            root_start=root_start,
            root_end=root_end,
            root_stride=args.root_stride,
            branches_per_root=args.branches_per_root,
            branch_horizon=args.branch_horizon,
        )
        if branch_data is not None:
            replay_parts.append(branch_data)
        replay = concat_data(replay_parts)
        print(
            f"  branches={branch_report['branches']} frames={branch_report['frames']:,} "
            f"cumulativeReplay={replay.length:,}",
            flush=True,
        )

        candidate, training = finetune_decoder(
            initial=best_decoder,
            data=replay,
            device=device,
            epochs=args.epochs_per_stage,
            learning_rate=args.decoder_lr,
            batch_size=args.decoder_batch,
            seed=args.rng_seed + stage_index * 101,
            progress_loss_weight=args.progress_loss_weight,
        )
        teacher_forced = action_metrics(candidate, teacher, device)
        branch_fit = action_metrics(candidate, branch_data, device) if branch_data is not None else None
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
        exact = closed_loop["exactExpo"]
        robust = closed_loop["robust"]
        improved = selection_key(closed_loop) > selection_key(best_metrics)
        print(
            f"  exactScore={exact['score']:.1f} steps={exact['steps']}/{args.max_steps} "
            f"success={int(exact['success'])} robust={robust['successes']}/{robust['episodes']} "
            f"mean={robust['meanScore']:.1f} TFacc={teacher_forced['acceptableActionRate']:.3f} "
            f"{'NEW BEST' if improved else 'keep best'}",
            flush=True,
        )

        stage_name = f"branch-stage-{stage_index}-through-{root_end}"
        save_decoder(
            args.out / f"stage-{stage_index}.pt",
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
            save_decoder(
                args.out / "best_decoder.pt",
                decoder=best_decoder,
                source_checkpoint=args.checkpoint,
                stage=best_stage,
                metrics=best_metrics,
                config=config,
            )

        stages.append(
            {
                "stage": stage_index,
                "rootWindow": [root_start, root_end],
                "branchCollection": branch_report,
                "cumulativeReplaySamples": replay.length,
                "training": training,
                "teacherForcedExact": teacher_forced,
                "branchFit": branch_fit,
                "closedLoop": closed_loop,
                "selectedGlobally": improved,
            }
        )

        if best_metrics["exactExpo"]["success"]:
            print("  early stop: exact deterministic expo seed reached real-progress target.", flush=True)
            break

    elapsed = time.perf_counter() - started
    final_teacher_forced = action_metrics(best_decoder, teacher, device)
    report = {
        "version": VERSION,
        "sourceCheckpoint": str(args.checkpoint.resolve()),
        "sourceDecoder": str(args.decoder_checkpoint.resolve()),
        "metadata": metadata,
        "decoderMetadata": decoder_metadata,
        "config": config,
        "frozenMaleCNS": True,
        "teacherTrace": teacher_report,
        "baseline": {
            "teacherForcedExact": baseline_teacher_forced,
            "closedLoop": baseline_closed_loop,
        },
        "stages": stages,
        "selectedStage": best_stage,
        "selectedTeacherForcedExact": final_teacher_forced,
        "selectedClosedLoop": best_metrics,
        "bestDecoder": str((args.out / "best_decoder.pt").resolve()),
        "elapsedSeconds": elapsed,
        "interpretation": (
            "The MaleCNS stayed frozen. Training started from Recovery DAgger V2's "
            "selected decoder and added exact-seed recovery branches from teacher "
            "prefixes in progressively deeper curriculum windows. A branch executes "
            "an immediate-safe deviation, then the planner replans recovery while the "
            "MaleCNS recurrent state continues from the real teacher prefix state."
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
        f"robust: {robust['successes']}/{robust['episodes']} real-progress successes; "
        f"meanScore={robust['meanScore']:.1f}",
        flush=True,
    )
    print(f"report: {report_path.resolve()}", flush=True)
    print(f"elapsed: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
