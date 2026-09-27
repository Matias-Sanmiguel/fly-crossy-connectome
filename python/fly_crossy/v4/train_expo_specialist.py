from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import shutil
import time
from typing import Iterable

import numpy as np
import pyarrow.feather as feather
import torch
from torch import nn
import torch.nn.functional as F

from fly_crossy.env import (
    DECISION_SECONDS,
    GridPosition,
    GameState,
    create_game,
    generate_rows,
    observe,
    step_game,
)
from fly_crossy.schema import ACTION_ORDER
from fly_crossy.v2.crossy_camera import render_crossy_neural_frame
from fly_crossy.v2.preference_distill import (
    PAIR_INDEX,
    pairwise_targets,
    planner_action_preferences,
    preference_loss,
    primary_acceptable_mask,
)
from fly_crossy.v4.expo_seed_audit import rollout_expert
from fly_crossy.v4.policy import (
    CHANNELS,
    FEATURES,
    IMAGE_H,
    IMAGE_W,
    FullMaleCNSRGBPolicy,
)

EXPO_SEED = "crossy-v4-expo:0006"
SEMANTIC_DIM = 34  # current lane 4 + next lane 4 + support 1 + traffic radar 25
LANE_INDEX = {"grass": 0, "road": 1, "rail": 2, "river": 3}


@dataclass
class EpisodeData:
    images: np.ndarray
    acceptable: np.ndarray
    immediate_safe: np.ndarray
    pair_sign: np.ndarray
    pair_weight: np.ndarray
    best_action: np.ndarray
    semantic: np.ndarray
    source: str
    student_action: np.ndarray
    critical: np.ndarray
    phase_offset: float
    initial_column: int

    @property
    def length(self) -> int:
        return int(len(self.best_action))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_flyhard_root() -> Path:
    value = os.environ.get("FLYHARD_ROOT")
    return Path(value) if value else repo_root().parent / "flyhard"


def resize_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected RGB frame [H,W,3].")
    h, w, _ = frame.shape
    ys = np.rint(np.linspace(0, h - 1, IMAGE_H)).astype(np.int64)
    xs = np.rint(np.linspace(0, w - 1, IMAGE_W)).astype(np.int64)
    return (
        frame[ys[:, None], xs[None, :]].astype(np.float32) / 255.0
    ).astype(np.float32)


def variant_state(seed: str, *, phase_offset: float, initial_column: int) -> GameState:
    if abs(initial_column) > 2:
        raise ValueError("V4 training initial_column must stay in [-2, 2].")
    state = create_game(seed)
    return replace(
        state,
        time=float(phase_offset),
        step=int(round(float(phase_offset) / DECISION_SECONDS)),
        fly=GridPosition(row=state.fly.row, column=float(initial_column)),
    )


def lane_kind(state: GameState, row: int) -> str:
    lane = next((lane for lane in state.lanes if lane.row == row), None)
    if lane is None:
        lane = generate_rows(state.seed, row, 1)[0]
    return lane.kind


def semantic_target(state: GameState) -> np.ndarray:
    target = np.zeros(SEMANTIC_DIM, dtype=np.float32)
    target[LANE_INDEX[lane_kind(state, state.fly.row)]] = 1.0
    target[4 + LANE_INDEX[lane_kind(state, state.fly.row + 1)]] = 1.0
    observation = observe(state)
    target[8] = float(observation.support)
    traffic = np.asarray(observation.traffic, dtype=np.float32).reshape(-1)
    if traffic.shape != (25,):
        raise RuntimeError(f"Expected 25-value traffic radar, got {traffic.shape}.")
    target[9:] = traffic
    return target


def _episode_from_lists(
    *,
    images,
    acceptable,
    immediate_safe,
    pair_sign,
    pair_weight,
    best_action,
    semantic,
    source: str,
    student_action,
    critical,
    phase_offset: float,
    initial_column: int,
) -> EpisodeData:
    return EpisodeData(
        images=np.asarray(images, dtype=np.float16),
        acceptable=np.asarray(acceptable, dtype=np.bool_),
        immediate_safe=np.asarray(immediate_safe, dtype=np.bool_),
        pair_sign=np.asarray(pair_sign, dtype=np.int8),
        pair_weight=np.asarray(pair_weight, dtype=np.float16),
        best_action=np.asarray(best_action, dtype=np.int64),
        semantic=np.asarray(semantic, dtype=np.float16),
        source=source,
        student_action=np.asarray(student_action, dtype=np.int64),
        critical=np.asarray(critical, dtype=np.bool_),
        phase_offset=float(phase_offset),
        initial_column=int(initial_column),
    )


def collect_teacher_episode(
    *,
    seed: str,
    phase_offset: float,
    initial_column: int,
    max_steps: int,
    planner_depth: int,
    exploration: float,
    rng: np.random.Generator,
) -> EpisodeData:
    state = variant_state(
        seed, phase_offset=phase_offset, initial_column=initial_column
    )
    images, acceptable, immediate_safe = [], [], []
    pair_sign, pair_weight, best_action, semantic = [], [], [], []
    executed_actions, critical = [], []

    for _ in range(max_steps):
        if state.terminal is not None:
            break
        values, best_index, safe = planner_action_preferences(
            state, depth=planner_depth
        )
        acc = primary_acceptable_mask(values)
        signs, weights = pairwise_targets(values)

        images.append(resize_rgb(render_crossy_neural_frame(state)))
        acceptable.append(acc)
        immediate_safe.append(safe)
        pair_sign.append(signs)
        pair_weight.append(weights)
        best_action.append(best_index)
        semantic.append(semantic_target(state))

        behavior = best_index
        if rng.random() < exploration:
            alternatives = np.flatnonzero(safe)
            alternatives = alternatives[alternatives != best_index]
            if len(alternatives):
                behavior = int(rng.choice(alternatives))
        executed_actions.append(behavior)
        critical.append(False)
        state = step_game(state, ACTION_ORDER[behavior]).state

    return _episode_from_lists(
        images=images,
        acceptable=acceptable,
        immediate_safe=immediate_safe,
        pair_sign=pair_sign,
        pair_weight=pair_weight,
        best_action=best_action,
        semantic=semantic,
        source="teacher",
        student_action=executed_actions,
        critical=critical,
        phase_offset=phase_offset,
        initial_column=initial_column,
    )


def collect_teacher_dataset(
    *,
    episodes: int,
    seed: str,
    max_steps: int,
    planner_depth: int,
    exploration: float,
    rng: np.random.Generator,
) -> list[EpisodeData]:
    result = []
    columns = np.asarray([-2, -1, 0, 1, 2], dtype=np.int64)
    for index in range(episodes):
        phase = 0.0 if index == 0 else float(rng.uniform(0.0, 1.2))
        column = 0 if index == 0 else int(rng.choice(columns))
        episode = collect_teacher_episode(
            seed=seed,
            phase_offset=phase,
            initial_column=column,
            max_steps=max_steps,
            planner_depth=planner_depth,
            exploration=exploration,
            rng=rng,
        )
        result.append(episode)
        if (index + 1) % 8 == 0 or index + 1 == episodes:
            samples = sum(item.length for item in result)
            print(
                f"  teacher {index + 1}/{episodes} samples={samples:,}",
                flush=True,
            )
    return result


@torch.no_grad()
def collect_dagger_episode(
    *,
    policy: FullMaleCNSRGBPolicy,
    device: torch.device,
    seed: str,
    phase_offset: float,
    initial_column: int,
    max_steps: int,
    planner_depth: int,
) -> EpisodeData:
    policy.eval()
    state = variant_state(
        seed, phase_offset=phase_offset, initial_column=initial_column
    )
    neural = policy.zero_state(1, device=device)

    images, acceptable, immediate_safe = [], [], []
    pair_sign, pair_weight, best_action, semantic = [], [], [], []
    student_actions, critical = [], []

    for _ in range(max_steps):
        if state.terminal is not None:
            break
        image_np = resize_rgb(render_crossy_neural_frame(state))
        image = torch.as_tensor(
            image_np[None], dtype=torch.float32, device=device
        )
        neural = policy.step_state(neural, image)
        student = int(policy.readout(neural)[0].argmax().item())

        values, best_index, safe = planner_action_preferences(
            state, depth=planner_depth
        )
        acc = primary_acceptable_mask(values)
        signs, weights = pairwise_targets(values)
        next_state = step_game(state, ACTION_ORDER[student]).state

        images.append(image_np)
        acceptable.append(acc)
        immediate_safe.append(safe)
        pair_sign.append(signs)
        pair_weight.append(weights)
        best_action.append(best_index)
        semantic.append(semantic_target(state))
        student_actions.append(student)
        critical.append(bool(next_state.terminal is not None and safe[best_index]))

        state = next_state

    policy.train()
    return _episode_from_lists(
        images=images,
        acceptable=acceptable,
        immediate_safe=immediate_safe,
        pair_sign=pair_sign,
        pair_weight=pair_weight,
        best_action=best_action,
        semantic=semantic,
        source="dagger",
        student_action=student_actions,
        critical=critical,
        phase_offset=phase_offset,
        initial_column=initial_column,
    )


def collect_dagger_dataset(
    *,
    policy: FullMaleCNSRGBPolicy,
    device: torch.device,
    episodes: int,
    seed: str,
    max_steps: int,
    planner_depth: int,
    rng: np.random.Generator,
) -> list[EpisodeData]:
    result = []
    columns = np.asarray([-2, -1, 0, 1, 2], dtype=np.int64)
    for index in range(episodes):
        phase = float(rng.uniform(0.0, 1.2))
        column = int(rng.choice(columns))
        item = collect_dagger_episode(
            policy=policy,
            device=device,
            seed=seed,
            phase_offset=phase,
            initial_column=column,
            max_steps=max_steps,
            planner_depth=planner_depth,
        )
        result.append(item)
        if (index + 1) % 6 == 0 or index + 1 == episodes:
            samples = sum(x.length for x in result)
            deaths = sum(bool(x.critical.any()) for x in result)
            print(
                f"  dagger {index + 1}/{episodes} samples={samples:,} "
                f"episodesWithFatal={deaths}",
                flush=True,
            )
    return result


def dataset_weights(
    episodes: list[EpisodeData],
) -> tuple[list[np.ndarray], dict[str, object]]:
    all_best = np.concatenate([episode.best_action for episode in episodes])
    counts = np.bincount(all_best, minlength=len(ACTION_ORDER)).astype(np.float64)
    total = float(counts.sum())
    balanced = np.sqrt(total / (len(ACTION_ORDER) * np.maximum(counts, 1.0)))
    floors = np.asarray([0.75, 1.5, 1.25, 1.25, 3.0], dtype=np.float64)
    class_weights = np.clip(np.maximum(balanced, floors), 0.75, 4.0)

    result = []
    for episode in episodes:
        weights = class_weights[episode.best_action].astype(np.float32)
        safe_count = episode.immediate_safe.sum(axis=1).astype(np.float32)
        weights *= 1.0 + 0.20 * (len(ACTION_ORDER) - safe_count)
        if episode.source == "dagger":
            rows = np.arange(episode.length)
            disagreement = ~episode.acceptable[rows, episode.student_action]
            weights *= np.where(disagreement, 1.5, 1.0)
            weights *= np.where(episode.critical, 2.5, 1.0)
        result.append(np.clip(weights, 0.5, 12.0).astype(np.float32))

    report = {
        "bestActionCounts": {
            ACTION_ORDER[index].value: int(counts[index])
            for index in range(len(ACTION_ORDER))
        },
        "classWeights": {
            ACTION_ORDER[index].value: float(class_weights[index])
            for index in range(len(ACTION_ORDER))
        },
        "samples": int(total),
        "meanWeight": float(np.concatenate(result).mean()),
        "maxWeight": float(np.concatenate(result).max()),
    }
    return result, report


def _padded_chunk(
    group: list[EpisodeData],
    weights: list[np.ndarray],
    start: int,
    window: int,
):
    batch = len(group)
    length = min(window, max(episode.length for episode in group) - start)
    images = np.zeros((length, batch, IMAGE_H, IMAGE_W, CHANNELS), dtype=np.float32)
    acceptable = np.zeros((length, batch, len(ACTION_ORDER)), dtype=np.bool_)
    immediate_safe = np.ones((length, batch, len(ACTION_ORDER)), dtype=np.bool_)
    pair_sign = np.zeros((length, batch, len(PAIR_INDEX)), dtype=np.int8)
    pair_weight = np.zeros((length, batch, len(PAIR_INDEX)), dtype=np.float32)
    semantic = np.zeros((length, batch, SEMANTIC_DIM), dtype=np.float32)
    sample_weight = np.ones((length, batch), dtype=np.float32)
    mask = np.zeros((length, batch), dtype=np.bool_)

    for b, episode in enumerate(group):
        end = min(episode.length, start + length)
        take = max(0, end - start)
        if take == 0:
            continue
        sl = slice(start, end)
        images[:take, b] = episode.images[sl]
        acceptable[:take, b] = episode.acceptable[sl]
        immediate_safe[:take, b] = episode.immediate_safe[sl]
        pair_sign[:take, b] = episode.pair_sign[sl]
        pair_weight[:take, b] = episode.pair_weight[sl]
        semantic[:take, b] = episode.semantic[sl]
        sample_weight[:take, b] = weights[b][sl]
        mask[:take, b] = True

    return (
        images,
        acceptable,
        immediate_safe,
        pair_sign,
        pair_weight,
        semantic,
        sample_weight,
        mask,
    )


def train_sequential_epoch(
    *,
    policy: FullMaleCNSRGBPolicy,
    semantic_head: nn.Linear,
    episodes: list[EpisodeData],
    episode_weights: list[np.ndarray],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    window: int,
    semantic_weight: float,
    rng: np.random.Generator,
    stage: str,
    history: list[dict[str, object]],
) -> None:
    order = rng.permutation(len(episodes))
    policy.train()
    semantic_head.train()

    for group_start in range(0, len(order), batch_size):
        indexes = order[group_start:group_start + batch_size].tolist()
        group = [episodes[index] for index in indexes]
        group_weights = [episode_weights[index] for index in indexes]
        state = policy.zero_state(len(group), device=device)
        max_length = max(episode.length for episode in group)

        for start in range(0, max_length, window):
            (
                images_np,
                acceptable_np,
                safe_np,
                sign_np,
                pair_weight_np,
                semantic_np,
                sample_weight_np,
                mask_np,
            ) = _padded_chunk(group, group_weights, start, window)
            if not mask_np.any():
                continue

            images = torch.as_tensor(images_np, dtype=torch.float32, device=device)
            acceptable = torch.as_tensor(acceptable_np, dtype=torch.bool, device=device)
            immediate_safe = torch.as_tensor(safe_np, dtype=torch.bool, device=device)
            pair_sign = torch.as_tensor(sign_np, dtype=torch.float32, device=device)
            pair_weight = torch.as_tensor(
                pair_weight_np, dtype=torch.float32, device=device
            )
            semantic = torch.as_tensor(semantic_np, dtype=torch.float32, device=device)
            sample_weight = torch.as_tensor(
                sample_weight_np, dtype=torch.float32, device=device
            )
            mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits, motor, next_state = policy.run_window_with_motor(images, state)

            valid_logits = logits[mask]
            pref_loss, parts = preference_loss(
                valid_logits,
                acceptable[mask],
                immediate_safe[mask],
                pair_sign[mask],
                pair_weight[mask],
                sample_weight[mask],
            )
            semantic_pred = semantic_head(motor[mask])
            semantic_loss = F.mse_loss(semantic_pred, semantic[mask])
            loss = pref_loss + semantic_weight * semantic_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite V4 training loss.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(semantic_head.parameters()), 1.0
            )
            optimizer.step()

            # True stateful TBPTT: preserve the neural state value between
            # consecutive chunks and only cut the gradient graph here.
            state = next_state.detach()
            row = {
                "update": len(history) + 1,
                "stage": stage,
                "loss": float(loss.detach()),
                "preferenceLoss": float(pref_loss.detach()),
                "setLoss": float(parts["set"]),
                "pairLoss": float(parts["pair"]),
                "fatalMass": float(parts["fatal"]),
                "semanticLoss": float(semantic_loss.detach()),
                "validFrames": int(mask.sum().item()),
            }
            history.append(row)
            if len(history) == 1 or len(history) % 25 == 0:
                print(
                    f"  update {len(history):4d} {stage} "
                    f"loss={row['loss']:.4f} pref={row['preferenceLoss']:.4f} "
                    f"semantic={row['semanticLoss']:.4f}",
                    flush=True,
                )


@torch.no_grad()
def evaluate_variant(
    policy: FullMaleCNSRGBPolicy,
    device: torch.device,
    *,
    seed: str,
    phase_offset: float,
    initial_column: int,
    max_steps: int,
) -> dict[str, object]:
    policy.eval()
    state = variant_state(
        seed, phase_offset=phase_offset, initial_column=initial_column
    )
    neural = policy.zero_state(1, device=device)
    actions: Counter[str] = Counter()
    steps = 0

    while state.terminal is None and steps < max_steps:
        image = torch.as_tensor(
            resize_rgb(render_crossy_neural_frame(state))[None],
            dtype=torch.float32,
            device=device,
        )
        neural = policy.step_state(neural, image)
        action_index = int(policy.readout(neural)[0].argmax().item())
        action = ACTION_ORDER[action_index]
        actions[action.value] += 1
        state = step_game(state, action).state
        steps += 1

    return {
        "phaseOffset": float(phase_offset),
        "initialColumn": int(initial_column),
        "steps": steps,
        "score": float(state.score),
        "reachedStepLimit": bool(state.terminal is None and steps >= max_steps),
        "terminalReason": state.terminal,
        "actions": dict(sorted(actions.items())),
    }


def robustness_configs(smoke: bool) -> list[tuple[float, int]]:
    if smoke:
        return [(0.0, 0), (0.4, 0), (0.8, 0)]
    offsets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    columns = [-1, 0, 1]
    return [(offset, column) for offset in offsets for column in columns]


@torch.no_grad()
def evaluate_suite(
    policy: FullMaleCNSRGBPolicy,
    device: torch.device,
    *,
    seed: str,
    max_steps: int,
    smoke: bool,
) -> dict[str, object]:
    episodes = [
        evaluate_variant(
            policy,
            device,
            seed=seed,
            phase_offset=phase,
            initial_column=column,
            max_steps=max_steps,
        )
        for phase, column in robustness_configs(smoke)
    ]
    exact = next(
        item
        for item in episodes
        if item["phaseOffset"] == 0.0 and item["initialColumn"] == 0
    )
    terminals = Counter(
        item["terminalReason"] for item in episodes if item["terminalReason"] is not None
    )
    actions: Counter[str] = Counter()
    for item in episodes:
        actions.update(item["actions"])
    scores = [float(item["score"]) for item in episodes]
    return {
        "exactExpo": exact,
        "robust": {
            "episodes": len(episodes),
            "reachedStepLimit": sum(item["reachedStepLimit"] for item in episodes),
            "meanScore": float(np.mean(scores)),
            "medianScore": float(np.median(scores)),
            "terminalReasons": dict(sorted(terminals.items())),
            "actions": dict(sorted(actions.items())),
        },
        "perVariant": episodes,
    }


def selection_key(metrics: dict[str, object]) -> tuple[float, ...]:
    exact = metrics["exactExpo"]
    robust = metrics["robust"]
    return (
        float(exact["reachedStepLimit"]),
        float(robust["reachedStepLimit"]),
        float(exact["score"]),
        float(robust["meanScore"]),
        float(robust["medianScore"]),
    )


def save_checkpoint(
    path: Path,
    *,
    policy: FullMaleCNSRGBPolicy,
    semantic_head: nn.Linear,
    stage: str,
    config: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "semanticHead": semantic_head.state_dict(),
            "stage": stage,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    policy: FullMaleCNSRGBPolicy,
    semantic_head: nn.Linear,
) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(checkpoint["model"], strict=True)
    semantic_head.load_state_dict(checkpoint["semanticHead"], strict=True)


def evaluate_and_maybe_select(
    *,
    stage: str,
    policy: FullMaleCNSRGBPolicy,
    semantic_head: nn.Linear,
    device: torch.device,
    seed: str,
    eval_steps: int,
    smoke: bool,
    out: Path,
    config: dict[str, object],
    best_metrics: dict[str, object] | None,
    best_stage: str | None,
) -> tuple[dict[str, object], str, bool]:
    metrics = evaluate_suite(
        policy,
        device,
        seed=seed,
        max_steps=eval_steps,
        smoke=smoke,
    )
    improved = best_metrics is None or selection_key(metrics) > selection_key(best_metrics)
    target = out / ("best.pt" if improved else "latest.pt")
    save_checkpoint(
        target,
        policy=policy,
        semantic_head=semantic_head,
        stage=stage,
        config=config,
    )
    if improved:
        best_metrics = metrics
        best_stage = stage
    print(
        f"  {stage}: exact={metrics['exactExpo']['steps']}/{eval_steps} "
        f"score={metrics['exactExpo']['score']:.1f} "
        f"robust={metrics['robust']['reachedStepLimit']}/{metrics['robust']['episodes']} "
        f"mean={metrics['robust']['meanScore']:.1f} "
        f"{'NEW BEST' if improved else 'keep best'}",
        flush=True,
    )
    return best_metrics, str(best_stage), improved


def _teacher_gate(seed: str, *, max_steps: int, smoke: bool) -> dict[str, object]:
    gate_steps = min(max_steps, 40) if smoke else 200
    offsets = [0.0] if smoke else [0.0, 0.4, 0.8, 1.2]
    checks = [
        rollout_expert(
            seed,
            depth=4,
            max_steps=gate_steps,
            time_offset=offset,
        )
        for offset in offsets
    ]
    if not all(item["reachedStepLimit"] for item in checks):
        raise SystemExit(
            "Teacher gate failed for the expo seed. Do not train V4 until the "
            "planner/seed audit is fixed."
        )
    return {"maxSteps": gate_steps, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crossy V4 Expo Specialist: RGB full-MaleCNS, true state-carried "
            "TBPTT, preference/value supervision, semantic auxiliary training, "
            "and same-seed DAgger."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flyhard-root", type=Path, default=default_flyhard_root())
    parser.add_argument("--seed", default=EXPO_SEED)
    parser.add_argument("--rng-seed", type=int, default=404)
    parser.add_argument("--planner-depth", type=int, default=4)
    parser.add_argument("--demo-episodes", type=int, default=48)
    parser.add_argument("--demo-steps", type=int, default=200)
    parser.add_argument("--exploration", type=float, default=0.15)
    parser.add_argument("--clone-epochs", type=int, default=4)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--dagger-episodes", type=int, default=24)
    parser.add_argument("--dagger-epochs", type=int, default=2)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--semantic-weight", type=float, default=0.20)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--target-robust", type=int, default=18)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root() / "runs" / "crossy-v4-expo-specialist",
    )
    args = parser.parse_args()

    if args.smoke:
        args.demo_episodes = 2
        args.demo_steps = 16
        args.clone_epochs = 1
        args.dagger_rounds = 0
        args.batch = 2
        args.eval_steps = 20
        args.out = repo_root() / "runs" / "crossy-v4-smoke"

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)

    print("=== CROSSY V4 / EXPO SPECIALIST ===", flush=True)
    print(f"expo seed: {args.seed}", flush=True)
    print(f"sensor: {IMAGE_W}x{IMAGE_H} RGB = {FEATURES:,} features", flush=True)
    print("reward shaping: NOT USED", flush=True)
    print("planner/GameState privilege: TRAINING ONLY", flush=True)
    print("state reset: EPISODE START ONLY", flush=True)

    teacher_gate = _teacher_gate(
        args.seed, max_steps=args.eval_steps, smoke=args.smoke
    )
    print("teacher gate: PASS", flush=True)

    graph_root = args.flyhard_root / "data" / "graph-traced-v1"
    graph_path = graph_root / "graph.npz"
    nodes_path = graph_root / "nodes.feather"
    if not graph_path.is_file() or not nodes_path.is_file():
        raise SystemExit(
            "Full traced MaleCNS graph missing. Expected "
            "flyhard/data/graph-traced-v1."
        )

    graph = dict(np.load(graph_path))
    nodes = feather.read_table(nodes_path)
    classes = np.asarray(nodes["superclass"].fill_null("").to_pylist())
    sensory_ids = np.flatnonzero(classes == "ol_sensory")
    motor_ids = np.flatnonzero(classes == "vnc_motor")

    policy = FullMaleCNSRGBPolicy(
        graph, sensory_ids, motor_ids, seed=args.rng_seed
    ).to(device)
    semantic_head = nn.Linear(len(motor_ids), SEMANTIC_DIM).to(device)

    trainable_policy = sum(parameter.numel() for parameter in policy.parameters())
    trainable_aux = sum(parameter.numel() for parameter in semantic_head.parameters())
    print(f"neurons: {len(graph['crow']) - 1:,}", flush=True)
    print(f"edges: {len(graph['col']):,}", flush=True)
    print(f"ol_sensory: {len(sensory_ids):,}", flush=True)
    print(f"vnc_motor: {len(motor_ids):,}", flush=True)
    print(f"trainable MaleCNS: {trainable_policy:,}", flush=True)
    print(f"training-only semantic head: {trainable_aux:,}", flush=True)

    print("\n[1] collect same-seed expert preferences...", flush=True)
    episodes = collect_teacher_dataset(
        episodes=args.demo_episodes,
        seed=args.seed,
        max_steps=args.demo_steps,
        planner_depth=args.planner_depth,
        exploration=args.exploration,
        rng=rng,
    )
    if not episodes or not sum(episode.length for episode in episodes):
        raise SystemExit("Teacher collection produced no samples.")

    calibration_frames = np.concatenate(
        [episode.images[: min(4, episode.length)] for episode in episodes]
    )[:64]
    policy.calibrate_decoder(
        torch.as_tensor(calibration_frames, dtype=torch.float32, device=device)
    )

    config = {
        "version": "crossy-v4-expo-specialist-1",
        "seed": args.seed,
        "rngSeed": args.rng_seed,
        "plannerDepth": args.planner_depth,
        "sensor": {
            "width": IMAGE_W,
            "height": IMAGE_H,
            "channels": CHANNELS,
            "features": FEATURES,
            "allRgbFeaturesCoveredAtLeastOnce": len(sensory_ids) >= FEATURES,
        },
        "training": {
            "demoEpisodes": args.demo_episodes,
            "demoSteps": args.demo_steps,
            "exploration": args.exploration,
            "cloneEpochs": args.clone_epochs,
            "daggerRounds": args.dagger_rounds,
            "daggerEpisodes": args.dagger_episodes,
            "daggerEpochs": args.dagger_epochs,
            "window": args.window,
            "batch": args.batch,
            "learningRate": args.lr,
            "semanticWeight": args.semantic_weight,
            "rewardUsed": False,
            "stateReset": "episode-start-only",
        },
        "architecture": {
            "neurons": int(len(graph["crow"]) - 1),
            "edges": int(len(graph["col"])),
            "sensory": int(len(sensory_ids)),
            "motor": int(len(motor_ids)),
            "MaleCNSTrainableParameters": int(trainable_policy),
            "trainingOnlyAuxParameters": int(trainable_aux),
        },
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(semantic_head.parameters()),
        lr=args.lr,
    )
    history: list[dict[str, object]] = []
    stages: dict[str, object] = {}
    started = time.perf_counter()

    print("\n[2] untrained calibrated baseline...", flush=True)
    baseline = evaluate_suite(
        policy,
        device,
        seed=args.seed,
        max_steps=args.eval_steps,
        smoke=args.smoke,
    )
    stages["baseline"] = baseline
    print(
        f"  exact={baseline['exactExpo']['steps']}/{args.eval_steps} "
        f"score={baseline['exactExpo']['score']:.1f}; "
        f"robust={baseline['robust']['reachedStepLimit']}/"
        f"{baseline['robust']['episodes']}",
        flush=True,
    )

    best_metrics: dict[str, object] | None = None
    best_stage: str | None = None

    print("\n[3] state-carried preference training...", flush=True)
    for epoch in range(1, args.clone_epochs + 1):
        weights, weight_report = dataset_weights(episodes)
        train_sequential_epoch(
            policy=policy,
            semantic_head=semantic_head,
            episodes=episodes,
            episode_weights=weights,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch,
            window=args.window,
            semantic_weight=args.semantic_weight,
            rng=rng,
            stage=f"clone-e{epoch}",
            history=history,
        )

    best_metrics, best_stage, _ = evaluate_and_maybe_select(
        stage="clone",
        policy=policy,
        semantic_head=semantic_head,
        device=device,
        seed=args.seed,
        eval_steps=args.eval_steps,
        smoke=args.smoke,
        out=args.out,
        config=config,
        best_metrics=best_metrics,
        best_stage=best_stage,
    )
    stages["clone"] = best_metrics if best_stage == "clone" else evaluate_suite(
        policy, device, seed=args.seed, max_steps=args.eval_steps, smoke=args.smoke
    )

    for round_index in range(1, args.dagger_rounds + 1):
        robust = best_metrics["robust"] if best_metrics is not None else None
        exact = best_metrics["exactExpo"] if best_metrics is not None else None
        if (
            exact is not None
            and exact["reachedStepLimit"]
            and robust["reachedStepLimit"] >= min(
                args.target_robust, robust["episodes"]
            )
        ):
            print("\nTarget reached; stopping before unnecessary DAgger.", flush=True)
            break

        if (args.out / "best.pt").is_file():
            load_checkpoint(
                args.out / "best.pt",
                policy=policy,
                semantic_head=semantic_head,
            )
            # If a prior candidate was rejected, restart Adam from the selected
            # checkpoint rather than carrying optimizer moments from worse weights.
            optimizer = torch.optim.Adam(
                list(policy.parameters()) + list(semantic_head.parameters()),
                lr=args.lr,
            )

        print(
            f"\n[DAgger {round_index}/{args.dagger_rounds}] same-seed on-policy collection...",
            flush=True,
        )
        new_episodes = collect_dagger_dataset(
            policy=policy,
            device=device,
            episodes=args.dagger_episodes,
            seed=args.seed,
            max_steps=args.demo_steps,
            planner_depth=args.planner_depth,
            rng=rng,
        )
        episodes.extend(new_episodes)
        weights, weight_report = dataset_weights(episodes)
        print(json.dumps(weight_report, indent=2), flush=True)

        for epoch in range(1, args.dagger_epochs + 1):
            train_sequential_epoch(
                policy=policy,
                semantic_head=semantic_head,
                episodes=episodes,
                episode_weights=weights,
                optimizer=optimizer,
                device=device,
                batch_size=args.batch,
                window=args.window,
                semantic_weight=args.semantic_weight,
                rng=rng,
                stage=f"dagger{round_index}-e{epoch}",
                history=history,
            )

        candidate = evaluate_suite(
            policy,
            device,
            seed=args.seed,
            max_steps=args.eval_steps,
            smoke=args.smoke,
        )
        stages[f"dagger{round_index}"] = candidate
        improved = selection_key(candidate) > selection_key(best_metrics)
        target = args.out / ("best.pt" if improved else "latest.pt")
        save_checkpoint(
            target,
            policy=policy,
            semantic_head=semantic_head,
            stage=f"dagger{round_index}",
            config=config,
        )
        if improved:
            best_metrics = candidate
            best_stage = f"dagger{round_index}"
        print(
            f"  dagger{round_index}: exact={candidate['exactExpo']['steps']}/"
            f"{args.eval_steps} score={candidate['exactExpo']['score']:.1f} "
            f"robust={candidate['robust']['reachedStepLimit']}/"
            f"{candidate['robust']['episodes']} "
            f"mean={candidate['robust']['meanScore']:.1f} "
            f"{'NEW BEST' if improved else 'KEEP PREVIOUS BEST'}",
            flush=True,
        )

    if not (args.out / "best.pt").is_file():
        raise RuntimeError("No V4 best checkpoint was produced.")
    load_checkpoint(
        args.out / "best.pt", policy=policy, semantic_head=semantic_head
    )
    final = evaluate_suite(
        policy,
        device,
        seed=args.seed,
        max_steps=args.eval_steps,
        smoke=args.smoke,
    )

    report = {
        "version": "crossy-v4-expo-specialist-1",
        "seed": args.seed,
        "teacherGate": teacher_gate,
        "config": config,
        "baseline": baseline,
        "stages": stages,
        "selectedStage": best_stage,
        "selected": final,
        "dataset": {
            "episodes": len(episodes),
            "samples": int(sum(episode.length for episode in episodes)),
            "sources": dict(Counter(episode.source for episode in episodes)),
            "weights": dataset_weights(episodes)[1],
        },
        "history": history,
        "elapsedSeconds": time.perf_counter() - started,
        "checkpoint": str((args.out / "best.pt").resolve()),
        "productionPath": "48x24 RGB camera -> measured MaleCNS -> fixed motor readout -> action",
        "trainingOnlyPrivilege": (
            "Planner preferences, authoritative safety labels and semantic auxiliary "
            "targets are never available at inference."
        ),
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== V4 COMPLETE ===", flush=True)
    print(f"selected stage: {best_stage}", flush=True)
    print(
        f"EXPO exact: {final['exactExpo']['steps']}/{args.eval_steps} "
        f"score={final['exactExpo']['score']:.1f} "
        f"terminal={final['exactExpo']['terminalReason']}",
        flush=True,
    )
    print(
        f"robustness: {final['robust']['reachedStepLimit']}/"
        f"{final['robust']['episodes']} reached limit; "
        f"meanScore={final['robust']['meanScore']:.1f}",
        flush=True,
    )
    print(f"actions: {final['robust']['actions']}", flush=True)
    print(f"checkpoint: {(args.out / 'best.pt').resolve()}", flush=True)
    print(f"report: {report_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
