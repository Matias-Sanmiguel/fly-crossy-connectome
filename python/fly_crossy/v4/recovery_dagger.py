from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pyarrow.feather as feather
import torch
from torch import nn
import torch.nn.functional as F

from fly_crossy.env import step_game
from fly_crossy.schema import ACTION_ORDER
from fly_crossy.v2.crossy_camera import render_crossy_neural_frame
from fly_crossy.v2.preference_distill import (
    pairwise_targets,
    planner_action_preferences,
    preference_loss,
    primary_acceptable_mask,
)
from fly_crossy.v4.policy import FullMaleCNSRGBPolicy
from fly_crossy.v4.train_expo_specialist import (
    EXPO_SEED,
    SEMANTIC_DIM,
    default_flyhard_root,
    resize_rgb,
    robustness_configs,
    semantic_target,
    variant_state,
)

ACTION_NAMES = [action.value for action in ACTION_ORDER]
ACTION_COUNT = len(ACTION_ORDER)
PAIR_COUNT = ACTION_COUNT * (ACTION_COUNT - 1) // 2


@dataclass
class RecoveryData:
    motor: np.ndarray
    acceptable: np.ndarray
    safe: np.ndarray
    pair_sign: np.ndarray
    pair_weight: np.ndarray
    best_action: np.ndarray
    semantic: np.ndarray
    behavior_action: np.ndarray
    recovery: np.ndarray
    source: np.ndarray

    @property
    def length(self) -> int:
        return int(len(self.best_action))


class RecoveryDecoder(nn.Module):
    """Tiny trainable 708->5 readout on top of a completely frozen MaleCNS."""

    def __init__(
        self,
        motor_dim: int,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", mean.reshape(1, motor_dim).float().clone())
        self.register_buffer("std", std.reshape(1, motor_dim).float().clamp_min(1e-4).clone())
        self.linear = nn.Linear(motor_dim, ACTION_COUNT, bias=True)

    def forward(self, motor: torch.Tensor) -> torch.Tensor:
        return self.linear((motor - self.mean) / self.std)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_frozen_policy(
    *,
    checkpoint_path: Path,
    flyhard_root: Path,
    device: torch.device,
) -> tuple[FullMaleCNSRGBPolicy, dict[str, Any]]:
    graph_root = flyhard_root / "data" / "graph-traced-v1"
    graph_path = graph_root / "graph.npz"
    nodes_path = graph_root / "nodes.feather"
    if not graph_path.is_file() or not nodes_path.is_file():
        raise FileNotFoundError(
            "Full traced MaleCNS graph missing. Expected "
            "flyhard/data/graph-traced-v1/{graph.npz,nodes.feather}."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V4 checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    graph = dict(np.load(graph_path))
    nodes = feather.read_table(nodes_path)
    classes = np.asarray(nodes["superclass"].fill_null("").to_pylist())
    sensory_ids = np.flatnonzero(classes == "ol_sensory")
    motor_ids = np.flatnonzero(classes == "vnc_motor")

    config = checkpoint.get("config", {})
    rng_seed = int(config.get("rngSeed", 404)) if isinstance(config, dict) else 404
    policy = FullMaleCNSRGBPolicy(
        graph,
        sensory_ids,
        motor_ids,
        seed=rng_seed,
    ).to(device)
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    return policy, {
        "checkpointStage": checkpoint.get("stage"),
        "rngSeed": rng_seed,
        "neurons": int(len(graph["crow"]) - 1),
        "edges": int(len(graph["col"])),
        "sensory": int(len(sensory_ids)),
        "motor": int(len(motor_ids)),
    }


def _to_data(rows: list[dict[str, Any]]) -> RecoveryData:
    if not rows:
        raise ValueError("Cannot build RecoveryData from zero rows.")
    return RecoveryData(
        motor=np.asarray([row["motor"] for row in rows], dtype=np.float32),
        acceptable=np.asarray([row["acceptable"] for row in rows], dtype=np.bool_),
        safe=np.asarray([row["safe"] for row in rows], dtype=np.bool_),
        pair_sign=np.asarray([row["pair_sign"] for row in rows], dtype=np.int8),
        pair_weight=np.asarray([row["pair_weight"] for row in rows], dtype=np.float32),
        best_action=np.asarray([row["best_action"] for row in rows], dtype=np.int64),
        semantic=np.asarray([row["semantic"] for row in rows], dtype=np.float32),
        behavior_action=np.asarray([row["behavior_action"] for row in rows], dtype=np.int64),
        recovery=np.asarray([row["recovery"] for row in rows], dtype=np.bool_),
        source=np.asarray([row["source"] for row in rows]),
    )


def concat_data(parts: list[RecoveryData]) -> RecoveryData:
    if not parts:
        raise ValueError("No RecoveryData parts supplied.")
    return RecoveryData(
        motor=np.concatenate([part.motor for part in parts], axis=0),
        acceptable=np.concatenate([part.acceptable for part in parts], axis=0),
        safe=np.concatenate([part.safe for part in parts], axis=0),
        pair_sign=np.concatenate([part.pair_sign for part in parts], axis=0),
        pair_weight=np.concatenate([part.pair_weight for part in parts], axis=0),
        best_action=np.concatenate([part.best_action for part in parts], axis=0),
        semantic=np.concatenate([part.semantic for part in parts], axis=0),
        behavior_action=np.concatenate([part.behavior_action for part in parts], axis=0),
        recovery=np.concatenate([part.recovery for part in parts], axis=0),
        source=np.concatenate([part.source for part in parts], axis=0),
    )


def _choose_safe_perturbation(
    *,
    acceptable: np.ndarray,
    safe: np.ndarray,
    best_index: int,
    rng: np.random.Generator,
) -> int | None:
    # Prefer an immediately-safe action that the planner does NOT consider
    # acceptable. This deliberately leaves the expert trajectory while avoiding
    # a pointless one-step suicide. If none exists, use any other safe action.
    preferred = np.flatnonzero(safe & ~acceptable)
    preferred = preferred[preferred != best_index]
    if len(preferred):
        return int(rng.choice(preferred))
    alternatives = np.flatnonzero(safe)
    alternatives = alternatives[alternatives != best_index]
    if len(alternatives):
        return int(rng.choice(alternatives))
    return None


@torch.no_grad()
def collect_episode(
    *,
    policy: FullMaleCNSRGBPolicy,
    device: torch.device,
    seed: str,
    phase_offset: float,
    initial_column: int,
    max_steps: int,
    planner_depth: int,
    mode: str,
    rng: np.random.Generator,
    decoder: RecoveryDecoder | None = None,
    perturb_probability: float = 0.18,
    recovery_horizon: int = 5,
) -> tuple[RecoveryData, dict[str, Any]]:
    if mode not in {"teacher", "perturb", "onpolicy"}:
        raise ValueError(f"Unknown collection mode: {mode}")
    if mode == "onpolicy" and decoder is None:
        raise ValueError("onpolicy collection requires a decoder")

    state = variant_state(
        seed,
        phase_offset=phase_offset,
        initial_column=initial_column,
    )
    neural = policy.zero_state(1, device=device)
    rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    perturbations = 0
    recovery_left = 0
    fatal_behavior = 0
    unacceptable_behavior = 0

    for _ in range(max_steps):
        if state.terminal is not None:
            break

        image = torch.as_tensor(
            resize_rgb(render_crossy_neural_frame(state))[None],
            dtype=torch.float32,
            device=device,
        )
        neural = policy.step_state(neural, image)
        motor = policy.motor_state(neural)[0]

        values, best_index, safe = planner_action_preferences(
            state,
            depth=planner_depth,
        )
        acceptable = primary_acceptable_mask(values)
        pair_sign, pair_weight = pairwise_targets(values)

        is_recovery_state = recovery_left > 0
        behavior = best_index
        if mode == "onpolicy":
            behavior = int(decoder(motor[None])[0].argmax().item())
        elif mode == "perturb":
            if recovery_left > 0:
                recovery_left -= 1
            elif rng.random() < perturb_probability:
                candidate = _choose_safe_perturbation(
                    acceptable=acceptable,
                    safe=safe,
                    best_index=best_index,
                    rng=rng,
                )
                if candidate is not None:
                    behavior = candidate
                    perturbations += 1
                    recovery_left = recovery_horizon

        rows.append(
            {
                "motor": motor.detach().cpu().numpy(),
                "acceptable": acceptable,
                "safe": safe,
                "pair_sign": pair_sign,
                "pair_weight": pair_weight,
                "best_action": best_index,
                "semantic": semantic_target(state),
                "behavior_action": behavior,
                "recovery": is_recovery_state,
                "source": mode,
            }
        )

        unacceptable_behavior += int(not acceptable[behavior])
        fatal_behavior += int(not safe[behavior])
        action_counts[ACTION_NAMES[behavior]] += 1
        state = step_game(state, ACTION_ORDER[behavior]).state

    data = _to_data(rows)
    return data, {
        "mode": mode,
        "phaseOffset": float(phase_offset),
        "initialColumn": int(initial_column),
        "frames": data.length,
        "score": float(state.score),
        "terminalReason": state.terminal,
        "reachedStepLimit": bool(state.terminal is None and data.length >= max_steps),
        "perturbations": perturbations,
        "recoveryFrames": int(data.recovery.sum()),
        "unacceptableBehavior": unacceptable_behavior,
        "unsafeBehavior": fatal_behavior,
        "actions": dict(sorted(action_counts.items())),
    }


def compute_training_weights(data: RecoveryData) -> tuple[np.ndarray, dict[str, Any]]:
    counts = np.bincount(data.best_action, minlength=ACTION_COUNT).astype(np.float64)
    total = max(1.0, float(counts.sum()))
    balanced = np.sqrt(total / (ACTION_COUNT * np.maximum(counts, 1.0)))
    floor_by_name = {
        "forward": 0.75,
        "backward": 1.0,
        "left": 1.10,
        "right": 1.10,
        "wait": 1.25,
    }
    floors = np.asarray([floor_by_name[name] for name in ACTION_NAMES], dtype=np.float64)
    class_weights = np.clip(np.maximum(balanced, floors), 0.70, 2.50)
    result = class_weights[data.best_action].astype(np.float32)

    source_scale = np.ones(data.length, dtype=np.float32)
    source_scale[data.source == "teacher"] = 0.80
    source_scale[data.source == "perturb"] = 1.20
    source_scale[data.source == "onpolicy"] = 1.50
    result *= source_scale
    result *= np.where(data.recovery, 2.0, 1.0).astype(np.float32)

    rows = np.arange(data.length)
    valid_behavior = data.behavior_action >= 0
    behavior = np.clip(data.behavior_action, 0, ACTION_COUNT - 1)
    unacceptable = np.zeros(data.length, dtype=np.bool_)
    unsafe = np.zeros(data.length, dtype=np.bool_)
    unacceptable[valid_behavior] = ~data.acceptable[rows[valid_behavior], behavior[valid_behavior]]
    unsafe[valid_behavior] = ~data.safe[rows[valid_behavior], behavior[valid_behavior]]
    result *= np.where(unacceptable, 2.0, 1.0).astype(np.float32)
    result *= np.where(unsafe, 4.0, 1.0).astype(np.float32)

    singleton = data.acceptable.sum(axis=1) == 1
    result *= np.where(singleton, 1.30, 1.0).astype(np.float32)
    result = np.clip(result, 0.40, 12.0).astype(np.float32)

    return result, {
        "samples": data.length,
        "bestActionCounts": {
            ACTION_NAMES[index]: int(counts[index])
            for index in range(ACTION_COUNT)
        },
        "classWeights": {
            ACTION_NAMES[index]: float(class_weights[index])
            for index in range(ACTION_COUNT)
        },
        "sourceCounts": dict(sorted(Counter(data.source.tolist()).items())),
        "recoveryFrames": int(data.recovery.sum()),
        "unacceptableBehaviorFrames": int(unacceptable.sum()),
        "unsafeBehaviorFrames": int(unsafe.sum()),
        "meanWeight": float(result.mean()),
        "maxWeight": float(result.max()),
    }


def _init_decoder_from_random_readout(
    *,
    policy: FullMaleCNSRGBPolicy,
    data: RecoveryData,
    device: torch.device,
) -> RecoveryDecoder:
    motor = torch.as_tensor(data.motor, dtype=torch.float32, device=device)
    mean = motor.mean(dim=0)
    std = motor.std(dim=0).clamp_min(1e-4)
    decoder = RecoveryDecoder(
        motor.shape[1],
        mean=mean,
        std=std,
    ).to(device)
    # Exact affine conversion of the original frozen random readout into the
    # normalized coordinate system used by RecoveryDecoder.
    with torch.no_grad():
        original = policy.decoder.to(device)
        decoder.linear.weight.copy_(original * std[None, :])
        decoder.linear.bias.copy_(mean @ original.T)
    return decoder


def fit_decoder(
    *,
    policy: FullMaleCNSRGBPolicy,
    data: RecoveryData,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> tuple[RecoveryDecoder, dict[str, Any]]:
    weights_np, weight_report = compute_training_weights(data)
    decoder = _init_decoder_from_random_readout(
        policy=policy,
        data=data,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    rng = np.random.default_rng(seed)

    x = torch.as_tensor(data.motor, dtype=torch.float32, device=device)
    acceptable = torch.as_tensor(data.acceptable, dtype=torch.bool, device=device)
    safe = torch.as_tensor(data.safe, dtype=torch.bool, device=device)
    pair_sign = torch.as_tensor(data.pair_sign, dtype=torch.float32, device=device)
    pair_weight = torch.as_tensor(data.pair_weight, dtype=torch.float32, device=device)
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)

    history: list[float] = []
    indices = np.arange(data.length)
    for epoch in range(1, epochs + 1):
        shuffled = rng.permutation(indices)
        epoch_losses = []
        for start in range(0, data.length, batch_size):
            batch_np = shuffled[start : start + batch_size]
            batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = decoder(x[batch])
            loss, _ = preference_loss(
                logits,
                acceptable[batch],
                safe[batch],
                pair_sign[batch],
                pair_weight[batch],
                weights[batch],
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite recovery decoder loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        mean_loss = float(np.mean(epoch_losses))
        history.append(mean_loss)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"  decoder epoch {epoch:3d}/{epochs} loss={mean_loss:.4f}",
                flush=True,
            )

    return decoder, {
        "epochs": epochs,
        "learningRate": learning_rate,
        "batchSize": batch_size,
        "firstLoss": history[0],
        "lastLoss": history[-1],
        "weights": weight_report,
    }


@torch.no_grad()
def action_metrics(decoder: RecoveryDecoder, data: RecoveryData, device: torch.device) -> dict[str, Any]:
    motor = torch.as_tensor(data.motor, dtype=torch.float32, device=device)
    logits = decoder(motor)
    predictions = logits.argmax(dim=1).cpu().numpy()
    rows = np.arange(data.length)
    return {
        "frames": data.length,
        "exactActionRate": float(np.mean(predictions == data.best_action)),
        "acceptableActionRate": float(np.mean(data.acceptable[rows, predictions])),
        "immediateSafeActionRate": float(np.mean(data.safe[rows, predictions])),
        "actions": dict(sorted(Counter(ACTION_NAMES[index] for index in predictions).items())),
    }


@torch.no_grad()
def evaluate_variant(
    *,
    policy: FullMaleCNSRGBPolicy,
    decoder: RecoveryDecoder,
    device: torch.device,
    seed: str,
    phase_offset: float,
    initial_column: int,
    max_steps: int,
) -> dict[str, Any]:
    state = variant_state(seed, phase_offset=phase_offset, initial_column=initial_column)
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
        motor = policy.motor_state(neural)
        action_index = int(decoder(motor)[0].argmax().item())
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


@torch.no_grad()
def evaluate_suite(
    *,
    policy: FullMaleCNSRGBPolicy,
    decoder: RecoveryDecoder,
    device: torch.device,
    seed: str,
    max_steps: int,
    smoke: bool,
) -> dict[str, Any]:
    variants = [
        evaluate_variant(
            policy=policy,
            decoder=decoder,
            device=device,
            seed=seed,
            phase_offset=phase,
            initial_column=column,
            max_steps=max_steps,
        )
        for phase, column in robustness_configs(smoke)
    ]
    exact = next(
        row for row in variants
        if row["phaseOffset"] == 0.0 and row["initialColumn"] == 0
    )
    terminals = Counter(
        row["terminalReason"] for row in variants if row["terminalReason"] is not None
    )
    actions: Counter[str] = Counter()
    for row in variants:
        actions.update(row["actions"])
    scores = [float(row["score"]) for row in variants]
    return {
        "exactExpo": exact,
        "robust": {
            "episodes": len(variants),
            "reachedStepLimit": int(sum(row["reachedStepLimit"] for row in variants)),
            "meanScore": float(np.mean(scores)),
            "medianScore": float(np.median(scores)),
            "terminalReasons": dict(sorted(terminals.items())),
            "actions": dict(sorted(actions.items())),
        },
        "perVariant": variants,
    }


def selection_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    exact = metrics["exactExpo"]
    robust = metrics["robust"]
    return (
        float(exact["reachedStepLimit"]),
        float(robust["reachedStepLimit"]),
        float(exact["score"]),
        float(robust["meanScore"]),
        float(robust["medianScore"]),
    )


def _semantic_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if len(pred) == 0:
        return {
            "frames": 0,
            "currentLaneAccuracy": 0.0,
            "nextLaneAccuracy": 0.0,
            "supportAccuracy": 0.0,
            "trafficMse": 0.0,
            "overallMse": 0.0,
        }
    return {
        "frames": int(len(pred)),
        "currentLaneAccuracy": float(
            np.mean(np.argmax(pred[:, :4], axis=1) == np.argmax(target[:, :4], axis=1))
        ),
        "nextLaneAccuracy": float(
            np.mean(np.argmax(pred[:, 4:8], axis=1) == np.argmax(target[:, 4:8], axis=1))
        ),
        "supportAccuracy": float(np.mean((pred[:, 8] >= 0.5) == (target[:, 8] >= 0.5))),
        "trafficMse": float(np.mean((pred[:, 9:] - target[:, 9:]) ** 2)),
        "overallMse": float(np.mean((pred - target) ** 2)),
    }


def fit_semantic_probe(
    *,
    data: RecoveryData,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> tuple[nn.Linear, torch.Tensor, torch.Tensor, dict[str, Any]]:
    x = torch.as_tensor(data.motor, dtype=torch.float32, device=device)
    y = torch.as_tensor(data.semantic, dtype=torch.float32, device=device)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-4)
    xn = (x - mean) / std
    probe = nn.Linear(x.shape[1], SEMANTIC_DIM).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    history = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = probe(xn)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        pred = probe(xn).cpu().numpy()
    return probe, mean, std, {
        "train": _semantic_metrics(pred, data.semantic),
        "firstLoss": history[0],
        "lastLoss": history[-1],
    }


@torch.no_grad()
def evaluate_semantic_probe(
    *,
    probe: nn.Linear,
    mean: torch.Tensor,
    std: torch.Tensor,
    data: RecoveryData,
    device: torch.device,
) -> dict[str, Any]:
    x = torch.as_tensor(data.motor, dtype=torch.float32, device=device)
    pred = probe((x - mean) / std).cpu().numpy()
    return _semantic_metrics(pred, data.semantic)


def _random_variant(rng: np.random.Generator, *, include_exact: bool) -> tuple[float, int]:
    if include_exact:
        return 0.0, 0
    phase = float(rng.uniform(0.0, 1.2))
    column = int(rng.choice(np.asarray([-2, -1, 0, 1, 2], dtype=np.int64)))
    # Avoid accidentally duplicating the exact expo start in anchor-only data.
    if phase < 0.05 and column == 0:
        phase = 0.2
    return phase, column


def collect_many(
    *,
    policy: FullMaleCNSRGBPolicy,
    device: torch.device,
    seed: str,
    episodes: int,
    max_steps: int,
    planner_depth: int,
    mode: str,
    rng: np.random.Generator,
    decoder: RecoveryDecoder | None = None,
    perturb_probability: float = 0.18,
    recovery_horizon: int = 5,
    include_exact_first: bool = False,
) -> tuple[RecoveryData, list[dict[str, Any]]]:
    parts: list[RecoveryData] = []
    reports: list[dict[str, Any]] = []
    for index in range(episodes):
        phase, column = (
            (0.0, 0)
            if include_exact_first and index == 0
            else _random_variant(rng, include_exact=False)
        )
        data, report = collect_episode(
            policy=policy,
            device=device,
            seed=seed,
            phase_offset=phase,
            initial_column=column,
            max_steps=max_steps,
            planner_depth=planner_depth,
            mode=mode,
            rng=rng,
            decoder=decoder,
            perturb_probability=perturb_probability,
            recovery_horizon=recovery_horizon,
        )
        parts.append(data)
        reports.append(report)
        if (index + 1) % 4 == 0 or index + 1 == episodes:
            print(
                f"  {mode} {index + 1}/{episodes} "
                f"frames={sum(part.length for part in parts):,}",
                flush=True,
            )
    return concat_data(parts), reports


def save_decoder(
    path: Path,
    *,
    decoder: RecoveryDecoder,
    source_checkpoint: Path,
    stage: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "decoder": decoder.state_dict(),
            "motorDim": int(decoder.linear.in_features),
            "sourceCheckpoint": str(source_checkpoint.resolve()),
            "stage": stage,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def clone_decoder(decoder: RecoveryDecoder, device: torch.device) -> RecoveryDecoder:
    copy = RecoveryDecoder(
        decoder.linear.in_features,
        mean=decoder.mean.detach().to(device).squeeze(0),
        std=decoder.std.detach().to(device).squeeze(0),
    ).to(device)
    copy.load_state_dict(decoder.state_dict(), strict=True)
    copy.eval()
    return copy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V4 Recovery DAgger: freeze the trained MaleCNS completely, then "
            "train only a tiny motor readout on teacher anchors, deliberate "
            "recoverable perturbations and states visited by the fly itself."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=EXPO_SEED)
    parser.add_argument("--rng-seed", type=int, default=505)
    parser.add_argument("--planner-depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--anchor-episodes", type=int, default=12)
    parser.add_argument("--perturb-episodes", type=int, default=16)
    parser.add_argument("--onpolicy-episodes", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--decoder-epochs", type=int, default=100)
    parser.add_argument("--decoder-lr", type=float, default=0.01)
    parser.add_argument("--decoder-batch", type=int, default=1024)
    parser.add_argument("--perturb-probability", type=float, default=0.18)
    parser.add_argument("--recovery-horizon", type=int, default=5)
    parser.add_argument("--semantic-probe-epochs", type=int, default=120)
    parser.add_argument("--semantic-probe-lr", type=float, default=0.02)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root() / "runs" / "crossy-v4-expo-specialist" / "best.pt",
    )
    parser.add_argument("--flyhard-root", type=Path, default=default_flyhard_root())
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root() / "runs" / "crossy-v4-recovery-dagger",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_steps = min(args.max_steps, 40)
        args.anchor_episodes = 2
        args.perturb_episodes = 2
        args.onpolicy_episodes = 2
        args.rounds = 1
        args.decoder_epochs = 8
        args.semantic_probe_epochs = 8

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    started = time.perf_counter()

    policy, metadata = load_frozen_policy(
        checkpoint_path=args.checkpoint,
        flyhard_root=args.flyhard_root,
        device=device,
    )
    trainable_policy_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print("=== CROSSY V4 / RECOVERY DAGGER ===", flush=True)
    print(f"source checkpoint: {args.checkpoint.resolve()}", flush=True)
    print(
        f"MaleCNS frozen: {metadata['neurons']:,} neurons / "
        f"{metadata['edges']:,} edges / trainable core params={trainable_policy_params}",
        flush=True,
    )
    print("trainable object: tiny 708 -> 5 motor decoder only", flush=True)

    print("\n[1] teacher anchors (exact expo phase withheld)...", flush=True)
    anchor, anchor_collection = collect_many(
        policy=policy,
        device=device,
        seed=args.seed,
        episodes=args.anchor_episodes,
        max_steps=args.max_steps,
        planner_depth=args.planner_depth,
        mode="teacher",
        rng=rng,
    )

    print("\n[2] deliberately perturbed but recoverable teacher trajectories...", flush=True)
    perturb, perturb_collection = collect_many(
        policy=policy,
        device=device,
        seed=args.seed,
        episodes=args.perturb_episodes,
        max_steps=args.max_steps,
        planner_depth=args.planner_depth,
        mode="perturb",
        rng=rng,
        perturb_probability=args.perturb_probability,
        recovery_horizon=args.recovery_horizon,
    )

    replay_parts: list[RecoveryData] = [anchor, perturb]
    initial_replay = concat_data(replay_parts)

    print("\n[3] fresh semantic probe on teacher anchors...", flush=True)
    semantic_probe, semantic_mean, semantic_std, semantic_train_report = fit_semantic_probe(
        data=anchor,
        device=device,
        epochs=args.semantic_probe_epochs,
        learning_rate=args.semantic_probe_lr,
    )

    print("\n[4] exact teacher-forced holdout features...", flush=True)
    exact_teacher, exact_teacher_collection = collect_episode(
        policy=policy,
        device=device,
        seed=args.seed,
        phase_offset=0.0,
        initial_column=0,
        max_steps=args.max_steps,
        planner_depth=args.planner_depth,
        mode="teacher",
        rng=rng,
    )
    semantic_exact = evaluate_semantic_probe(
        probe=semantic_probe,
        mean=semantic_mean,
        std=semantic_std,
        data=exact_teacher,
        device=device,
    )

    print("\n[5] initial recovery decoder...", flush=True)
    decoder, train_report = fit_decoder(
        policy=policy,
        data=initial_replay,
        device=device,
        epochs=args.decoder_epochs,
        learning_rate=args.decoder_lr,
        batch_size=args.decoder_batch,
        seed=args.rng_seed,
    )
    initial_teacher_forced = action_metrics(decoder, exact_teacher, device)
    initial_metrics = evaluate_suite(
        policy=policy,
        decoder=decoder,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        smoke=args.smoke,
    )
    print(
        f"  init: exact={initial_metrics['exactExpo']['steps']}/{args.max_steps} "
        f"score={initial_metrics['exactExpo']['score']:.1f} "
        f"robust={initial_metrics['robust']['reachedStepLimit']}/"
        f"{initial_metrics['robust']['episodes']} "
        f"teacherForcedAcceptable={initial_teacher_forced['acceptableActionRate']:.3f}",
        flush=True,
    )

    config = {
        "seed": args.seed,
        "rngSeed": args.rng_seed,
        "plannerDepth": args.planner_depth,
        "maxSteps": args.max_steps,
        "anchorEpisodes": args.anchor_episodes,
        "perturbEpisodes": args.perturb_episodes,
        "onpolicyEpisodes": args.onpolicy_episodes,
        "rounds": args.rounds,
        "decoderEpochs": args.decoder_epochs,
        "decoderLearningRate": args.decoder_lr,
        "decoderBatch": args.decoder_batch,
        "perturbProbability": args.perturb_probability,
        "recoveryHorizon": args.recovery_horizon,
        "MaleCNSRetrained": False,
    }

    best_decoder = clone_decoder(decoder, device)
    best_metrics = initial_metrics
    best_stage = "initial-recovery"
    save_decoder(
        args.out / "best_decoder.pt",
        decoder=best_decoder,
        source_checkpoint=args.checkpoint,
        stage=best_stage,
        metrics=best_metrics,
        config=config,
    )

    rounds_report: list[dict[str, Any]] = []
    for round_index in range(1, args.rounds + 1):
        print(f"\n[round {round_index}/{args.rounds}] on-policy recovery collection...", flush=True)
        onpolicy, onpolicy_collection = collect_many(
            policy=policy,
            device=device,
            seed=args.seed,
            episodes=args.onpolicy_episodes,
            max_steps=args.max_steps,
            planner_depth=args.planner_depth,
            mode="onpolicy",
            rng=rng,
            decoder=best_decoder,
            include_exact_first=True,
        )
        replay_parts.append(onpolicy)
        replay = concat_data(replay_parts)

        student_semantic = evaluate_semantic_probe(
            probe=semantic_probe,
            mean=semantic_mean,
            std=semantic_std,
            data=onpolicy,
            device=device,
        )
        print(
            "  semantic on student states: "
            f"lane={student_semantic['currentLaneAccuracy']:.3f} "
            f"next={student_semantic['nextLaneAccuracy']:.3f} "
            f"support={student_semantic['supportAccuracy']:.3f}",
            flush=True,
        )

        candidate, candidate_train = fit_decoder(
            policy=policy,
            data=replay,
            device=device,
            epochs=args.decoder_epochs,
            learning_rate=args.decoder_lr,
            batch_size=args.decoder_batch,
            seed=args.rng_seed + round_index,
        )
        teacher_forced = action_metrics(candidate, exact_teacher, device)
        metrics = evaluate_suite(
            policy=policy,
            decoder=candidate,
            device=device,
            seed=args.seed,
            max_steps=args.max_steps,
            smoke=args.smoke,
        )
        improved = selection_key(metrics) > selection_key(best_metrics)
        print(
            f"  round {round_index}: exact={metrics['exactExpo']['steps']}/{args.max_steps} "
            f"score={metrics['exactExpo']['score']:.1f} "
            f"robust={metrics['robust']['reachedStepLimit']}/"
            f"{metrics['robust']['episodes']} "
            f"mean={metrics['robust']['meanScore']:.1f} "
            f"TFacceptable={teacher_forced['acceptableActionRate']:.3f} "
            f"{'NEW BEST' if improved else 'keep best'}",
            flush=True,
        )

        save_decoder(
            args.out / f"round-{round_index}.pt",
            decoder=candidate,
            source_checkpoint=args.checkpoint,
            stage=f"recovery-round-{round_index}",
            metrics=metrics,
            config=config,
        )
        if improved:
            best_decoder = clone_decoder(candidate, device)
            best_metrics = metrics
            best_stage = f"recovery-round-{round_index}"
            save_decoder(
                args.out / "best_decoder.pt",
                decoder=best_decoder,
                source_checkpoint=args.checkpoint,
                stage=best_stage,
                metrics=best_metrics,
                config=config,
            )

        rounds_report.append(
            {
                "round": round_index,
                "collection": onpolicy_collection,
                "studentSemanticProbe": student_semantic,
                "replaySamples": replay.length,
                "training": candidate_train,
                "teacherForcedExact": teacher_forced,
                "closedLoop": metrics,
                "selected": improved,
            }
        )

        robust_target = 2 if args.smoke else 18
        if (
            best_metrics["exactExpo"]["reachedStepLimit"]
            and best_metrics["robust"]["reachedStepLimit"] >= robust_target
        ):
            print("  early stop: expo specialization target reached.", flush=True)
            break

    final_teacher_forced = action_metrics(best_decoder, exact_teacher, device)
    elapsed = time.perf_counter() - started
    report = {
        "version": "crossy-v4-recovery-dagger-1",
        "sourceCheckpoint": str(args.checkpoint.resolve()),
        "metadata": metadata,
        "config": config,
        "frozenMaleCNS": True,
        "trainableDecoderParameters": sum(p.numel() for p in best_decoder.parameters()),
        "anchorCollection": anchor_collection,
        "perturbCollection": perturb_collection,
        "semanticProbe": {
            "train": semantic_train_report,
            "heldoutExactTeacherTrajectory": semantic_exact,
        },
        "exactTeacherCollection": exact_teacher_collection,
        "initial": {
            "training": train_report,
            "teacherForcedExact": initial_teacher_forced,
            "closedLoop": initial_metrics,
        },
        "rounds": rounds_report,
        "selectedStage": best_stage,
        "selectedTeacherForcedExact": final_teacher_forced,
        "selectedClosedLoop": best_metrics,
        "bestDecoder": str((args.out / "best_decoder.pt").resolve()),
        "elapsedSeconds": elapsed,
        "interpretation": (
            "MaleCNS remained frozen. Improvement therefore comes only from a "
            "small motor readout trained on off-trajectory recovery states. "
            "Compare teacher-forced action quality, student-state semantic probe "
            "quality and closed-loop survival to decide whether recovery/readout "
            "is sufficient or the sensory/core representation fails off-policy."
        ),
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== RECOVERY DAGGER COMPLETE ===", flush=True)
    print(f"selected stage: {best_stage}", flush=True)
    print(
        f"exact: {best_metrics['exactExpo']['steps']}/{args.max_steps} "
        f"score={best_metrics['exactExpo']['score']:.1f} "
        f"terminal={best_metrics['exactExpo']['terminalReason']}",
        flush=True,
    )
    print(
        f"robust: {best_metrics['robust']['reachedStepLimit']}/"
        f"{best_metrics['robust']['episodes']} "
        f"mean={best_metrics['robust']['meanScore']:.1f}",
        flush=True,
    )
    print(
        "teacher-forced selected: "
        f"exact={final_teacher_forced['exactActionRate']:.3f} "
        f"acceptable={final_teacher_forced['acceptableActionRate']:.3f} "
        f"safe={final_teacher_forced['immediateSafeActionRate']:.3f}",
        flush=True,
    )
    print(f"best decoder: {(args.out / 'best_decoder.pt').resolve()}", flush=True)
    print(f"report: {report_path.resolve()}", flush=True)
    print(f"elapsed: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
