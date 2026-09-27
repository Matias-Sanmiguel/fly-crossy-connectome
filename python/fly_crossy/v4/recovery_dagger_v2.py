from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
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
class RecoveryDataV2:
    motor: np.ndarray
    acceptable: np.ndarray
    safe: np.ndarray
    progress: np.ndarray
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


class RecoveryDecoderV2(nn.Module):
    """Small motor readout on top of a completely frozen MaleCNS.

    `linear` is the same 708->5 family tested in Recovery DAgger V1.
    `mlp64` adds a 64-unit residual branch.  The residual output starts at
    exactly zero, so both architectures begin from the same converted random
    V4 readout rather than giving the MLP a different starting policy.
    """

    def __init__(
        self,
        motor_dim: int,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        architecture: str,
    ) -> None:
        super().__init__()
        if architecture not in {"linear", "mlp64"}:
            raise ValueError(f"Unknown decoder architecture: {architecture}")
        self.architecture = architecture
        self.motor_dim = int(motor_dim)
        self.register_buffer("mean", mean.reshape(1, motor_dim).float().clone())
        self.register_buffer("std", std.reshape(1, motor_dim).float().clamp_min(1e-4).clone())
        self.linear = nn.Linear(motor_dim, ACTION_COUNT, bias=True)
        if architecture == "mlp64":
            self.hidden = nn.Linear(motor_dim, 64, bias=True)
            self.residual = nn.Linear(64, ACTION_COUNT, bias=True)
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)
        else:
            self.hidden = None
            self.residual = None

    def forward(self, motor: torch.Tensor) -> torch.Tensor:
        normalized = (motor - self.mean) / self.std
        logits = self.linear(normalized)
        if self.hidden is not None and self.residual is not None:
            logits = logits + self.residual(F.gelu(self.hidden(normalized)))
        return logits


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
    policy = FullMaleCNSRGBPolicy(graph, sensory_ids, motor_ids, seed=rng_seed).to(device)
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


def _to_data(rows: list[dict[str, Any]]) -> RecoveryDataV2:
    if not rows:
        raise ValueError("Cannot build RecoveryDataV2 from zero rows.")
    return RecoveryDataV2(
        motor=np.asarray([row["motor"] for row in rows], dtype=np.float32),
        acceptable=np.asarray([row["acceptable"] for row in rows], dtype=np.bool_),
        safe=np.asarray([row["safe"] for row in rows], dtype=np.bool_),
        progress=np.asarray([row["progress"] for row in rows], dtype=np.bool_),
        pair_sign=np.asarray([row["pair_sign"] for row in rows], dtype=np.int8),
        pair_weight=np.asarray([row["pair_weight"] for row in rows], dtype=np.float32),
        best_action=np.asarray([row["best_action"] for row in rows], dtype=np.int64),
        semantic=np.asarray([row["semantic"] for row in rows], dtype=np.float32),
        behavior_action=np.asarray([row["behavior_action"] for row in rows], dtype=np.int64),
        recovery=np.asarray([row["recovery"] for row in rows], dtype=np.bool_),
        source=np.asarray([row["source"] for row in rows]),
    )


def concat_data(parts: list[RecoveryDataV2]) -> RecoveryDataV2:
    if not parts:
        raise ValueError("No RecoveryDataV2 parts supplied.")
    return RecoveryDataV2(
        motor=np.concatenate([part.motor for part in parts], axis=0),
        acceptable=np.concatenate([part.acceptable for part in parts], axis=0),
        safe=np.concatenate([part.safe for part in parts], axis=0),
        progress=np.concatenate([part.progress for part in parts], axis=0),
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
    preferred = np.flatnonzero(safe & ~acceptable)
    preferred = preferred[preferred != best_index]
    if len(preferred):
        return int(rng.choice(preferred))
    alternatives = np.flatnonzero(safe)
    alternatives = alternatives[alternatives != best_index]
    if len(alternatives):
        return int(rng.choice(alternatives))
    return None


def _immediate_progress_mask(state) -> np.ndarray:
    """Actions that survive and increase the all-time score immediately."""
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    for index, action in enumerate(ACTION_ORDER):
        candidate = step_game(state, action).state
        mask[index] = candidate.terminal is None and candidate.score > state.score
    return mask


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
    decoder: RecoveryDecoderV2 | None = None,
    perturb_probability: float = 0.18,
    recovery_horizon: int = 5,
) -> tuple[RecoveryDataV2, dict[str, Any]]:
    if mode not in {"teacher", "perturb", "onpolicy"}:
        raise ValueError(f"Unknown collection mode: {mode}")
    if mode == "onpolicy" and decoder is None:
        raise ValueError("onpolicy collection requires a decoder")

    state = variant_state(seed, phase_offset=phase_offset, initial_column=initial_column)
    neural = policy.zero_state(1, device=device)
    rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    perturbations = 0
    recovery_left = 0
    fatal_behavior = 0
    unacceptable_behavior = 0
    no_progress = 0
    max_no_progress = 0

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

        values, best_index, safe = planner_action_preferences(state, depth=planner_depth)
        acceptable = primary_acceptable_mask(values)
        pair_sign, pair_weight = pairwise_targets(values)
        progress = _immediate_progress_mask(state)

        is_recovery_state = recovery_left > 0
        if recovery_left > 0:
            recovery_left -= 1

        behavior = best_index
        if mode == "onpolicy":
            behavior = int(decoder(motor[None])[0].argmax().item())
        elif mode == "perturb" and not is_recovery_state and rng.random() < perturb_probability:
            candidate = _choose_safe_perturbation(
                acceptable=acceptable,
                safe=safe,
                best_index=best_index,
                rng=rng,
            )
            if candidate is not None:
                behavior = candidate
                perturbations += 1
                recovery_left = max(recovery_left, recovery_horizon)

        rows.append(
            {
                "motor": motor.detach().cpu().numpy(),
                "acceptable": acceptable,
                "safe": safe,
                "progress": progress,
                "pair_sign": pair_sign,
                "pair_weight": pair_weight,
                "best_action": best_index,
                "semantic": semantic_target(state),
                "behavior_action": behavior,
                "recovery": is_recovery_state,
                "source": mode,
            }
        )

        behavior_unacceptable = not bool(acceptable[behavior])
        behavior_unsafe = not bool(safe[behavior])
        unacceptable_behavior += int(behavior_unacceptable)
        fatal_behavior += int(behavior_unsafe)
        if mode == "onpolicy" and (behavior_unacceptable or behavior_unsafe):
            recovery_left = max(recovery_left, recovery_horizon)

        action_counts[ACTION_NAMES[behavior]] += 1
        previous_score = float(state.score)
        state = step_game(state, ACTION_ORDER[behavior]).state
        if float(state.score) > previous_score:
            no_progress = 0
        else:
            no_progress += 1
            max_no_progress = max(max_no_progress, no_progress)

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
        "maxNoProgressStreak": max_no_progress,
        "actions": dict(sorted(action_counts.items())),
    }


def compute_training_weights(data: RecoveryDataV2) -> tuple[np.ndarray, dict[str, Any]]:
    """Weight state difficulty/source, not action class frequency.

    V1 upweighted rare action classes (WAIT/BACKWARD).  Because preference loss
    already compares every action at every state, that can distort the policy.
    V2 intentionally removes class balancing and spends its extra weight on
    actual off-policy/recovery and unsafe-decision states instead.
    """
    result = np.ones(data.length, dtype=np.float32)

    source_scale = np.ones(data.length, dtype=np.float32)
    source_scale[data.source == "teacher"] = 0.75
    source_scale[data.source == "perturb"] = 1.20
    source_scale[data.source == "onpolicy"] = 2.25
    result *= source_scale
    result *= np.where(data.recovery, 2.0, 1.0).astype(np.float32)

    rows = np.arange(data.length)
    valid_behavior = data.behavior_action >= 0
    behavior = np.clip(data.behavior_action, 0, ACTION_COUNT - 1)
    unacceptable = np.zeros(data.length, dtype=np.bool_)
    unsafe = np.zeros(data.length, dtype=np.bool_)
    unacceptable[valid_behavior] = ~data.acceptable[rows[valid_behavior], behavior[valid_behavior]]
    unsafe[valid_behavior] = ~data.safe[rows[valid_behavior], behavior[valid_behavior]]
    result *= np.where(unacceptable, 2.5, 1.0).astype(np.float32)
    result *= np.where(unsafe, 6.0, 1.0).astype(np.float32)

    singleton = data.acceptable.sum(axis=1) == 1
    result *= np.where(singleton, 1.20, 1.0).astype(np.float32)
    preferred_progress = data.acceptable & data.progress
    progress_opportunity = preferred_progress.any(axis=1)
    result *= np.where(progress_opportunity, 1.35, 1.0).astype(np.float32)
    result = np.clip(result, 0.40, 20.0).astype(np.float32)

    best_counts = np.bincount(data.best_action, minlength=ACTION_COUNT)
    return result, {
        "samples": data.length,
        "bestActionCounts": {
            ACTION_NAMES[index]: int(best_counts[index]) for index in range(ACTION_COUNT)
        },
        "classBalancing": False,
        "sourceCounts": dict(sorted(Counter(data.source.tolist()).items())),
        "recoveryFrames": int(data.recovery.sum()),
        "unacceptableBehaviorFrames": int(unacceptable.sum()),
        "unsafeBehaviorFrames": int(unsafe.sum()),
        "progressOpportunityFrames": int(progress_opportunity.sum()),
        "meanWeight": float(result.mean()),
        "maxWeight": float(result.max()),
    }


def progress_preference_loss(
    logits: torch.Tensor,
    acceptable: torch.Tensor,
    progress: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Prefer immediate progress only when the planner also accepts it.

    This does not force FORWARD through danger.  A state participates only when
    at least one planner-acceptable action is both safe-by-construction in the
    planner preference set and immediately increases score.
    """
    preferred = acceptable & progress
    eligible = preferred.any(dim=1)
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    selected_logits = logits[eligible]
    selected_preferred = preferred[eligible]
    masked = selected_logits.masked_fill(~selected_preferred, -1e9)
    each = torch.logsumexp(selected_logits, dim=1) - torch.logsumexp(masked, dim=1)
    selected_weight = sample_weight[eligible]
    normalized = selected_weight / selected_weight.mean().clamp_min(1e-6)
    return (each * normalized).mean()


def _init_decoder_from_random_readout(
    *,
    policy: FullMaleCNSRGBPolicy,
    data: RecoveryDataV2,
    device: torch.device,
    architecture: str,
    seed: int,
) -> RecoveryDecoderV2:
    torch.manual_seed(seed)
    motor = torch.as_tensor(data.motor, dtype=torch.float32, device=device)
    mean = motor.mean(dim=0)
    std = motor.std(dim=0).clamp_min(1e-4)
    decoder = RecoveryDecoderV2(
        motor.shape[1], mean=mean, std=std, architecture=architecture
    ).to(device)
    with torch.no_grad():
        original = policy.decoder.to(device)
        decoder.linear.weight.copy_(original * std[None, :])
        decoder.linear.bias.copy_(mean @ original.T)
    return decoder


def fit_decoder(
    *,
    policy: FullMaleCNSRGBPolicy,
    data: RecoveryDataV2,
    device: torch.device,
    architecture: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    progress_loss_weight: float,
) -> tuple[RecoveryDecoderV2, dict[str, Any]]:
    weights_np, weight_report = compute_training_weights(data)
    decoder = _init_decoder_from_random_readout(
        policy=policy,
        data=data,
        device=device,
        architecture=architecture,
        seed=seed,
    )
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
        totals, prefs, progresses = [], [], []
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
                raise RuntimeError("Non-finite recovery decoder V2 loss.")
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
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"  {architecture:6s} epoch {epoch:3d}/{epochs} "
                f"loss={record['total']:.4f} pref={record['preference']:.4f} "
                f"progress={record['progress']:.4f}",
                flush=True,
            )

    return decoder, {
        "architecture": architecture,
        "trainableParameters": sum(p.numel() for p in decoder.parameters()),
        "epochs": epochs,
        "learningRate": learning_rate,
        "batchSize": batch_size,
        "progressLossWeight": progress_loss_weight,
        "first": history[0],
        "last": history[-1],
        "weights": weight_report,
    }


@torch.no_grad()
def action_metrics(
    decoder: RecoveryDecoderV2,
    data: RecoveryDataV2,
    device: torch.device,
) -> dict[str, Any]:
    motor = torch.as_tensor(data.motor, dtype=torch.float32, device=device)
    predictions = decoder(motor).argmax(dim=1).cpu().numpy()
    rows = np.arange(data.length)
    preferred_progress = data.acceptable & data.progress
    eligible = preferred_progress.any(axis=1)
    progress_choice = np.zeros(data.length, dtype=np.bool_)
    progress_choice[eligible] = preferred_progress[rows[eligible], predictions[eligible]]
    return {
        "frames": data.length,
        "exactActionRate": float(np.mean(predictions == data.best_action)),
        "acceptableActionRate": float(np.mean(data.acceptable[rows, predictions])),
        "immediateSafeActionRate": float(np.mean(data.safe[rows, predictions])),
        "progressOpportunityFrames": int(eligible.sum()),
        "progressChoiceRate": float(progress_choice[eligible].mean()) if eligible.any() else 1.0,
        "actions": dict(sorted(Counter(ACTION_NAMES[index] for index in predictions).items())),
    }


@torch.no_grad()
def evaluate_variant(
    *,
    policy: FullMaleCNSRGBPolicy,
    decoder: RecoveryDecoderV2,
    device: torch.device,
    seed: str,
    phase_offset: float,
    initial_column: int,
    max_steps: int,
    target_score: float,
    stall_steps: int,
) -> dict[str, Any]:
    state = variant_state(seed, phase_offset=phase_offset, initial_column=initial_column)
    neural = policy.zero_state(1, device=device)
    actions: Counter[str] = Counter()
    steps = 0
    no_progress = 0
    max_no_progress = 0
    stalled = False
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
        previous_score = float(state.score)
        state = step_game(state, action).state
        steps += 1
        if float(state.score) > previous_score:
            no_progress = 0
        else:
            no_progress += 1
            max_no_progress = max(max_no_progress, no_progress)
        if state.terminal is None and no_progress >= stall_steps:
            stalled = True
            break

    reached_limit = bool(state.terminal is None and steps >= max_steps)
    progress_qualified = float(state.score) >= float(target_score)
    success = bool(reached_limit and progress_qualified and not stalled)
    return {
        "phaseOffset": float(phase_offset),
        "initialColumn": int(initial_column),
        "steps": steps,
        "score": float(state.score),
        "targetScore": float(target_score),
        "scoreRatioToTarget": float(state.score) / max(1e-6, float(target_score)),
        "reachedStepLimit": reached_limit,
        "progressQualified": progress_qualified,
        "success": success,
        "stalled": stalled,
        "maxNoProgressStreak": int(max_no_progress),
        "terminalReason": "stagnation" if stalled else state.terminal,
        "actions": dict(sorted(actions.items())),
    }


@torch.no_grad()
def evaluate_suite(
    *,
    policy: FullMaleCNSRGBPolicy,
    decoder: RecoveryDecoderV2,
    device: torch.device,
    seed: str,
    max_steps: int,
    target_score: float,
    stall_steps: int,
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
            target_score=target_score,
            stall_steps=stall_steps,
        )
        for phase, column in robustness_configs(smoke)
    ]
    exact = next(
        row for row in variants if row["phaseOffset"] == 0.0 and row["initialColumn"] == 0
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
            "successes": int(sum(row["success"] for row in variants)),
            "reachedStepLimit": int(sum(row["reachedStepLimit"] for row in variants)),
            "progressQualified": int(sum(row["progressQualified"] for row in variants)),
            "stalled": int(sum(row["stalled"] for row in variants)),
            "meanScore": float(np.mean(scores)),
            "medianScore": float(np.median(scores)),
            "terminalReasons": dict(sorted(terminals.items())),
            "actions": dict(sorted(actions.items())),
        },
        "perVariant": variants,
    }


def selection_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    """Progress-first model selection; survival without progress cannot win."""
    exact = metrics["exactExpo"]
    robust = metrics["robust"]
    return (
        float(exact["success"]),
        float(exact["score"]),
        float(robust["successes"]),
        float(robust["meanScore"]),
        float(robust["medianScore"]),
        -float(robust["stalled"]),
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
    data: RecoveryDataV2,
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
    data: RecoveryDataV2,
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
    decoder: RecoveryDecoderV2 | None = None,
    perturb_probability: float = 0.18,
    recovery_horizon: int = 5,
    include_exact_first: bool = False,
) -> tuple[RecoveryDataV2, list[dict[str, Any]]]:
    parts: list[RecoveryDataV2] = []
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
                f"  {mode} {index + 1}/{episodes} frames={sum(part.length for part in parts):,}",
                flush=True,
            )
    return concat_data(parts), reports


def save_decoder(
    path: Path,
    *,
    decoder: RecoveryDecoderV2,
    source_checkpoint: Path,
    stage: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "decoder": decoder.state_dict(),
            "motorDim": decoder.motor_dim,
            "architecture": decoder.architecture,
            "sourceCheckpoint": str(source_checkpoint.resolve()),
            "stage": stage,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def clone_decoder(decoder: RecoveryDecoderV2, device: torch.device) -> RecoveryDecoderV2:
    copy = RecoveryDecoderV2(
        decoder.motor_dim,
        mean=decoder.mean.detach().to(device).squeeze(0),
        std=decoder.std.detach().to(device).squeeze(0),
        architecture=decoder.architecture,
    ).to(device)
    copy.load_state_dict(decoder.state_dict(), strict=True)
    copy.eval()
    return copy


def train_architecture_candidates(
    *,
    policy: FullMaleCNSRGBPolicy,
    replay: RecoveryDataV2,
    exact_teacher: RecoveryDataV2,
    device: torch.device,
    seed: str,
    max_steps: int,
    target_score: float,
    stall_steps: int,
    smoke: bool,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    rng_seed: int,
    progress_loss_weight: float,
) -> tuple[RecoveryDecoderV2, dict[str, Any], list[dict[str, Any]]]:
    architectures = ("linear", "mlp64")
    rows: list[dict[str, Any]] = []
    best_decoder: RecoveryDecoderV2 | None = None
    best_row: dict[str, Any] | None = None

    for arch_index, architecture in enumerate(architectures):
        decoder, training = fit_decoder(
            policy=policy,
            data=replay,
            device=device,
            architecture=architecture,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=rng_seed + arch_index * 1000,
            progress_loss_weight=progress_loss_weight,
        )
        teacher_forced = action_metrics(decoder, exact_teacher, device)
        closed_loop = evaluate_suite(
            policy=policy,
            decoder=decoder,
            device=device,
            seed=seed,
            max_steps=max_steps,
            target_score=target_score,
            stall_steps=stall_steps,
            smoke=smoke,
        )
        row = {
            "architecture": architecture,
            "training": training,
            "teacherForcedExact": teacher_forced,
            "closedLoop": closed_loop,
        }
        rows.append(row)
        exact = closed_loop["exactExpo"]
        robust = closed_loop["robust"]
        print(
            f"  {architecture:6s}: exactScore={exact['score']:.1f} "
            f"steps={exact['steps']}/{max_steps} success={int(exact['success'])} "
            f"robust={robust['successes']}/{robust['episodes']} "
            f"mean={robust['meanScore']:.1f} "
            f"TFacc={teacher_forced['acceptableActionRate']:.3f} "
            f"TFprogress={teacher_forced['progressChoiceRate']:.3f}",
            flush=True,
        )
        if best_row is None or selection_key(closed_loop) > selection_key(best_row["closedLoop"]):
            best_decoder = clone_decoder(decoder, device)
            best_row = row

    assert best_decoder is not None and best_row is not None
    return best_decoder, best_row, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V4 Recovery DAgger V2: freeze MaleCNS; train progress-aware linear/MLP "
            "motor readouts on teacher, perturbation and on-policy recovery states."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=EXPO_SEED)
    parser.add_argument("--rng-seed", type=int, default=606)
    parser.add_argument("--planner-depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--anchor-episodes", type=int, default=12)
    parser.add_argument("--perturb-episodes", type=int, default=16)
    parser.add_argument("--onpolicy-episodes", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--decoder-epochs", type=int, default=100)
    parser.add_argument("--decoder-lr", type=float, default=0.008)
    parser.add_argument("--decoder-batch", type=int, default=1024)
    parser.add_argument("--progress-loss-weight", type=float, default=1.0)
    parser.add_argument("--perturb-probability", type=float, default=0.20)
    parser.add_argument("--recovery-horizon", type=int, default=6)
    parser.add_argument("--stall-steps", type=int, default=12)
    parser.add_argument("--min-teacher-fraction", type=float, default=0.80)
    parser.add_argument("--min-score", type=float, default=None)
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
        default=repo_root() / "runs" / "crossy-v4-recovery-dagger-v2",
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
        args.stall_steps = min(args.stall_steps, 8)

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

    print("=== CROSSY V4 / RECOVERY DAGGER V2 ===", flush=True)
    print(f"source checkpoint: {args.checkpoint.resolve()}", flush=True)
    print(
        f"MaleCNS frozen: {metadata['neurons']:,} neurons / {metadata['edges']:,} edges / "
        f"trainable core params={trainable_policy_params}",
        flush=True,
    )
    print("readout ablation: linear vs residual MLP64", flush=True)

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

    print("\n[2] deliberately perturbed recoverable trajectories...", flush=True)
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

    print("\n[3] fresh semantic probe on teacher anchors...", flush=True)
    semantic_probe, semantic_mean, semantic_std, semantic_train_report = fit_semantic_probe(
        data=anchor,
        device=device,
        epochs=args.semantic_probe_epochs,
        learning_rate=args.semantic_probe_lr,
    )

    print("\n[4] exact teacher holdout + real progress target...", flush=True)
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
    teacher_score = float(exact_teacher_collection["score"])
    teacher_max_no_progress = max(
        [int(exact_teacher_collection.get("maxNoProgressStreak", 0))]
        + [int(row.get("maxNoProgressStreak", 0)) for row in anchor_collection]
    )
    effective_stall_steps = max(int(args.stall_steps), teacher_max_no_progress + 2)
    target_score = (
        float(args.min_score)
        if args.min_score is not None
        else float(args.min_teacher_fraction) * teacher_score
    )
    semantic_exact = evaluate_semantic_probe(
        probe=semantic_probe,
        mean=semantic_mean,
        std=semantic_std,
        data=exact_teacher,
        device=device,
    )
    print(
        f"  exact teacher score={teacher_score:.1f}; success requires "
        f"score>={target_score:.1f} AND {args.max_steps} steps; "
        f"stagnation cutoff={effective_stall_steps} "
        f"(teacher max streak={teacher_max_no_progress})",
        flush=True,
    )

    replay_parts: list[RecoveryDataV2] = [anchor, perturb]
    replay = concat_data(replay_parts)

    print("\n[5] initial readout ablation...", flush=True)
    best_decoder, initial_best, initial_candidates = train_architecture_candidates(
        policy=policy,
        replay=replay,
        exact_teacher=exact_teacher,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        target_score=target_score,
        stall_steps=effective_stall_steps,
        smoke=args.smoke,
        epochs=args.decoder_epochs,
        learning_rate=args.decoder_lr,
        batch_size=args.decoder_batch,
        rng_seed=args.rng_seed,
        progress_loss_weight=args.progress_loss_weight,
    )
    best_metrics = initial_best["closedLoop"]
    best_stage = f"initial-{best_decoder.architecture}"

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
        "progressLossWeight": args.progress_loss_weight,
        "perturbProbability": args.perturb_probability,
        "recoveryHorizon": args.recovery_horizon,
        "requestedStallSteps": args.stall_steps,
        "effectiveStallSteps": effective_stall_steps,
        "teacherMaxNoProgressStreak": teacher_max_no_progress,
        "teacherScore": teacher_score,
        "minimumProgressScore": target_score,
        "minimumTeacherFraction": args.min_teacher_fraction,
        "MaleCNSRetrained": False,
        "selectionRule": "exact real-progress success -> exact score -> robust real-progress successes -> robust score",
    }
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
            recovery_horizon=args.recovery_horizon,
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

        candidate, candidate_best, candidate_rows = train_architecture_candidates(
            policy=policy,
            replay=replay,
            exact_teacher=exact_teacher,
            device=device,
            seed=args.seed,
            max_steps=args.max_steps,
            target_score=target_score,
            stall_steps=effective_stall_steps,
            smoke=args.smoke,
            epochs=args.decoder_epochs,
            learning_rate=args.decoder_lr,
            batch_size=args.decoder_batch,
            rng_seed=args.rng_seed + round_index * 17,
            progress_loss_weight=args.progress_loss_weight,
        )
        metrics = candidate_best["closedLoop"]
        improved = selection_key(metrics) > selection_key(best_metrics)
        exact = metrics["exactExpo"]
        robust = metrics["robust"]
        print(
            f"  round {round_index} winner={candidate.architecture}: "
            f"exactScore={exact['score']:.1f} steps={exact['steps']}/{args.max_steps} "
            f"success={int(exact['success'])} robust={robust['successes']}/{robust['episodes']} "
            f"mean={robust['meanScore']:.1f} {'NEW BEST' if improved else 'keep best'}",
            flush=True,
        )

        for row in candidate_rows:
            architecture = row["architecture"]
            # Refit object is not retained for each non-winning architecture, so
            # detailed metrics go to JSON while only the round winner is saved.
            row["selectedWithinRound"] = architecture == candidate.architecture

        save_decoder(
            args.out / f"round-{round_index}-winner.pt",
            decoder=candidate,
            source_checkpoint=args.checkpoint,
            stage=f"recovery-v2-round-{round_index}-{candidate.architecture}",
            metrics=metrics,
            config=config,
        )
        if improved:
            best_decoder = clone_decoder(candidate, device)
            best_metrics = metrics
            best_stage = f"recovery-v2-round-{round_index}-{candidate.architecture}"
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
                "architectureCandidates": candidate_rows,
                "roundWinner": candidate.architecture,
                "selectedGlobally": improved,
            }
        )

        robust_target = 1 if args.smoke else 18
        if best_metrics["exactExpo"]["success"] and best_metrics["robust"]["successes"] >= robust_target:
            print("  early stop: real-progress expo specialization target reached.", flush=True)
            break

    final_teacher_forced = action_metrics(best_decoder, exact_teacher, device)
    elapsed = time.perf_counter() - started
    report = {
        "version": "crossy-v4-recovery-dagger-v2",
        "sourceCheckpoint": str(args.checkpoint.resolve()),
        "metadata": metadata,
        "config": config,
        "frozenMaleCNS": True,
        "selectedDecoderArchitecture": best_decoder.architecture,
        "trainableDecoderParameters": sum(p.numel() for p in best_decoder.parameters()),
        "anchorCollection": anchor_collection,
        "perturbCollection": perturb_collection,
        "semanticProbe": {
            "train": semantic_train_report,
            "heldoutExactTeacherTrajectory": semantic_exact,
        },
        "exactTeacherCollection": exact_teacher_collection,
        "initialArchitectureCandidates": initial_candidates,
        "selectedStage": best_stage,
        "selectedTeacherForcedExact": final_teacher_forced,
        "selectedClosedLoop": best_metrics,
        "rounds": rounds_report,
        "bestDecoder": str((args.out / "best_decoder.pt").resolve()),
        "elapsedSeconds": elapsed,
        "interpretation": (
            "MaleCNS stayed frozen. Unlike V1, model selection requires real progress, "
            "stagnation is an explicit failure, action-class balancing is removed, "
            "planner-acceptable immediate progress receives an auxiliary preference loss, "
            "and linear versus residual-MLP motor readouts are compared directly."
        ),
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    exact = best_metrics["exactExpo"]
    robust = best_metrics["robust"]
    print("\n=== V2 SELECTED ===", flush=True)
    print(f"stage: {best_stage}", flush=True)
    print(f"architecture: {best_decoder.architecture}", flush=True)
    print(
        f"exact: score={exact['score']:.1f}/{target_score:.1f} "
        f"steps={exact['steps']}/{args.max_steps} success={exact['success']} "
        f"stalled={exact['stalled']}",
        flush=True,
    )
    print(
        f"robust: successes={robust['successes']}/{robust['episodes']} "
        f"meanScore={robust['meanScore']:.1f} stalled={robust['stalled']}",
        flush=True,
    )
    print(f"report: {report_path.resolve()}", flush=True)
    print(f"elapsed: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
