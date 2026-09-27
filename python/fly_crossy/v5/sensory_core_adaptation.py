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
from fly_crossy.v4.exact_seed_branch_curriculum import load_decoder_checkpoint
from fly_crossy.v4.policy import (
    CHANNELS,
    FEATURES,
    IMAGE_H,
    IMAGE_W,
    FullMaleCNSRGBPolicy,
)
from fly_crossy.v4.recovery_dagger_v2 import (
    ACTION_COUNT,
    ACTION_NAMES,
    RecoveryDecoderV2,
    _immediate_progress_mask,
    evaluate_suite,
    evaluate_variant,
    progress_preference_loss,
    repo_root,
)
from fly_crossy.v4.train_expo_specialist import (
    EXPO_SEED,
    default_flyhard_root,
    resize_rgb,
    robustness_configs,
    variant_state,
)

VERSION = "crossy-v5-sensory-core-adaptation-1"


@dataclass
class TrainEpisode:
    images: np.ndarray
    acceptable: np.ndarray
    safe: np.ndarray
    progress: np.ndarray
    pair_sign: np.ndarray
    pair_weight: np.ndarray
    best_action: np.ndarray
    behavior_action: np.ndarray
    reference_logits: np.ndarray
    source: str
    phase_offset: float
    initial_column: int

    @property
    def length(self) -> int:
        return int(len(self.best_action))


class SensoryPlasticPolicy(nn.Module):
    """Measured MaleCNS with a small trainable camera->sensory calibration layer.

    The existing V4 random RGB->ol_sensory drive remains the baseline.  V5 adds:
      * one multiplicative gain per ol_sensory neuron; and
      * a low-rank residual RGB->ol_sensory projection.

    The residual output is initialized to exactly zero and the gain starts at 1,
    so the policy is bit-for-bit behaviorally identical to the loaded V4 policy
    before adaptation begins.  This changes the artificial sensory interface,
    not the action decoder.
    """

    def __init__(
        self,
        base: FullMaleCNSRGBPolicy,
        *,
        rank: int = 16,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.residual_scale = float(residual_scale)
        sensory_count = int(len(base.sensory_ids))

        self.sensory_gain = nn.Parameter(torch.zeros(sensory_count, dtype=torch.float32))
        self.input_projection = nn.Linear(FEATURES, self.rank, bias=False)
        self.output_projection = nn.Linear(self.rank, sensory_count, bias=True)
        nn.init.normal_(self.input_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def zero_state(self, batch: int, *, device=None) -> torch.Tensor:
        return self.base.zero_state(batch, device=device)

    def drive_for(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (
            IMAGE_H,
            IMAGE_W,
            CHANNELS,
        ):
            raise ValueError(
                f"Expected RGB images [B,{IMAGE_H},{IMAGE_W},{CHANNELS}]."
            )
        features = ((images.flatten(1) - 0.5) * 2.0).clamp(-2.0, 2.0)
        baseline = (
            features[:, self.base.feature_ids].T * self.base.input_signs[:, None]
        )
        gain = 1.0 + 0.25 * torch.tanh(self.sensory_gain)[:, None]
        hidden = torch.tanh(self.input_projection(features))
        residual = self.output_projection(hidden).T * self.residual_scale
        sensory_drive = baseline * gain + residual

        drive = torch.zeros(
            self.base.core.n,
            len(images),
            dtype=torch.float32,
            device=images.device,
        )
        drive[self.base.sensory_ids] = sensory_drive
        return drive

    def step_state(self, state: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        return self.base.core(
            state,
            steps=4,
            drive=self.drive_for(images),
        )

    def motor_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.base.motor_state(state)

    def run_window_with_motor(
        self,
        images_seq: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        motors = []
        for k in range(images_seq.shape[0]):
            state = self.step_state(state, images_seq[k])
            motors.append(self.motor_state(state))
        return torch.stack(motors), state


def _load_base_policy(
    *,
    checkpoint_path: Path,
    flyhard_root: Path,
    device: torch.device,
) -> tuple[FullMaleCNSRGBPolicy, dict[str, Any], np.ndarray, np.ndarray]:
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
    return policy, {
        "checkpointStage": checkpoint.get("stage"),
        "rngSeed": rng_seed,
        "neurons": int(len(graph["crow"]) - 1),
        "edges": int(len(graph["col"])),
        "sensory": int(len(sensory_ids)),
        "motor": int(len(motor_ids)),
    }, np.asarray(graph["col"]), sensory_ids


def build_plasticity_masks(
    *,
    base: FullMaleCNSRGBPolicy,
    graph_col: np.ndarray,
    sensory_ids: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Plasticize only the core boundary immediately downstream of ol_sensory.

    CSR `col` indexes presynaptic neurons.  Therefore edges whose col is an
    ol_sensory neuron are outgoing sensory edges.  Leak plasticity is limited to
    sensory neurons and their directly postsynaptic targets.
    """
    edge_mask_np = np.isin(graph_col, sensory_ids)
    rows_np = base.core.rows.detach().cpu().numpy()
    first_hop = np.unique(rows_np[edge_mask_np])
    leak_ids = np.unique(np.concatenate([sensory_ids, first_hop]))
    leak_mask_np = np.zeros(base.core.n, dtype=np.bool_)
    leak_mask_np[leak_ids] = True
    edge_mask = torch.as_tensor(edge_mask_np, dtype=torch.bool, device=device)
    leak_mask = torch.as_tensor(leak_mask_np, dtype=torch.bool, device=device)
    return edge_mask, leak_mask, {
        "sensoryOutgoingEdges": int(edge_mask_np.sum()),
        "sensoryAndFirstHopLeaks": int(leak_mask_np.sum()),
        "firstHopNeurons": int(len(first_hop)),
    }


def configure_trainable(
    policy: SensoryPlasticPolicy,
    *,
    phase: str,
    edge_mask: torch.Tensor,
    leak_mask: torch.Tensor,
) -> list[Any]:
    for p in policy.parameters():
        p.requires_grad_(False)
    for p in (
        policy.sensory_gain,
        policy.input_projection.weight,
        policy.output_projection.weight,
        policy.output_projection.bias,
    ):
        p.requires_grad_(True)

    hooks = []
    if phase == "sensory-core-boundary":
        policy.base.core.edge_gain.requires_grad_(True)
        policy.base.core.leak.requires_grad_(True)
        hooks.append(policy.base.core.edge_gain.register_hook(lambda g: g * edge_mask))
        hooks.append(policy.base.core.leak.register_hook(lambda g: g * leak_mask))
    elif phase != "sensory-only":
        raise ValueError(f"Unknown phase: {phase}")
    return hooks


def _episode(
    *,
    images,
    acceptable,
    safe,
    progress,
    pair_sign,
    pair_weight,
    best_action,
    behavior_action,
    reference_logits,
    source: str,
    phase_offset: float,
    initial_column: int,
) -> TrainEpisode:
    return TrainEpisode(
        images=np.asarray(images, dtype=np.float16),
        acceptable=np.asarray(acceptable, dtype=np.bool_),
        safe=np.asarray(safe, dtype=np.bool_),
        progress=np.asarray(progress, dtype=np.bool_),
        pair_sign=np.asarray(pair_sign, dtype=np.int8),
        pair_weight=np.asarray(pair_weight, dtype=np.float16),
        best_action=np.asarray(best_action, dtype=np.int64),
        behavior_action=np.asarray(behavior_action, dtype=np.int64),
        reference_logits=np.asarray(reference_logits, dtype=np.float32),
        source=source,
        phase_offset=float(phase_offset),
        initial_column=int(initial_column),
    )


@torch.no_grad()
def collect_episode(
    *,
    policy: SensoryPlasticPolicy,
    decoder: RecoveryDecoderV2,
    device: torch.device,
    seed: str,
    phase_offset: float,
    initial_column: int,
    max_steps: int,
    planner_depth: int,
    behavior: str,
) -> tuple[TrainEpisode, dict[str, Any]]:
    if behavior not in {"teacher", "student"}:
        raise ValueError("behavior must be teacher or student")
    state = variant_state(seed, phase_offset=phase_offset, initial_column=initial_column)
    neural = policy.zero_state(1, device=device)

    images = []
    acceptable_rows = []
    safe_rows = []
    progress_rows = []
    signs_rows = []
    pair_weight_rows = []
    best_rows = []
    behavior_rows = []
    ref_logits_rows = []
    actions: Counter[str] = Counter()
    fatal_behavior = 0
    unacceptable_behavior = 0

    for _ in range(max_steps):
        if state.terminal is not None:
            break
        image_np = resize_rgb(render_crossy_neural_frame(state))
        image = torch.as_tensor(image_np[None], dtype=torch.float32, device=device)
        neural = policy.step_state(neural, image)
        motor = policy.motor_state(neural)
        logits = decoder(motor)[0]

        values, best, safe = planner_action_preferences(state, depth=planner_depth)
        acceptable = primary_acceptable_mask(values)
        signs, pair_weights = pairwise_targets(values)
        progress = _immediate_progress_mask(state)
        student = int(logits.argmax().item())
        action_index = int(best if behavior == "teacher" else student)

        images.append(image_np)
        acceptable_rows.append(acceptable)
        safe_rows.append(safe)
        progress_rows.append(progress)
        signs_rows.append(signs)
        pair_weight_rows.append(pair_weights)
        best_rows.append(int(best))
        behavior_rows.append(action_index)
        ref_logits_rows.append(logits.detach().cpu().numpy().copy())

        if not bool(acceptable[action_index]):
            unacceptable_behavior += 1
        next_state = step_game(state, ACTION_ORDER[action_index]).state
        if next_state.terminal is not None:
            fatal_behavior += 1
        actions[ACTION_NAMES[action_index]] += 1
        state = next_state

    ep = _episode(
        images=images,
        acceptable=acceptable_rows,
        safe=safe_rows,
        progress=progress_rows,
        pair_sign=signs_rows,
        pair_weight=pair_weight_rows,
        best_action=best_rows,
        behavior_action=behavior_rows,
        reference_logits=ref_logits_rows,
        source=behavior,
        phase_offset=phase_offset,
        initial_column=initial_column,
    )
    return ep, {
        "source": behavior,
        "phaseOffset": float(phase_offset),
        "initialColumn": int(initial_column),
        "frames": ep.length,
        "score": float(state.score),
        "terminalReason": state.terminal,
        "fatalBehavior": int(fatal_behavior),
        "unacceptableBehavior": int(unacceptable_behavior),
        "actions": dict(sorted(actions.items())),
    }


def collect_round_dataset(
    *,
    policy: SensoryPlasticPolicy,
    decoder: RecoveryDecoderV2,
    device: torch.device,
    seed: str,
    max_steps: int,
    planner_depth: int,
    smoke: bool,
) -> tuple[list[TrainEpisode], list[dict[str, Any]]]:
    teacher_configs = [(0.0, 0)] if smoke else [(0.0, 0), (0.4, 0)]
    student_configs = [(0.0, 0)] if smoke else [(0.0, 0), (0.2, 0), (0.4, 0), (0.0, -1), (0.0, 1)]
    episodes: list[TrainEpisode] = []
    reports: list[dict[str, Any]] = []
    for behavior, configs in (("teacher", teacher_configs), ("student", student_configs)):
        for phase, column in configs:
            ep, report = collect_episode(
                policy=policy,
                decoder=decoder,
                device=device,
                seed=seed,
                phase_offset=phase,
                initial_column=column,
                max_steps=max_steps,
                planner_depth=planner_depth,
                behavior=behavior,
            )
            episodes.append(ep)
            reports.append(report)
    return episodes, reports


def episode_weights(ep: TrainEpisode) -> np.ndarray:
    rows = np.arange(ep.length)
    acceptable_behavior = ep.acceptable[rows, ep.behavior_action]
    safe_behavior = ep.safe[rows, ep.behavior_action]
    weights = np.full(ep.length, 0.8 if ep.source == "teacher" else 1.5, dtype=np.float32)
    if ep.source == "student":
        weights *= np.where(~acceptable_behavior, 4.0, 1.0).astype(np.float32)
        weights *= np.where(~safe_behavior, 12.0, 1.0).astype(np.float32)
        tail = np.zeros(ep.length, dtype=np.float32)
        tail[max(0, ep.length - 10):] = 1.0
        weights *= np.where(tail > 0, 2.5, 1.0).astype(np.float32)
    return np.clip(weights, 0.4, 20.0)


def _padded_chunk(
    group: list[TrainEpisode],
    weights: list[np.ndarray],
    start: int,
    window: int,
):
    batch = len(group)
    length = min(window, max(ep.length for ep in group) - start)
    images = np.zeros((length, batch, IMAGE_H, IMAGE_W, CHANNELS), dtype=np.float32)
    acceptable = np.zeros((length, batch, ACTION_COUNT), dtype=np.bool_)
    safe = np.ones((length, batch, ACTION_COUNT), dtype=np.bool_)
    progress = np.zeros((length, batch, ACTION_COUNT), dtype=np.bool_)
    pair_count = ACTION_COUNT * (ACTION_COUNT - 1) // 2
    signs = np.zeros((length, batch, pair_count), dtype=np.int8)
    pair_weights = np.zeros((length, batch, pair_count), dtype=np.float32)
    sample_weights = np.ones((length, batch), dtype=np.float32)
    reference_logits = np.zeros((length, batch, ACTION_COUNT), dtype=np.float32)
    protected = np.zeros((length, batch), dtype=np.bool_)
    mask = np.zeros((length, batch), dtype=np.bool_)

    for b, ep in enumerate(group):
        end = min(ep.length, start + length)
        take = max(0, end - start)
        if take == 0:
            continue
        sl = slice(start, end)
        images[:take, b] = ep.images[sl]
        acceptable[:take, b] = ep.acceptable[sl]
        safe[:take, b] = ep.safe[sl]
        progress[:take, b] = ep.progress[sl]
        signs[:take, b] = ep.pair_sign[sl]
        pair_weights[:take, b] = ep.pair_weight[sl]
        sample_weights[:take, b] = weights[b][sl]
        reference_logits[:take, b] = ep.reference_logits[sl]
        rows = np.arange(start, end)
        behavior = ep.behavior_action[sl]
        local = np.arange(take)
        if ep.source == "student":
            protected[:take, b] = (
                ep.acceptable[rows, behavior] & ep.safe[rows, behavior]
            )
        mask[:take, b] = True
    return (
        images,
        acceptable,
        safe,
        progress,
        signs,
        pair_weights,
        sample_weights,
        reference_logits,
        protected,
        mask,
    )


def _trainable_snapshot(
    policy: SensoryPlasticPolicy,
    *,
    edge_mask: torch.Tensor,
    leak_mask: torch.Tensor,
) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "sensory_gain": policy.sensory_gain.detach().clone(),
        "input_projection": policy.input_projection.weight.detach().clone(),
        "output_weight": policy.output_projection.weight.detach().clone(),
        "output_bias": policy.output_projection.bias.detach().clone(),
    }
    if policy.base.core.edge_gain.requires_grad:
        snap["edge_gain"] = policy.base.core.edge_gain.detach()[edge_mask].clone()
    if policy.base.core.leak.requires_grad:
        snap["leak"] = policy.base.core.leak.detach()[leak_mask].clone()
    return snap


def _trust_penalty(
    policy: SensoryPlasticPolicy,
    snap: dict[str, Any],
    *,
    edge_mask: torch.Tensor,
    leak_mask: torch.Tensor,
) -> torch.Tensor:
    terms = [
        F.mse_loss(policy.sensory_gain, snap["sensory_gain"]),
        F.mse_loss(policy.input_projection.weight, snap["input_projection"]),
        F.mse_loss(policy.output_projection.weight, snap["output_weight"]),
        F.mse_loss(policy.output_projection.bias, snap["output_bias"]),
    ]
    if "edge_gain" in snap:
        terms.append(F.mse_loss(policy.base.core.edge_gain[edge_mask], snap["edge_gain"]))
    if "leak" in snap:
        terms.append(F.mse_loss(policy.base.core.leak[leak_mask], snap["leak"]))
    return torch.stack(terms).mean()


def train_round(
    *,
    policy: SensoryPlasticPolicy,
    decoder: RecoveryDecoderV2,
    episodes: list[TrainEpisode],
    device: torch.device,
    phase: str,
    epochs: int,
    batch_size: int,
    window: int,
    adapter_lr: float,
    core_lr: float,
    distill_weight: float,
    progress_weight: float,
    trust_weight: float,
    edge_mask: torch.Tensor,
    leak_mask: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    hooks = configure_trainable(
        policy,
        phase=phase,
        edge_mask=edge_mask,
        leak_mask=leak_mask,
    )
    for p in decoder.parameters():
        p.requires_grad_(False)
    decoder.eval()

    adapter_params = [
        policy.sensory_gain,
        policy.input_projection.weight,
        policy.output_projection.weight,
        policy.output_projection.bias,
    ]
    groups = [{"params": adapter_params, "lr": adapter_lr}]
    if phase == "sensory-core-boundary":
        groups.append({"params": [policy.base.core.edge_gain, policy.base.core.leak], "lr": core_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    snap = _trainable_snapshot(policy, edge_mask=edge_mask, leak_mask=leak_mask)
    rng = np.random.default_rng(seed)
    weights = [episode_weights(ep) for ep in episodes]
    history: list[dict[str, float]] = []

    policy.train()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(episodes))
        epoch_losses = []
        epoch_pref = []
        epoch_distill = []
        epoch_trust = []
        for group_start in range(0, len(order), batch_size):
            idx = order[group_start: group_start + batch_size].tolist()
            group = [episodes[i] for i in idx]
            group_weights = [weights[i] for i in idx]
            state = policy.zero_state(len(group), device=device)
            max_length = max(ep.length for ep in group)

            for start in range(0, max_length, window):
                packed = _padded_chunk(group, group_weights, start, window)
                (
                    images_np,
                    acceptable_np,
                    safe_np,
                    progress_np,
                    signs_np,
                    pair_weights_np,
                    sample_weights_np,
                    ref_logits_np,
                    protected_np,
                    mask_np,
                ) = packed
                if not mask_np.any():
                    continue
                images = torch.as_tensor(images_np, dtype=torch.float32, device=device)
                acceptable = torch.as_tensor(acceptable_np, dtype=torch.bool, device=device)
                safe = torch.as_tensor(safe_np, dtype=torch.bool, device=device)
                progress = torch.as_tensor(progress_np, dtype=torch.bool, device=device)
                signs = torch.as_tensor(signs_np, dtype=torch.float32, device=device)
                pair_weights = torch.as_tensor(pair_weights_np, dtype=torch.float32, device=device)
                sample_weights = torch.as_tensor(sample_weights_np, dtype=torch.float32, device=device)
                ref_logits = torch.as_tensor(ref_logits_np, dtype=torch.float32, device=device)
                protected = torch.as_tensor(protected_np, dtype=torch.bool, device=device)
                mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)

                optimizer.zero_grad(set_to_none=True)
                motors, next_state = policy.run_window_with_motor(images, state)
                logits = decoder(motors)
                valid_logits = logits[mask]
                pref, _ = preference_loss(
                    valid_logits,
                    acceptable[mask],
                    safe[mask],
                    signs[mask],
                    pair_weights[mask],
                    sample_weights[mask],
                )
                prog = progress_preference_loss(
                    valid_logits,
                    acceptable[mask],
                    progress[mask],
                    sample_weights[mask],
                )
                if bool(protected.any()):
                    distill = F.mse_loss(logits[protected], ref_logits[protected])
                else:
                    distill = logits.sum() * 0.0
                trust = _trust_penalty(
                    policy,
                    snap,
                    edge_mask=edge_mask,
                    leak_mask=leak_mask,
                )
                loss = pref + progress_weight * prog + distill_weight * distill + trust_weight * trust
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite V5 sensory/core adaptation loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                state = next_state.detach()

                epoch_losses.append(float(loss.detach()))
                epoch_pref.append(float(pref.detach()))
                epoch_distill.append(float(distill.detach()))
                epoch_trust.append(float(trust.detach()))

        row = {
            "epoch": int(epoch),
            "loss": float(np.mean(epoch_losses)),
            "preference": float(np.mean(epoch_pref)),
            "distill": float(np.mean(epoch_distill)),
            "trust": float(np.mean(epoch_trust)),
        }
        history.append(row)
        print(
            f"    epoch {epoch}/{epochs} loss={row['loss']:.4f} "
            f"pref={row['preference']:.4f} distill={row['distill']:.4f}",
            flush=True,
        )

    for h in hooks:
        h.remove()
    policy.eval()
    return {
        "phase": phase,
        "epochs": int(epochs),
        "adapterLearningRate": float(adapter_lr),
        "coreLearningRate": float(core_lr),
        "distillWeight": float(distill_weight),
        "progressWeight": float(progress_weight),
        "trustWeight": float(trust_weight),
        "trainableParameters": int(sum(p.numel() for p in policy.parameters() if p.requires_grad)),
        "history": history,
        "samples": int(sum(ep.length for ep in episodes)),
        "sources": dict(sorted(Counter(ep.source for ep in episodes).items())),
    }


def _exact_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["success"]),
        float(row["score"]),
        float(row["steps"]),
    )


def save_checkpoint(
    path: Path,
    *,
    policy: SensoryPlasticPolicy,
    decoder: RecoveryDecoderV2,
    source_checkpoint: Path,
    source_decoder: Path,
    stage: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": VERSION,
            "policy": policy.state_dict(),
            "decoder": decoder.state_dict(),
            "decoderArchitecture": decoder.architecture,
            "motorDim": decoder.motor_dim,
            "rank": policy.rank,
            "residualScale": policy.residual_scale,
            "sourceCheckpoint": str(source_checkpoint.resolve()),
            "sourceDecoder": str(source_decoder.resolve()),
            "stage": stage,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def load_adaptation_checkpoint(
    path: Path,
    *,
    policy: SensoryPlasticPolicy,
    decoder: RecoveryDecoderV2,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["policy"], strict=True)
    decoder.load_state_dict(payload["decoder"], strict=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crossy V5 sensory/core adaptation. Decoder-side work is closed: keep the "
            "selected MLP64 decoder frozen, learn the artificial RGB->ol_sensory "
            "interface, then (if needed) plasticize only the measured core boundary "
            "immediately downstream of ol_sensory."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=EXPO_SEED)
    parser.add_argument("--rng-seed", type=int, default=1001)
    parser.add_argument("--planner-depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--sensory-rounds", type=int, default=3)
    parser.add_argument("--boundary-rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--adapter-lr", type=float, default=0.001)
    parser.add_argument("--boundary-adapter-lr", type=float, default=0.0005)
    parser.add_argument("--core-lr", type=float, default=0.0002)
    parser.add_argument("--distill-weight", type=float, default=4.0)
    parser.add_argument("--progress-weight", type=float, default=0.75)
    parser.add_argument("--trust-weight", type=float, default=0.05)
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
        default=repo_root() / "runs" / "crossy-v4-exact-seed-branch-curriculum" / "best_decoder.pt",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--flyhard-root", type=Path, default=default_flyhard_root())
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root() / "runs" / "crossy-v5-sensory-core-adaptation",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_steps = min(args.max_steps, 40)
        args.sensory_rounds = 1
        args.boundary_rounds = 1
        args.epochs = 1
        args.batch = 1
        args.window = min(args.window, 8)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    args.out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    started = time.perf_counter()

    base, metadata, graph_col, sensory_ids = _load_base_policy(
        checkpoint_path=args.checkpoint,
        flyhard_root=args.flyhard_root,
        device=device,
    )
    policy = SensoryPlasticPolicy(
        base,
        rank=args.rank,
        residual_scale=args.residual_scale,
    ).to(device)
    decoder, decoder_metadata = load_decoder_checkpoint(args.decoder_checkpoint, device=device)
    if decoder.architecture != "mlp64":
        raise SystemExit("V5 expects the selected mlp64 decoder.")
    for p in decoder.parameters():
        p.requires_grad_(False)

    edge_mask, leak_mask, plasticity_report = build_plasticity_masks(
        base=base,
        graph_col=graph_col,
        sensory_ids=sensory_ids,
        device=device,
    )

    teacher_state = variant_state(args.seed, phase_offset=0.0, initial_column=0)
    teacher_score = 0.0
    teacher_no_progress = 0
    teacher_max_no_progress = 0
    for _ in range(args.max_steps):
        if teacher_state.terminal is not None:
            break
        values, best, _ = planner_action_preferences(teacher_state, depth=args.planner_depth)
        before = float(teacher_state.score)
        teacher_state = step_game(teacher_state, ACTION_ORDER[int(best)]).state
        if float(teacher_state.score) > before:
            teacher_no_progress = 0
        else:
            teacher_no_progress += 1
            teacher_max_no_progress = max(teacher_max_no_progress, teacher_no_progress)
    teacher_score = float(teacher_state.score)
    target_score = teacher_score * float(args.min_teacher_fraction)
    effective_stall = max(int(args.stall_steps), int(teacher_max_no_progress) + 2)

    config = {
        "seed": args.seed,
        "rngSeed": args.rng_seed,
        "plannerDepth": args.planner_depth,
        "maxSteps": args.max_steps,
        "rank": args.rank,
        "residualScale": args.residual_scale,
        "sensoryRounds": args.sensory_rounds,
        "boundaryRounds": args.boundary_rounds,
        "epochsPerRound": args.epochs,
        "batch": args.batch,
        "window": args.window,
        "adapterLearningRate": args.adapter_lr,
        "boundaryAdapterLearningRate": args.boundary_adapter_lr,
        "coreLearningRate": args.core_lr,
        "distillWeight": args.distill_weight,
        "progressWeight": args.progress_weight,
        "trustWeight": args.trust_weight,
        "teacherScore": teacher_score,
        "minimumTeacherFraction": args.min_teacher_fraction,
        "minimumProgressScore": target_score,
        "effectiveStallSteps": effective_stall,
        "decoderRetrained": False,
        "sensoryInterfaceChanged": True,
        "coreBoundaryPlasticity": "outgoing ol_sensory edges + sensory/first-hop leaks",
        "hardStop": "after sensory-only and sensory-core-boundary phases, no more decoder audits",
    }

    print("=== CROSSY V5 / SENSORY + CORE ADAPTATION ===", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(
        f"MaleCNS: {metadata['neurons']:,} neurons / {metadata['edges']:,} measured edges",
        flush=True,
    )
    print(
        f"adapter: rank={args.rank}, residualScale={args.residual_scale}; decoder frozen ({decoder.architecture})",
        flush=True,
    )
    print(
        f"boundary plasticity: {plasticity_report['sensoryOutgoingEdges']:,} sensory-outgoing edges / "
        f"{plasticity_report['sensoryAndFirstHopLeaks']:,} leaks",
        flush=True,
    )
    print(f"teacher score={teacher_score:.1f}; target={target_score:.1f}", flush=True)

    if args.resume is not None:
        payload = load_adaptation_checkpoint(args.resume, policy=policy, decoder=decoder)
        print(f"resume: {args.resume} stage={payload.get('stage')}", flush=True)

    baseline = evaluate_suite(
        policy=policy,
        decoder=decoder,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        target_score=target_score,
        stall_steps=effective_stall,
        smoke=args.smoke,
    )
    best_suite = baseline
    best_stage = "v5-baseline"
    save_checkpoint(
        args.out / "best.pt",
        policy=policy,
        decoder=decoder,
        source_checkpoint=args.checkpoint,
        source_decoder=args.decoder_checkpoint,
        stage=best_stage,
        metrics=best_suite,
        config=config,
    )
    print(
        f"baseline: score={baseline['exactExpo']['score']:.1f} "
        f"steps={baseline['exactExpo']['steps']}/{args.max_steps}",
        flush=True,
    )

    phases = [
        ("sensory-only", args.sensory_rounds, args.adapter_lr, 0.0),
        ("sensory-core-boundary", args.boundary_rounds, args.boundary_adapter_lr, args.core_lr),
    ]
    rounds: list[dict[str, Any]] = []
    global_round = 0
    stop_reason = "completed-plasticity-budget"

    for phase, phase_rounds, adapter_lr, core_lr in phases:
        for phase_round in range(1, phase_rounds + 1):
            global_round += 1
            load_adaptation_checkpoint(args.out / "best.pt", policy=policy, decoder=decoder)
            episodes, collection = collect_round_dataset(
                policy=policy,
                decoder=decoder,
                device=device,
                seed=args.seed,
                max_steps=args.max_steps,
                planner_depth=args.planner_depth,
                smoke=args.smoke,
            )
            print(
                f"\n[{phase} {phase_round}/{phase_rounds}] samples={sum(ep.length for ep in episodes):,}",
                flush=True,
            )
            training = train_round(
                policy=policy,
                decoder=decoder,
                episodes=episodes,
                device=device,
                phase=phase,
                epochs=args.epochs,
                batch_size=args.batch,
                window=args.window,
                adapter_lr=adapter_lr,
                core_lr=core_lr,
                distill_weight=args.distill_weight,
                progress_weight=args.progress_weight,
                trust_weight=args.trust_weight,
                edge_mask=edge_mask,
                leak_mask=leak_mask,
                seed=args.rng_seed + global_round * 101,
            )
            exact = evaluate_variant(
                policy=policy,
                decoder=decoder,
                device=device,
                seed=args.seed,
                phase_offset=0.0,
                initial_column=0,
                max_steps=args.max_steps,
                target_score=target_score,
                stall_steps=effective_stall,
            )
            best_exact = best_suite["exactExpo"]
            exact_improved = _exact_key(exact) > _exact_key(best_exact)
            suite = None
            if exact_improved:
                suite = evaluate_suite(
                    policy=policy,
                    decoder=decoder,
                    device=device,
                    seed=args.seed,
                    max_steps=args.max_steps,
                    target_score=target_score,
                    stall_steps=effective_stall,
                    smoke=args.smoke,
                )
                best_suite = suite
                best_stage = f"{phase}-round-{phase_round}"
                save_checkpoint(
                    args.out / "best.pt",
                    policy=policy,
                    decoder=decoder,
                    source_checkpoint=args.checkpoint,
                    source_decoder=args.decoder_checkpoint,
                    stage=best_stage,
                    metrics=best_suite,
                    config=config,
                )
            else:
                save_checkpoint(
                    args.out / "latest.pt",
                    policy=policy,
                    decoder=decoder,
                    source_checkpoint=args.checkpoint,
                    source_decoder=args.decoder_checkpoint,
                    stage=f"{phase}-round-{phase_round}-rejected",
                    metrics={"exactExpo": exact},
                    config=config,
                )

            rounds.append(
                {
                    "globalRound": global_round,
                    "phase": phase,
                    "phaseRound": phase_round,
                    "collection": collection,
                    "training": training,
                    "exactExpo": exact,
                    "accepted": bool(exact_improved),
                    "acceptedClosedLoop": suite,
                }
            )
            print(
                f"  exact score={exact['score']:.1f} steps={exact['steps']}/{args.max_steps} "
                f"{'NEW BEST' if exact_improved else 'reject'}",
                flush=True,
            )
            if exact_improved and bool(exact["success"]):
                stop_reason = "target-achieved"
                break
        if stop_reason == "target-achieved":
            break

    load_adaptation_checkpoint(args.out / "best.pt", policy=policy, decoder=decoder)
    final = evaluate_suite(
        policy=policy,
        decoder=decoder,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        target_score=target_score,
        stall_steps=effective_stall,
        smoke=args.smoke,
    )
    final_exact = final["exactExpo"]
    if bool(final_exact["success"]):
        next_action = "validate-and-integrate-v5-for-expo"
    elif _exact_key(final_exact) > _exact_key(baseline["exactExpo"]):
        next_action = "continue-v5-training-from-best-checkpoint"
    else:
        next_action = "redesign-sensory-topology-and-retrain-core"

    report = {
        "version": VERSION,
        "sourceCheckpoint": str(args.checkpoint.resolve()),
        "sourceDecoder": str(args.decoder_checkpoint.resolve()),
        "metadata": metadata,
        "decoderMetadata": decoder_metadata,
        "config": config,
        "plasticity": plasticity_report,
        "teacherReference": {
            "score": teacher_score,
            "targetScore": target_score,
            "maxNoProgressStreak": teacher_max_no_progress,
        },
        "baseline": baseline,
        "rounds": rounds,
        "selectedStage": best_stage,
        "selectedClosedLoop": final,
        "bestCheckpoint": str((args.out / "best.pt").resolve()),
        "stopReason": stop_reason,
        "nextAction": next_action,
        "elapsedSeconds": time.perf_counter() - started,
        "interpretation": (
            "Decoder parameters stayed frozen. V5 directly trained the artificial camera-to-ol_sensory "
            "interface and, in phase 2, only measured connectome parameters at the immediate sensory/core "
            "boundary. This is a model intervention, not another decoder audit."
        ),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== V5 FINAL ===", flush=True)
    print(
        f"stage={best_stage} score={final_exact['score']:.1f} "
        f"steps={final_exact['steps']}/{args.max_steps} success={int(final_exact['success'])}",
        flush=True,
    )
    print(f"nextAction={next_action}", flush=True)
    print(f"report: {(args.out / 'report.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
