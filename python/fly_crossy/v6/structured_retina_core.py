from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fly_crossy.v2.preference_distill import preference_loss
from fly_crossy.v4.exact_seed_branch_curriculum import load_decoder_checkpoint
from fly_crossy.v4.policy import CHANNELS, IMAGE_H, IMAGE_W, FullMaleCNSRGBPolicy
from fly_crossy.v4.recovery_dagger_v2 import (
    RecoveryDecoderV2,
    evaluate_suite,
    evaluate_variant,
    progress_preference_loss,
    repo_root,
)
from fly_crossy.v4.train_expo_specialist import EXPO_SEED, default_flyhard_root
from fly_crossy.v5.sensory_core_adaptation import (
    TrainEpisode,
    _load_base_policy,
    _padded_chunk,
    collect_round_dataset,
    episode_weights,
)

VERSION = "crossy-v6-structured-retina-core-1"
RETINA_CHANNELS = 8


def _structured_slots(sensory_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evenly assign sensory neurons to local retina channels and image positions.

    Slots are flattened position-major: each pixel has RETINA_CHANNELS local features.
    Even spacing means every image region gets approximately the same sensory density.
    """
    total = IMAGE_H * IMAGE_W * RETINA_CHANNELS
    slots = np.floor(np.arange(sensory_count, dtype=np.float64) * total / sensory_count).astype(np.int64)
    slots = np.clip(slots, 0, total - 1)
    position = slots // RETINA_CHANNELS
    channel = slots % RETINA_CHANNELS
    y = position // IMAGE_W
    x = position % IMAGE_W
    return channel, y, x


def _initial_retina_kernel() -> torch.Tensor:
    """Local, interpretable RGB filters used only as an initialization.

    Channels:
      0 luminance
      1 red-green opponent
      2 blue-yellow opponent
      3 local luminance contrast
      4 horizontal luminance edge
      5 vertical luminance edge
      6 red-blue opponent
      7 green-blue opponent
    """
    w = torch.zeros(RETINA_CHANNELS, CHANNELS, 3, 3, dtype=torch.float32)

    # Center-only color channels.
    w[0, :, 1, 1] = 1.0 / 3.0
    w[1, 0, 1, 1] = 1.0
    w[1, 1, 1, 1] = -1.0
    w[2, 2, 1, 1] = 1.0
    w[2, 0, 1, 1] = -0.5
    w[2, 1, 1, 1] = -0.5
    w[6, 0, 1, 1] = 1.0
    w[6, 2, 1, 1] = -1.0
    w[7, 1, 1, 1] = 1.0
    w[7, 2, 1, 1] = -1.0

    # Local contrast: center luminance minus 8-neighbor luminance mean.
    for c in range(CHANNELS):
        w[3, c, 1, 1] = 1.0 / 3.0
        for yy in range(3):
            for xx in range(3):
                if yy == 1 and xx == 1:
                    continue
                w[3, c, yy, xx] = -1.0 / 24.0

    # Local spatial derivatives on luminance.
    for c in range(CHANNELS):
        w[4, c, 1, 0] = -1.0 / 6.0
        w[4, c, 1, 2] = 1.0 / 6.0
        w[5, c, 0, 1] = -1.0 / 6.0
        w[5, c, 2, 1] = 1.0 / 6.0
    return w


class StructuredRetinaPolicy(nn.Module):
    """48x24 RGB -> local structured retina -> measured MaleCNS.

    Unlike V4/V5, there is no random pixel-to-sensory assignment and no global
    low-rank projection. Each ol_sensory unit samples one feature from a fixed
    local retinal position. The only visual mixing is a shared 3x3 convolution.
    """

    def __init__(self, base: FullMaleCNSRGBPolicy) -> None:
        super().__init__()
        self.base = base
        sensory_count = int(len(base.sensory_ids))
        ch, yy, xx = _structured_slots(sensory_count)
        self.register_buffer("retina_channel", torch.as_tensor(ch, dtype=torch.long))
        self.register_buffer("retina_y", torch.as_tensor(yy, dtype=torch.long))
        self.register_buffer("retina_x", torch.as_tensor(xx, dtype=torch.long))

        self.retina_kernel = nn.Parameter(_initial_retina_kernel())
        self.retina_bias = nn.Parameter(torch.zeros(RETINA_CHANNELS, dtype=torch.float32))
        self.sensory_gain = nn.Parameter(torch.zeros(sensory_count, dtype=torch.float32))
        self.sensory_bias = nn.Parameter(torch.zeros(sensory_count, dtype=torch.float32))

    def zero_state(self, batch: int, *, device=None) -> torch.Tensor:
        return self.base.zero_state(batch, device=device)

    def retinal_maps(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (IMAGE_H, IMAGE_W, CHANNELS):
            raise ValueError(f"Expected RGB images [B,{IMAGE_H},{IMAGE_W},{CHANNELS}].")
        x = ((images - 0.5) * 2.0).clamp(-2.0, 2.0).permute(0, 3, 1, 2)
        return torch.tanh(F.conv2d(x, self.retina_kernel, self.retina_bias, padding=1))

    def drive_for(self, images: torch.Tensor) -> torch.Tensor:
        maps = self.retinal_maps(images)
        sampled = maps[:, self.retina_channel, self.retina_y, self.retina_x]
        gain = 1.0 + 0.25 * torch.tanh(self.sensory_gain)[None, :]
        bias = 0.10 * torch.tanh(self.sensory_bias)[None, :]
        sensory_drive = sampled * gain + bias

        drive = torch.zeros(
            self.base.core.n,
            len(images),
            dtype=torch.float32,
            device=images.device,
        )
        drive[self.base.sensory_ids] = sensory_drive.T
        return drive

    def step_state(self, state: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        return self.base.core(state, steps=4, drive=self.drive_for(images))

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


def build_multihop_masks(
    *,
    base: FullMaleCNSRGBPolicy,
    graph_col: np.ndarray,
    sensory_ids: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    n = int(base.core.n)
    rows = base.core.rows.detach().cpu().numpy()

    sensory_node_mask = np.zeros(n, dtype=np.bool_)
    sensory_node_mask[sensory_ids] = True
    edge_hop1_np = sensory_node_mask[graph_col]
    hop1 = np.unique(rows[edge_hop1_np])

    source_hop2 = np.unique(np.concatenate([sensory_ids, hop1]))
    hop2_source_mask = np.zeros(n, dtype=np.bool_)
    hop2_source_mask[source_hop2] = True
    edge_hop2_np = hop2_source_mask[graph_col]
    hop2 = np.unique(rows[edge_hop2_np])

    leak_hop1_np = np.zeros(n, dtype=np.bool_)
    leak_hop1_np[np.unique(np.concatenate([sensory_ids, hop1]))] = True
    leak_hop2_np = np.zeros(n, dtype=np.bool_)
    leak_hop2_np[np.unique(np.concatenate([sensory_ids, hop1, hop2]))] = True

    return {
        "edgeHop1": torch.as_tensor(edge_hop1_np, dtype=torch.bool, device=device),
        "edgeHop2": torch.as_tensor(edge_hop2_np, dtype=torch.bool, device=device),
        "leakHop1": torch.as_tensor(leak_hop1_np, dtype=torch.bool, device=device),
        "leakHop2": torch.as_tensor(leak_hop2_np, dtype=torch.bool, device=device),
        "report": {
            "sensoryOutgoingEdges": int(edge_hop1_np.sum()),
            "sensoryPlusHop1OutgoingEdges": int(edge_hop2_np.sum()),
            "firstHopNeurons": int(len(hop1)),
            "secondHopTargetNeurons": int(len(hop2)),
            "hop1LeakNeurons": int(leak_hop1_np.sum()),
            "hop2LeakNeurons": int(leak_hop2_np.sum()),
        },
    }


def _retina_parameters(policy: StructuredRetinaPolicy) -> list[nn.Parameter]:
    return [
        policy.retina_kernel,
        policy.retina_bias,
        policy.sensory_gain,
        policy.sensory_bias,
    ]


def configure_phase(
    policy: StructuredRetinaPolicy,
    *,
    phase: str,
    masks: dict[str, Any],
) -> tuple[list[Any], torch.Tensor | None, torch.Tensor | None]:
    for p in policy.parameters():
        p.requires_grad_(False)
    for p in _retina_parameters(policy):
        p.requires_grad_(True)

    hooks: list[Any] = []
    edge_mask = None
    leak_mask = None
    if phase == "retina-only":
        pass
    elif phase == "retina-hop1":
        edge_mask = masks["edgeHop1"]
        leak_mask = masks["leakHop1"]
    elif phase == "retina-hop2":
        edge_mask = masks["edgeHop2"]
        leak_mask = masks["leakHop2"]
    else:
        raise ValueError(f"Unknown phase: {phase}")

    if edge_mask is not None:
        policy.base.core.edge_gain.requires_grad_(True)
        policy.base.core.leak.requires_grad_(True)
        hooks.append(policy.base.core.edge_gain.register_hook(lambda g: g * edge_mask))
        hooks.append(policy.base.core.leak.register_hook(lambda g: g * leak_mask))
    return hooks, edge_mask, leak_mask


def _snapshot(
    policy: StructuredRetinaPolicy,
    *,
    edge_mask: torch.Tensor | None,
    leak_mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    snap = {
        "retina_kernel": policy.retina_kernel.detach().clone(),
        "retina_bias": policy.retina_bias.detach().clone(),
        "sensory_gain": policy.sensory_gain.detach().clone(),
        "sensory_bias": policy.sensory_bias.detach().clone(),
    }
    if edge_mask is not None:
        snap["edge_gain"] = policy.base.core.edge_gain.detach()[edge_mask].clone()
        snap["leak"] = policy.base.core.leak.detach()[leak_mask].clone()
    return snap


def _trust_penalty(
    policy: StructuredRetinaPolicy,
    snap: dict[str, torch.Tensor],
    *,
    edge_mask: torch.Tensor | None,
    leak_mask: torch.Tensor | None,
) -> torch.Tensor:
    terms = [
        F.mse_loss(policy.retina_kernel, snap["retina_kernel"]),
        F.mse_loss(policy.retina_bias, snap["retina_bias"]),
        F.mse_loss(policy.sensory_gain, snap["sensory_gain"]),
        F.mse_loss(policy.sensory_bias, snap["sensory_bias"]),
    ]
    if edge_mask is not None:
        terms.append(F.mse_loss(policy.base.core.edge_gain[edge_mask], snap["edge_gain"]))
        terms.append(F.mse_loss(policy.base.core.leak[leak_mask], snap["leak"]))
    return torch.stack(terms).mean()


def train_round(
    *,
    policy: StructuredRetinaPolicy,
    decoder: RecoveryDecoderV2,
    episodes: list[TrainEpisode],
    device: torch.device,
    phase: str,
    epochs: int,
    batch_size: int,
    window: int,
    retina_lr: float,
    core_lr: float,
    distill_weight: float,
    progress_weight: float,
    trust_weight: float,
    masks: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    hooks, edge_mask, leak_mask = configure_phase(policy, phase=phase, masks=masks)
    for p in decoder.parameters():
        p.requires_grad_(False)
    decoder.eval()

    groups: list[dict[str, Any]] = [{"params": _retina_parameters(policy), "lr": retina_lr}]
    if edge_mask is not None:
        groups.append({"params": [policy.base.core.edge_gain, policy.base.core.leak], "lr": core_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    snap = _snapshot(policy, edge_mask=edge_mask, leak_mask=leak_mask)
    weights = [episode_weights(ep) for ep in episodes]
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []

    policy.train()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(episodes))
        losses: list[float] = []
        prefs: list[float] = []
        distills: list[float] = []
        trusts: list[float] = []

        for group_start in range(0, len(order), batch_size):
            ids = order[group_start: group_start + batch_size].tolist()
            group = [episodes[i] for i in ids]
            group_weights = [weights[i] for i in ids]
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
                    reference_logits_np,
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
                ref_logits = torch.as_tensor(reference_logits_np, dtype=torch.float32, device=device)
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
                    raise RuntimeError("Non-finite V6 training loss.")
                loss.backward()
                trainable = [p for p in policy.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                state = next_state.detach()

                losses.append(float(loss.detach()))
                prefs.append(float(pref.detach()))
                distills.append(float(distill.detach()))
                trusts.append(float(trust.detach()))

        row = {
            "epoch": int(epoch),
            "loss": float(np.mean(losses)),
            "preference": float(np.mean(prefs)),
            "distill": float(np.mean(distills)),
            "trust": float(np.mean(trusts)),
        }
        history.append(row)
        print(
            f"    epoch {epoch}/{epochs} loss={row['loss']:.4f} "
            f"pref={row['preference']:.4f} distill={row['distill']:.4f}",
            flush=True,
        )

    for hook in hooks:
        hook.remove()
    policy.eval()

    effective_edges = 0 if edge_mask is None else int(edge_mask.sum().item())
    effective_leaks = 0 if leak_mask is None else int(leak_mask.sum().item())
    return {
        "phase": phase,
        "epochs": int(epochs),
        "retinaLearningRate": float(retina_lr),
        "coreLearningRate": float(core_lr),
        "distillWeight": float(distill_weight),
        "progressWeight": float(progress_weight),
        "trustWeight": float(trust_weight),
        "retinaParameters": int(sum(p.numel() for p in _retina_parameters(policy))),
        "effectivePlasticEdges": effective_edges,
        "effectivePlasticLeaks": effective_leaks,
        "samples": int(sum(ep.length for ep in episodes)),
        "sources": dict(sorted(Counter(ep.source for ep in episodes).items())),
        "history": history,
    }


def _exact_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (float(row["success"]), float(row["score"]), float(row["steps"]))


def save_checkpoint(
    path: Path,
    *,
    policy: StructuredRetinaPolicy,
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
            "sourceCheckpoint": str(source_checkpoint.resolve()),
            "sourceDecoder": str(source_decoder.resolve()),
            "stage": stage,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    policy: StructuredRetinaPolicy,
    decoder: RecoveryDecoderV2,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["policy"], strict=True)
    decoder.load_state_dict(payload["decoder"], strict=True)
    return payload


def _reference_v5_score(path: Path) -> tuple[float, dict[str, Any] | None]:
    if not path.is_file():
        return 41.0, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("exactExpo"), dict):
        return float(metrics["exactExpo"].get("score", 41.0)), metrics
    return 41.0, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crossy V6: structured local retina into the measured MaleCNS, followed by "
            "progressive hop-1 and hop-2 connectome plasticity. The MLP64 motor decoder stays frozen."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=EXPO_SEED)
    parser.add_argument("--rng-seed", type=int, default=1201)
    parser.add_argument("--planner-depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--retina-rounds", type=int, default=4)
    parser.add_argument("--hop1-rounds", type=int, default=3)
    parser.add_argument("--hop2-rounds", type=int, default=3)
    parser.add_argument("--retina-epochs", type=int, default=3)
    parser.add_argument("--core-epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--retina-lr", type=float, default=0.001)
    parser.add_argument("--hop1-retina-lr", type=float, default=0.0005)
    parser.add_argument("--hop2-retina-lr", type=float, default=0.0003)
    parser.add_argument("--hop1-core-lr", type=float, default=0.00015)
    parser.add_argument("--hop2-core-lr", type=float, default=0.00008)
    parser.add_argument("--distill-weight", type=float, default=4.0)
    parser.add_argument("--progress-weight", type=float, default=0.75)
    parser.add_argument("--trust-weight", type=float, default=0.05)
    parser.add_argument("--stall-steps", type=int, default=12)
    parser.add_argument("--min-teacher-fraction", type=float, default=0.80)
    parser.add_argument("--architecture-min-score", type=float, default=60.0)
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
    parser.add_argument(
        "--reference-v5",
        type=Path,
        default=repo_root() / "runs" / "crossy-v5-sensory-core-adaptation" / "best.pt",
    )
    parser.add_argument("--flyhard-root", type=Path, default=default_flyhard_root())
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root() / "runs" / "crossy-v6-structured-retina-core",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_steps = min(args.max_steps, 40)
        args.retina_rounds = 1
        args.hop1_rounds = 1
        args.hop2_rounds = 1
        args.retina_epochs = 1
        args.core_epochs = 1
        args.batch = 1
        args.window = min(args.window, 8)
        args.out = repo_root() / "runs" / "crossy-v6-smoke"

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
    policy = StructuredRetinaPolicy(base).to(device)
    decoder, decoder_metadata = load_decoder_checkpoint(args.decoder_checkpoint, device=device)
    if decoder.architecture != "mlp64":
        raise SystemExit("V6 expects the selected mlp64 decoder.")
    for p in decoder.parameters():
        p.requires_grad_(False)

    masks = build_multihop_masks(
        base=base,
        graph_col=graph_col,
        sensory_ids=sensory_ids,
        device=device,
    )

    # Calibrate target from the same teacher convention used by V5.
    from fly_crossy.env import step_game
    from fly_crossy.schema import ACTION_ORDER
    from fly_crossy.v2.preference_distill import planner_action_preferences
    from fly_crossy.v4.train_expo_specialist import variant_state

    teacher_state = variant_state(args.seed, phase_offset=0.0, initial_column=0)
    teacher_no_progress = 0
    teacher_max_no_progress = 0
    for _ in range(args.max_steps):
        if teacher_state.terminal is not None:
            break
        _, best, _ = planner_action_preferences(teacher_state, depth=args.planner_depth)
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
    v5_reference_score, v5_reference_metrics = _reference_v5_score(args.reference_v5)

    channel_counts = np.bincount(
        policy.retina_channel.detach().cpu().numpy(), minlength=RETINA_CHANNELS
    ).tolist()
    config = {
        "version": VERSION,
        "seed": args.seed,
        "rngSeed": args.rng_seed,
        "maxSteps": args.max_steps,
        "plannerDepth": args.planner_depth,
        "retina": {
            "type": "shared-local-3x3",
            "channels": RETINA_CHANNELS,
            "routing": "deterministic-position-major-even-spacing",
            "globalPixelMixing": False,
            "randomPixelRouting": False,
            "channelCounts": channel_counts,
        },
        "training": {
            "retinaRounds": args.retina_rounds,
            "hop1Rounds": args.hop1_rounds,
            "hop2Rounds": args.hop2_rounds,
            "retinaEpochs": args.retina_epochs,
            "coreEpochs": args.core_epochs,
            "batch": args.batch,
            "window": args.window,
            "retinaLearningRate": args.retina_lr,
            "hop1RetinaLearningRate": args.hop1_retina_lr,
            "hop2RetinaLearningRate": args.hop2_retina_lr,
            "hop1CoreLearningRate": args.hop1_core_lr,
            "hop2CoreLearningRate": args.hop2_core_lr,
            "distillWeight": args.distill_weight,
            "progressWeight": args.progress_weight,
            "trustWeight": args.trust_weight,
        },
        "teacherScore": teacher_score,
        "minimumProgressScore": target_score,
        "architectureMinimumScore": args.architecture_min_score,
        "decoderRetrained": False,
    }

    print("=== CROSSY V6 / STRUCTURED RETINA + CORE CURRICULUM ===", flush=True)
    print(
        f"MaleCNS: {metadata['neurons']:,} neurons / {metadata['edges']:,} measured edges",
        flush=True,
    )
    print(
        f"retina: {RETINA_CHANNELS} local 3x3 feature maps -> {metadata['sensory']:,} ol_sensory; random routing: NO",
        flush=True,
    )
    print(f"decoder: {decoder.architecture} FROZEN", flush=True)
    print(
        f"hop1 edges={masks['report']['sensoryOutgoingEdges']:,}; "
        f"hop2 edge budget={masks['report']['sensoryPlusHop1OutgoingEdges']:,}",
        flush=True,
    )
    print(
        f"teacher={teacher_score:.1f}; real target={target_score:.1f}; V5 reference={v5_reference_score:.1f}; "
        f"V6 architecture bar={args.architecture_min_score:.1f}",
        flush=True,
    )

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
    best_stage = "v6-structured-baseline"
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
        ("retina-only", args.retina_rounds, args.retina_epochs, args.retina_lr, 0.0),
        ("retina-hop1", args.hop1_rounds, args.core_epochs, args.hop1_retina_lr, args.hop1_core_lr),
        ("retina-hop2", args.hop2_rounds, args.core_epochs, args.hop2_retina_lr, args.hop2_core_lr),
    ]
    rounds: list[dict[str, Any]] = []
    global_round = 0
    stop_reason = "completed-v6-curriculum"

    for phase, phase_rounds, epochs, retina_lr, core_lr in phases:
        for phase_round in range(1, phase_rounds + 1):
            global_round += 1
            load_checkpoint(args.out / "best.pt", policy=policy, decoder=decoder)
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
                epochs=epochs,
                batch_size=args.batch,
                window=args.window,
                retina_lr=retina_lr,
                core_lr=core_lr,
                distill_weight=args.distill_weight,
                progress_weight=args.progress_weight,
                trust_weight=args.trust_weight,
                masks=masks,
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
            improved = _exact_key(exact) > _exact_key(best_suite["exactExpo"])
            suite = None
            if improved:
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
                    stage=f"{phase}-round-{phase_round}",
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
                    "accepted": bool(improved),
                    "acceptedClosedLoop": suite,
                }
            )
            print(
                f"  exact score={exact['score']:.1f} steps={exact['steps']}/{args.max_steps} "
                f"{'NEW BEST' if improved else 'reject'}",
                flush=True,
            )
            if bool(exact["success"]):
                stop_reason = "real-success"
                break
        if stop_reason == "real-success":
            break

    load_checkpoint(args.out / "best.pt", policy=policy, decoder=decoder)
    selected = evaluate_suite(
        policy=policy,
        decoder=decoder,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        target_score=target_score,
        stall_steps=effective_stall,
        smoke=args.smoke,
    )
    best_score = float(selected["exactExpo"]["score"])
    if bool(selected["exactExpo"]["success"]):
        next_action = "validate-and-integrate-v6-for-expo"
    elif best_score >= float(args.architecture_min_score):
        next_action = "continue-v6-structured-retina-training"
    else:
        next_action = "reject-v6-structured-retina-topology-and-redesign"

    report = {
        "version": VERSION,
        "sourceCheckpoint": str(args.checkpoint.resolve()),
        "sourceDecoder": str(args.decoder_checkpoint.resolve()),
        "metadata": metadata,
        "decoderMetadata": decoder_metadata,
        "config": config,
        "plasticity": masks["report"],
        "teacherReference": {
            "score": teacher_score,
            "targetScore": target_score,
            "maxNoProgressStreak": teacher_max_no_progress,
        },
        "v5Reference": {
            "checkpoint": str(args.reference_v5.resolve()),
            "score": v5_reference_score,
            "metrics": v5_reference_metrics,
        },
        "baseline": baseline,
        "rounds": rounds,
        "selectedStage": best_stage,
        "selectedClosedLoop": selected,
        "bestCheckpoint": str((args.out / "best.pt").resolve()),
        "stopReason": stop_reason,
        "nextAction": next_action,
        "elapsedSeconds": time.perf_counter() - started,
        "interpretation": (
            "V6 replaced random/global camera routing with a deterministic local structured retina. "
            "The MLP64 motor decoder stayed frozen. Core plasticity was progressively expanded "
            "from retina-only to measured sensory outgoing edges and then one additional hop."
        ),
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== V6 COMPLETE ===", flush=True)
    print(f"selected stage: {best_stage}", flush=True)
    print(
        f"exact: score={selected['exactExpo']['score']:.1f} "
        f"steps={selected['exactExpo']['steps']}/{args.max_steps} "
        f"success={selected['exactExpo']['success']}",
        flush=True,
    )
    print(f"V5 reference score: {v5_reference_score:.1f}", flush=True)
    print(f"architecture bar: {args.architecture_min_score:.1f}", flush=True)
    print(f"nextAction: {next_action}", flush=True)
    print(f"checkpoint: {(args.out / 'best.pt').resolve()}", flush=True)
    print(f"report: {report_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
