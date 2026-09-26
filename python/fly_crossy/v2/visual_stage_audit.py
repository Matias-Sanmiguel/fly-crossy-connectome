from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from fly_crossy.env import create_game, observe, step_game
from fly_crossy.schema import ACTION_ORDER, flatten_observation

from .core import FrozenMaleCNSCore
from .crossy_camera import render_crossy_neural_frame
from .dagger_probe import make_episode_folds, sample_weights
from .decoder import decoder_logits, load_decoder
from .information_probe import metrics_from_predictions
from .substrate import MaleCNSV2Substrate
from .vision_frontend import FrozenVisualFrontEnd
from .vision_geometry import VisualGeometry


ACTION_NAMES = tuple(action.value for action in ACTION_ORDER)
AUDIT_VERSION = "malecns-crossy-v2-visual-stage-audit-1"

CAMERA_POOL_H = 30
CAMERA_POOL_W = 40
CAMERA_CURRENT_SIZE = CAMERA_POOL_H * CAMERA_POOL_W * 3  # 3,600
NEURAL_FEATURE_SIZE = CAMERA_CURRENT_SIZE * 2            # 7,200
ORACLE_FEATURE_SIZE = 517


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class AuditLabels:
    expert_action: np.ndarray
    student_action: np.ndarray
    disagreement: np.ndarray
    critical_error: np.ndarray
    episode: np.ndarray
    step: np.ndarray
    seeds: tuple[str, ...]
    stored_brain: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "AuditLabels":
        with np.load(Path(path), allow_pickle=False) as z:
            required = (
                "expert_action",
                "student_action",
                "disagreement",
                "critical_error",
                "episode",
                "step",
                "seeds",
                "features",
            )
            missing = [name for name in required if name not in z.files]
            if missing:
                raise ValueError(f"DAgger dataset missing arrays: {missing}")

            value = cls(
                expert_action=z["expert_action"].astype(np.uint8, copy=True),
                student_action=z["student_action"].astype(np.uint8, copy=True),
                disagreement=z["disagreement"].astype(np.bool_, copy=True),
                critical_error=z["critical_error"].astype(np.bool_, copy=True),
                episode=z["episode"].astype(np.int16, copy=True),
                step=z["step"].astype(np.int16, copy=True),
                seeds=tuple(str(item) for item in z["seeds"].tolist()),
                stored_brain=z["features"].astype(np.float16, copy=True),
            )
        value.validate()
        return value

    def validate(self) -> None:
        n = len(self.expert_action)
        for name in (
            "student_action",
            "disagreement",
            "critical_error",
            "episode",
            "step",
        ):
            array = getattr(self, name)
            if array.shape != (n,):
                raise ValueError(f"{name} shape mismatch.")
        if self.stored_brain.shape != (n, 21_022):
            raise ValueError("Expected stored DAgger brain features [N,21022].")
        if not np.isfinite(self.stored_brain).all():
            raise ValueError("Stored brain features contain non-finite values.")


def _mix64(values: np.ndarray, seed: int) -> np.ndarray:
    x = values.astype(np.uint64, copy=True)
    x ^= np.uint64(seed & 0xFFFF_FFFF_FFFF_FFFF)
    x += np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@dataclass(frozen=True, slots=True)
class TorchSketch:
    source_size: int
    target_size: int
    bucket: Tensor
    sign: Tensor
    scale: float

    @torch.no_grad()
    def project(self, values: Tensor) -> Tensor:
        flat = values.reshape(-1)
        if flat.numel() != self.source_size:
            raise ValueError(
                f"Sketch source {flat.numel()} != expected {self.source_size}."
            )
        out = torch.zeros(
            self.target_size,
            dtype=torch.float32,
            device=flat.device,
        )
        out.scatter_add_(0, self.bucket, flat.to(torch.float32) * self.sign)
        out.mul_(self.scale)
        return out


def make_sketch(
    source_size: int,
    target_size: int,
    *,
    seed: int,
    device: torch.device,
) -> TorchSketch:
    source = np.arange(source_size, dtype=np.uint64)
    hashed = _mix64(source, seed)
    bucket = (hashed % np.uint64(target_size)).astype(np.int64)
    sign = np.where(
        (hashed >> np.uint64(63)) == 0,
        1.0,
        -1.0,
    ).astype(np.float32)
    average_load = max(1.0, source_size / float(target_size))
    return TorchSketch(
        source_size=source_size,
        target_size=target_size,
        bucket=torch.as_tensor(bucket, dtype=torch.long, device=device),
        sign=torch.as_tensor(sign, dtype=torch.float32, device=device),
        scale=float(1.0 / np.sqrt(average_load)),
    )


@torch.no_grad()
def pool_camera_frame(
    frame: np.ndarray | Tensor,
    *,
    device: torch.device,
) -> Tensor:
    image = torch.as_tensor(frame, device=device)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected camera frame [H,W,3].")
    chw = image.to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    pooled = F.adaptive_avg_pool2d(
        chw,
        (CAMERA_POOL_H, CAMERA_POOL_W),
    )
    return pooled[0].permute(1, 2, 0).reshape(-1)


def camera_now_features(current: Tensor) -> Tensor:
    if current.numel() != CAMERA_CURRENT_SIZE:
        raise ValueError("Camera pooled feature size changed.")
    zeros = torch.zeros_like(current)
    return torch.cat([current, zeros], dim=0)


def camera_delta_features(current: Tensor, previous: Tensor) -> Tensor:
    if current.shape != previous.shape:
        raise ValueError("Current/previous camera pooled shapes differ.")
    return torch.cat([current, current - previous], dim=0)


class StageProbe(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, len(ACTION_ORDER)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


def _stats(features: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = features[rows].astype(np.float32)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    variance = x.var(axis=0, dtype=np.float64).astype(np.float32)
    std = np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32)
    return mean, std


@torch.no_grad()
def _predict(
    model: StageProbe,
    features: np.ndarray,
    rows: np.ndarray,
    mean: Tensor,
    std: Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        x = torch.as_tensor(
            features[batch_rows],
            dtype=torch.float32,
            device=device,
        )
        logits = model((x - mean) / std)
        parts.append(
            logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
        )
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)


def train_fold(
    *,
    features: np.ndarray,
    labels: AuditLabels,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[np.ndarray, dict[str, object]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    mean_np, std_np = _stats(features, train_rows)
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)

    model = StageProbe(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    weights = sample_weights(
        labels.expert_action[train_rows],
        labels.disagreement[train_rows],
        labels.critical_error[train_rows],
    )

    order = np.arange(len(train_rows), dtype=np.int64)
    best_state: dict[str, Tensor] | None = None
    best_score = (-1.0, -1.0)
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)

        for start in range(0, len(order), batch_size):
            local = order[start:start + batch_size]
            rows = train_rows[local]

            x = torch.as_tensor(
                features[rows],
                dtype=torch.float32,
                device=device,
            )
            y = torch.as_tensor(
                labels.expert_action[rows],
                dtype=torch.long,
                device=device,
            )
            w = torch.as_tensor(
                weights[local],
                dtype=torch.float32,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model((x - mean) / std)
            loss_each = F.cross_entropy(
                logits,
                y,
                reduction="none",
                label_smoothing=0.02,
            )
            loss = (loss_each * w).sum() / w.sum().clamp_min(1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        predicted = _predict(
            model,
            features,
            validation_rows,
            mean,
            std,
            device=device,
            batch_size=batch_size,
        )
        metrics = metrics_from_predictions(
            labels.expert_action[validation_rows],
            predicted,
        )
        disagreement = labels.disagreement[validation_rows]
        disagreement_recovery = (
            float(
                (
                    predicted[disagreement]
                    == labels.expert_action[validation_rows][disagreement]
                ).mean()
            )
            if disagreement.any()
            else 0.0
        )
        score = (
            float(metrics["coreMacroAccuracy"]),
            disagreement_recovery,
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("Visual-stage fold produced no checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    predicted = _predict(
        model,
        features,
        validation_rows,
        mean,
        std,
        device=device,
        batch_size=batch_size,
    )

    metrics = metrics_from_predictions(
        labels.expert_action[validation_rows],
        predicted,
    )
    disagreement = labels.disagreement[validation_rows]
    critical = labels.critical_error[validation_rows]

    return predicted, {
        "seed": seed,
        "bestEpoch": best_epoch,
        "validation": metrics,
        "disagreementSupport": int(disagreement.sum()),
        "disagreementRecovery": (
            float(
                (
                    predicted[disagreement]
                    == labels.expert_action[validation_rows][disagreement]
                ).mean()
            )
            if disagreement.any()
            else None
        ),
        "criticalSupport": int(critical.sum()),
        "criticalCorrection": (
            float(
                (
                    predicted[critical]
                    == labels.expert_action[validation_rows][critical]
                ).mean()
            )
            if critical.any()
            else None
        ),
    }


def evaluate_stage(
    *,
    name: str,
    features: np.ndarray,
    labels: AuditLabels,
    folds: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, object]:
    oof = np.full(len(labels.expert_action), 255, dtype=np.uint8)
    fold_reports = []

    for fold_index, (train_rows, validation_rows) in enumerate(folds):
        predicted, report = train_fold(
            features=features,
            labels=labels,
            train_rows=train_rows,
            validation_rows=validation_rows,
            device=device,
            seed=seed + fold_index * 101,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        oof[validation_rows] = predicted
        report["fold"] = fold_index
        fold_reports.append(report)

    if np.any(oof == 255):
        raise RuntimeError(f"OOF coverage incomplete for {name}.")

    metrics = metrics_from_predictions(labels.expert_action, oof)
    disagreement = labels.disagreement
    critical = labels.critical_error

    return {
        "name": name,
        "inputFeatures": int(features.shape[1]),
        "folds": fold_reports,
        "oof": metrics,
        "disagreementRecovery": float(
            (oof[disagreement] == labels.expert_action[disagreement]).mean()
        ),
        "criticalCorrection": float(
            (oof[critical] == labels.expert_action[critical]).mean()
        ),
    }


@torch.no_grad()
def replay_and_collect(
    *,
    labels: AuditLabels,
    frontend: FrozenVisualFrontEnd,
    brain: FrozenMaleCNSCore,
    decoder,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    n = len(labels.expert_action)
    device = brain.device

    camera_now = np.empty((n, NEURAL_FEATURE_SIZE), dtype=np.float16)
    camera_delta = np.empty((n, NEURAL_FEATURE_SIZE), dtype=np.float16)
    retina_sequence = np.empty((n, NEURAL_FEATURE_SIZE), dtype=np.float16)
    brain_readout = np.empty((n, NEURAL_FEATURE_SIZE), dtype=np.float16)
    oracle = np.empty((n, ORACLE_FEATURE_SIZE), dtype=np.float16)

    retina_source_size = frontend.internal_steps * frontend.n_clamped
    retina_sketch = make_sketch(
        retina_source_size,
        NEURAL_FEATURE_SIZE,
        seed=seed + 11,
        device=device,
    )
    brain_sketch = make_sketch(
        21_022,
        NEURAL_FEATURE_SIZE,
        seed=seed + 23,
        device=device,
    )

    verified = 0
    max_abs_brain_error = 0.0

    for episode_index, episode_seed in enumerate(labels.seeds):
        rows = np.flatnonzero(labels.episode == episode_index)
        if len(rows) == 0:
            continue

        expected_steps = labels.step[rows]
        if not np.array_equal(expected_steps, np.arange(len(rows), dtype=np.int16)):
            raise RuntimeError(
                f"Episode {episode_index} does not have contiguous stored steps."
            )

        state = create_game(episode_seed)
        vision_state = frontend.init_state(1)
        brain_state = brain.init_state(1)
        previous_pooled: Tensor | None = None

        for local_step, global_row in enumerate(rows.tolist()):
            frame = render_crossy_neural_frame(state)
            pooled = pool_camera_frame(frame, device=device)
            previous = pooled if previous_pooled is None else previous_pooled

            camera_now[global_row] = (
                camera_now_features(pooled)
                .detach()
                .cpu()
                .to(torch.float16)
                .numpy()
            )
            camera_delta[global_row] = (
                camera_delta_features(pooled, previous)
                .detach()
                .cpu()
                .to(torch.float16)
                .numpy()
            )

            visual_rates, vision_state = frontend.process(frame[None], vision_state)
            retina_sequence[global_row] = (
                retina_sketch.project(visual_rates[:, 0, :])
                .detach()
                .cpu()
                .to(torch.float16)
                .numpy()
            )

            brain_state = brain.step(brain_state, visual_rates)
            exact_brain = brain.decision_features(brain_state)[0]
            brain_readout[global_row] = (
                brain_sketch.project(exact_brain)
                .detach()
                .cpu()
                .to(torch.float16)
                .numpy()
            )

            stored = labels.stored_brain[global_row]
            exact_cpu = exact_brain.detach().cpu().to(torch.float16).numpy()
            max_abs_brain_error = max(
                max_abs_brain_error,
                float(
                    np.max(
                        np.abs(
                            exact_cpu.astype(np.float32)
                            - stored.astype(np.float32)
                        )
                    )
                ),
            )

            oracle_vector = flatten_observation(observe(state))
            if oracle_vector.shape != (ORACLE_FEATURE_SIZE,):
                raise RuntimeError("ObservationV4 oracle feature width changed.")
            oracle[global_row] = oracle_vector.astype(np.float16)

            logits = decoder_logits(
                decoder,
                brain.decision_features(brain_state),
            )[0]
            action_index = int(torch.argmax(logits).item())
            stored_action = int(labels.student_action[global_row])
            if action_index != stored_action:
                raise RuntimeError(
                    f"Replay diverged at episode {episode_index}, step {local_step}: "
                    f"{action_index} != stored {stored_action}."
                )

            state = step_game(state, ACTION_ORDER[action_index]).state
            previous_pooled = pooled.clone()
            verified += 1

        if (episode_index + 1) % 20 == 0 or episode_index + 1 == len(labels.seeds):
            print(
                f"  replay {episode_index + 1:3d}/{len(labels.seeds)} "
                f"rows={verified:,} maxBrainErr={max_abs_brain_error:.6f}",
                flush=True,
            )

    if verified != n:
        raise RuntimeError(f"Replay covered {verified} states, expected {n}.")

    return {
        "camera_now": camera_now,
        "camera_rgb_delta": camera_delta,
        "retina_sequence": retina_sequence,
        "brain_readout": brain_readout,
        "symbolic_oracle": oracle,
    }, {
        "rowsVerified": verified,
        "decoderActionsExact": True,
        "maxAbsBrainFeatureError": max_abs_brain_error,
    }


def _strong_gain(
    downstream: dict[str, object],
    upstream: dict[str, object],
) -> tuple[bool, dict[str, float]]:
    core = (
        float(upstream["oof"]["coreMacroAccuracy"])
        - float(downstream["oof"]["coreMacroAccuracy"])
    )
    disagreement = (
        float(upstream["disagreementRecovery"])
        - float(downstream["disagreementRecovery"])
    )
    critical = (
        float(upstream["criticalCorrection"])
        - float(downstream["criticalCorrection"])
    )
    strong = core >= 0.08 and (disagreement >= 0.08 or critical >= 0.08)
    return strong, {
        "coreMacro": core,
        "disagreementRecovery": disagreement,
        "criticalCorrection": critical,
    }


def interpret(results: dict[str, dict[str, object]]) -> dict[str, object]:
    brain = results["brain_readout"]
    retina = results["retina_sequence"]
    camera_now = results["camera_now"]
    camera_delta = results["camera_rgb_delta"]
    oracle = results["symbolic_oracle"]

    brain_transform, retina_vs_brain = _strong_gain(brain, retina)
    retina_loss, camera_vs_retina = _strong_gain(retina, camera_delta)
    temporal_visual, delta_vs_now = _strong_gain(camera_now, camera_delta)
    symbolic_ceiling, oracle_vs_camera = _strong_gain(camera_delta, oracle)

    if brain_transform:
        diagnosis = "malecns_dynamics_transform_bottleneck"
    elif retina_loss:
        diagnosis = "retina_frontend_bottleneck"
    elif temporal_visual:
        diagnosis = "camera_temporal_information_required"
    elif symbolic_ceiling:
        diagnosis = "neural_camera_visual_information_insufficient_vs_symbolic_state"
    else:
        diagnosis = "no_single_visual_stage_bottleneck_supported"

    return {
        "retinaVsBrain": {
            "strong": brain_transform,
            "gain": retina_vs_brain,
        },
        "cameraDeltaVsRetina": {
            "strong": retina_loss,
            "gain": camera_vs_retina,
        },
        "cameraDeltaVsCameraNow": {
            "strong": temporal_visual,
            "gain": delta_vs_now,
        },
        "symbolicOracleVsCameraDelta": {
            "strong": symbolic_ceiling,
            "gain": oracle_vs_camera,
            "note": (
                "ObservationV4 is a privileged diagnostic ceiling only. It contains "
                "symbolic lane/hazard/motion, previous-action, support, edge and TTC "
                "information and is never a candidate final neural input."
            ),
        },
        "diagnosis": diagnosis,
        "rule": (
            "A stage separation is called strong only when the upstream representation "
            "improves OOF coreMacro by >= +0.08 and improves disagreement recovery or "
            "fatal critical correction by >= +0.08 on the same episode-grouped folds."
        ),
    }


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "MaleCNS Crossy V2 visual-stage information audit. Replays the exact "
            "R0 DAgger trajectories and compares camera pixels, frozen retina output, "
            "MaleCNS readout, and a privileged symbolic ceiling."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument(
        "--dagger-dataset",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "dagger-r1-student-states.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "decoder-r0.pt",
    )
    parser.add_argument(
        "--substrate",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "substrate-w5-h4.npz",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "visual-geometry-160x120.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "malecns-crossy-v2-visual-stage-audit.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    labels = AuditLabels.load(args.dagger_dataset)
    substrate = MaleCNSV2Substrate.load(args.substrate, strict=True)
    geometry = VisualGeometry.load(args.geometry, strict=True)
    decoder = load_decoder(args.checkpoint, device=device)

    if not np.array_equal(
        geometry.cell_body_id,
        substrate.body_ids[substrate.is_clamped],
    ):
        raise SystemExit("Visual geometry and substrate clamped order do not match.")

    frontend = FrozenVisualFrontEnd(
        geometry,
        decision_ms=200.0,
        internal_steps=10,
        device=device,
    )
    brain = FrozenMaleCNSCore(
        substrate,
        decision_ms=200.0,
        internal_steps=10,
        tau_ms=20.0,
        device=device,
    )
    gain = brain.match_rest_gain(target=0.95, iterations=80)

    print("=== MaleCNS Crossy V2 / Visual-Stage Information Audit ===", flush=True)
    print("No production weights are modified.", flush=True)
    print(
        f"states={len(labels.expert_action):,} episodes={len(labels.seeds)} "
        f"neuralFeatureWidth={NEURAL_FEATURE_SIZE:,}",
        flush=True,
    )

    started = time.perf_counter()
    print("\n[1/2] Exact R0 replay + camera/retina/brain/oracle collection...", flush=True)
    features, parity = replay_and_collect(
        labels=labels,
        frontend=frontend,
        brain=brain,
        decoder=decoder,
        seed=args.seed,
    )

    folds = make_episode_folds(
        labels.episode,
        folds=args.folds,
        seed=args.seed,
    )

    print("\n[2/2] Episode-grouped stage probes...", flush=True)
    order = (
        "camera_now",
        "camera_rgb_delta",
        "retina_sequence",
        "brain_readout",
        "symbolic_oracle",
    )
    results: dict[str, dict[str, object]] = {}

    for index, name in enumerate(order):
        print(f"\n  stage: {name}", flush=True)
        result = evaluate_stage(
            name=name,
            features=features[name],
            labels=labels,
            folds=folds,
            device=device,
            seed=args.seed + 1_000 + index * 1_003,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        results[name] = result

        metrics = result["oof"]
        print(
            f"    OOF acc={metrics['accuracy']:.3f} "
            f"coreMacro={metrics['coreMacroAccuracy']:.3f} "
            f"L/R/W={metrics['perActionAccuracy']['left']}/"
            f"{metrics['perActionAccuracy']['right']}/"
            f"{metrics['perActionAccuracy']['wait']}",
            flush=True,
        )
        print(
            f"    disagree={result['disagreementRecovery']:.3f} "
            f"critical={result['criticalCorrection']:.3f}",
            flush=True,
        )

    conclusion = interpret(results)

    report = {
        "version": 1,
        "auditVersion": AUDIT_VERSION,
        "purpose": (
            "Localize information loss upstream of the MaleCNS readout by comparing "
            "the exact neural camera, frozen retinal sequence, frozen MaleCNS readout "
            "and a privileged symbolic diagnostic ceiling on the same R0 states."
        ),
        "productionWeightsChanged": False,
        "frozenSystem": {
            "camera": True,
            "visualFrontend": True,
            "MaleCNS": True,
            "decoderR0": True,
            "globalGain": gain,
        },
        "protocol": {
            "states": len(labels.expert_action),
            "episodes": len(labels.seeds),
            "folds": args.folds,
            "splitUnit": "episode",
            "sameExpertLabels": True,
            "sameR0Trajectories": True,
            "sameCorrectionWeighting": True,
            "sameHiddenProbeCapacity": 256,
            "neuralFeatureWidth": NEURAL_FEATURE_SIZE,
            "cameraNow": "40x30 RGB average pool + 3600 zero temporal slots",
            "cameraRgbDelta": "40x30 RGB current + current-minus-previous RGB",
            "retinaSequence": (
                "all 10 x 30,906 frozen visual rates projected label-free by "
                "deterministic CountSketch to 7,200 features"
            ),
            "brainReadout": (
                "exact 21,022 DN+VPN current+delta projected label-free by "
                "deterministic CountSketch to 7,200 features"
            ),
            "symbolicOracle": (
                "existing 517-value ObservationV4, diagnostic ceiling only; "
                "never exported to final neural controller"
            ),
        },
        "replayParity": parity,
        "results": results,
        "interpretation": conclusion,
        "elapsedSeconds": time.perf_counter() - started,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== VISUAL-STAGE AUDIT SUMMARY ===", flush=True)
    for name in order:
        result = results[name]
        print(
            f"{name:18s} "
            f"core={result['oof']['coreMacroAccuracy']:.3f} "
            f"disagree={result['disagreementRecovery']:.3f} "
            f"critical={result['criticalCorrection']:.3f}",
            flush=True,
        )

    print(f"\ndiagnosis: {conclusion['diagnosis']}", flush=True)
    print(f"report: {args.output.resolve()}", flush=True)
    print("\nVISUAL-STAGE INFORMATION AUDIT: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
