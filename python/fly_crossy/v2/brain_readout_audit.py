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

from fly_crossy.env import create_game, step_game
from fly_crossy.schema import ACTION_ORDER

from .core import FrozenMaleCNSCore
from .crossy_camera import render_crossy_neural_frame
from .dagger_probe import make_episode_folds, sample_weights
from .decoder import decoder_logits, load_decoder
from .information_probe import metrics_from_predictions
from .substrate import MaleCNSV2Substrate
from .vision_frontend import FrozenVisualFrontEnd
from .vision_geometry import VisualGeometry


ACTION_NAMES = tuple(action.value for action in ACTION_ORDER)
DEFAULT_TOTAL_FEATURES = 8_192
DEFAULT_HALF_FEATURES = DEFAULT_TOTAL_FEATURES // 2
AUDIT_VERSION = "malecns-crossy-v2-brain-readout-audit-1"


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
    stored_readout_features: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "AuditLabels":
        with np.load(path, allow_pickle=False) as z:
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

            labels = cls(
                expert_action=z["expert_action"].astype(np.uint8, copy=True),
                student_action=z["student_action"].astype(np.uint8, copy=True),
                disagreement=z["disagreement"].astype(np.bool_, copy=True),
                critical_error=z["critical_error"].astype(np.bool_, copy=True),
                episode=z["episode"].astype(np.int16, copy=True),
                step=z["step"].astype(np.int16, copy=True),
                seeds=tuple(str(item) for item in z["seeds"].tolist()),
                stored_readout_features=z["features"].astype(np.float16, copy=True),
            )
        labels.validate()
        return labels

    def validate(self) -> None:
        n = len(self.expert_action)
        for name in (
            "student_action",
            "disagreement",
            "critical_error",
            "episode",
            "step",
        ):
            value = getattr(self, name)
            if value.shape != (n,):
                raise ValueError(f"{name} shape mismatch.")
        if self.stored_readout_features.shape != (n, 21_022):
            raise ValueError(
                "Stored DAgger readout features must have shape (N, 21022)."
            )
        if not np.isfinite(self.stored_readout_features).all():
            raise ValueError("Stored readout features contain non-finite values.")


@dataclass(frozen=True, slots=True)
class PopulationProjector:
    name: str
    node_indices: Tensor
    bucket: Tensor
    sign: Tensor
    half_features: int
    scale: float

    @property
    def cell_count(self) -> int:
        return int(self.node_indices.numel())

    @property
    def output_size(self) -> int:
        return self.half_features * 2

    @torch.no_grad()
    def project(self, current: Tensor, delta: Tensor) -> Tensor:
        """CountSketch current rates and temporal deltas into equal-width features."""
        cur = current.index_select(0, self.node_indices) * self.sign
        dif = delta.index_select(0, self.node_indices) * self.sign

        out_cur = torch.zeros(
            self.half_features,
            dtype=current.dtype,
            device=current.device,
        )
        out_delta = torch.zeros_like(out_cur)
        out_cur.scatter_add_(0, self.bucket, cur)
        out_delta.scatter_add_(0, self.bucket, dif)

        if self.scale != 1.0:
            out_cur.mul_(self.scale)
            out_delta.mul_(self.scale)

        return torch.cat([out_cur, out_delta], dim=0)


def _mix64(values: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic SplitMix64-style hash, label-free."""
    x = values.astype(np.uint64, copy=True)
    x ^= np.uint64(seed & 0xFFFF_FFFF_FFFF_FFFF)
    x += np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def make_projector(
    *,
    name: str,
    substrate: MaleCNSV2Substrate,
    node_mask: np.ndarray,
    half_features: int,
    seed: int,
    device: torch.device,
) -> PopulationProjector:
    indices = np.flatnonzero(node_mask).astype(np.int64)
    if len(indices) == 0:
        raise ValueError(f"Population {name} is empty.")
    if np.any(substrate.is_clamped[indices]):
        raise ValueError(f"Population {name} unexpectedly includes clamped visual cells.")

    # Hash anatomical body IDs, not row positions, so the projection remains
    # stable if local indexing is regenerated without changing the cells.
    hashed = _mix64(substrate.body_ids[indices], seed)
    bucket = (hashed % np.uint64(half_features)).astype(np.int64)
    sign = np.where((hashed >> np.uint64(63)) == 0, 1.0, -1.0).astype(np.float32)

    # CountSketch norm equalization only. Fold standardization handles all
    # remaining per-feature scale differences.
    average_load = max(1.0, len(indices) / float(half_features))
    scale = float(1.0 / np.sqrt(average_load))

    return PopulationProjector(
        name=name,
        node_indices=torch.as_tensor(indices, dtype=torch.long, device=device),
        bucket=torch.as_tensor(bucket, dtype=torch.long, device=device),
        sign=torch.as_tensor(sign, dtype=torch.float32, device=device),
        half_features=half_features,
        scale=scale,
    )


def population_masks(substrate: MaleCNSV2Substrate) -> dict[str, np.ndarray]:
    dynamic = substrate.is_dynamic.copy()
    backward = substrate.backward_hops

    masks = {
        # Exact cells the R0 decoder currently reads, but compressed through the
        # same label-free projection as every broader population.
        "dn_vpn_readout": dynamic & substrate.decision_readout_mask,
        # Increasingly broad populations near the descending output side.
        "near_dn_hop1": dynamic & (backward >= 0) & (backward <= 1),
        "near_dn_hop2": dynamic & (backward >= 0) & (backward <= 2),
        # All recurrent/dynamic cells in the selected 138,968-cell MaleCNS.
        "all_dynamic": dynamic,
    }

    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"Population mask {name} is empty.")
    return masks


class AuditProbe(nn.Module):
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


def _feature_stats(features: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = features[rows].astype(np.float32)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    variance = x.var(axis=0, dtype=np.float64).astype(np.float32)
    std = np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32)
    return mean, std


@torch.no_grad()
def _predict(
    model: AuditProbe,
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

    mean_np, std_np = _feature_stats(features, train_rows)
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)

    model = AuditProbe(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    train_labels = labels.expert_action[train_rows]
    weights = sample_weights(
        train_labels,
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
        disagreement_mask = labels.disagreement[validation_rows]
        disagreement_accuracy = (
            float(
                (
                    predicted[disagreement_mask]
                    == labels.expert_action[validation_rows][disagreement_mask]
                ).mean()
            )
            if disagreement_mask.any()
            else 0.0
        )
        score = (
            float(metrics["coreMacroAccuracy"]),
            disagreement_accuracy,
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
        raise RuntimeError("Audit fold produced no checkpoint.")

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
    disagreement_mask = labels.disagreement[validation_rows]
    critical_mask = labels.critical_error[validation_rows]

    return predicted, {
        "seed": seed,
        "bestEpoch": best_epoch,
        "validation": metrics,
        "disagreementSupport": int(disagreement_mask.sum()),
        "disagreementRecovery": (
            float(
                (
                    predicted[disagreement_mask]
                    == labels.expert_action[validation_rows][disagreement_mask]
                ).mean()
            )
            if disagreement_mask.any()
            else None
        ),
        "criticalSupport": int(critical_mask.sum()),
        "criticalCorrection": (
            float(
                (
                    predicted[critical_mask]
                    == labels.expert_action[validation_rows][critical_mask]
                ).mean()
            )
            if critical_mask.any()
            else None
        ),
    }


def evaluate_population_oof(
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
    disagreement_mask = labels.disagreement
    critical_mask = labels.critical_error

    disagreement_recovery = float(
        (oof[disagreement_mask] == labels.expert_action[disagreement_mask]).mean()
    )
    critical_correction = float(
        (oof[critical_mask] == labels.expert_action[critical_mask]).mean()
    )

    return {
        "name": name,
        "inputFeatures": int(features.shape[1]),
        "folds": fold_reports,
        "oof": metrics,
        "disagreementRecovery": disagreement_recovery,
        "criticalCorrection": critical_correction,
    }


@torch.no_grad()
def replay_and_collect(
    *,
    labels: AuditLabels,
    projectors: dict[str, PopulationProjector],
    frontend: FrozenVisualFrontEnd,
    brain: FrozenMaleCNSCore,
    decoder,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    n = len(labels.expert_action)
    feature_sets = {
        name: np.empty((n, projector.output_size), dtype=np.float16)
        for name, projector in projectors.items()
    }

    parity_rows = 0
    max_abs_readout_error = 0.0
    row_cursor = 0

    for episode_index, seed in enumerate(labels.seeds):
        rows = np.flatnonzero(labels.episode == episode_index)
        if len(rows) == 0:
            continue
        expected_steps = labels.step[rows]
        if not np.array_equal(expected_steps, np.arange(len(rows), dtype=np.int16)):
            raise RuntimeError(
                f"Episode {episode_index} does not have contiguous stored steps."
            )

        state = create_game(seed)
        vision_state = frontend.init_state(1)
        brain_state = brain.init_state(1)

        for local_step, global_row in enumerate(rows.tolist()):
            previous_rates = brain.rates(brain_state)[:, 0].clone()

            frame = render_crossy_neural_frame(state)
            visual_rates, vision_state = frontend.process(frame[None], vision_state)
            brain_state = brain.step(brain_state, visual_rates)

            current_rates = brain.rates(brain_state)[:, 0]
            delta_rates = current_rates - previous_rates

            # Exact replay guard against the already-collected DAgger R1 features.
            readout = (
                brain.decision_features(brain_state)[0]
                .detach()
                .to("cpu", dtype=torch.float16)
                .numpy()
            )
            stored = labels.stored_readout_features[global_row]
            if readout.shape != stored.shape:
                raise RuntimeError("Stored readout feature width changed.")
            max_abs_readout_error = max(
                max_abs_readout_error,
                float(
                    np.max(
                        np.abs(
                            readout.astype(np.float32)
                            - stored.astype(np.float32)
                        )
                    )
                ),
            )

            logits = decoder_logits(
                decoder,
                brain.decision_features(brain_state),
            )[0]
            action_index = int(torch.argmax(logits).item())
            if action_index != int(labels.student_action[global_row]):
                raise RuntimeError(
                    f"Replay diverged at episode {episode_index}, step {local_step}: "
                    f"decoder action {action_index} != stored "
                    f"{int(labels.student_action[global_row])}."
                )

            for name, projector in projectors.items():
                feature_sets[name][global_row] = (
                    projector.project(current_rates, delta_rates)
                    .detach()
                    .to("cpu", dtype=torch.float16)
                    .numpy()
                )

            state = step_game(state, ACTION_ORDER[action_index]).state
            parity_rows += 1
            row_cursor += 1

        if (episode_index + 1) % 20 == 0 or episode_index + 1 == len(labels.seeds):
            print(
                f"  replay {episode_index + 1:3d}/{len(labels.seeds)} "
                f"rows={parity_rows:,} maxReadoutErr={max_abs_readout_error:.6f}",
                flush=True,
            )

    if parity_rows != n:
        raise RuntimeError(f"Replay covered {parity_rows} rows, expected {n}.")

    return feature_sets, {
        "rowsVerified": parity_rows,
        "storedRows": n,
        "decoderActionsExact": True,
        "maxAbsReadoutFeatureError": max_abs_readout_error,
        "readoutFloat16Parity": max_abs_readout_error == 0.0,
    }


def interpretation(
    results: dict[str, dict[str, object]],
) -> dict[str, object]:
    baseline = results["dn_vpn_readout"]
    base_core = float(baseline["oof"]["coreMacroAccuracy"])
    base_dis = float(baseline["disagreementRecovery"])
    base_crit = float(baseline["criticalCorrection"])

    broader = [
        result
        for name, result in results.items()
        if name != "dn_vpn_readout"
    ]
    best_core_result = max(
        broader,
        key=lambda result: float(result["oof"]["coreMacroAccuracy"]),
    )
    best_dis_result = max(
        broader,
        key=lambda result: float(result["disagreementRecovery"]),
    )
    best_crit_result = max(
        broader,
        key=lambda result: float(result["criticalCorrection"]),
    )

    core_gain = float(best_core_result["oof"]["coreMacroAccuracy"]) - base_core
    dis_gain = float(best_dis_result["disagreementRecovery"]) - base_dis
    crit_gain = float(best_crit_result["criticalCorrection"]) - base_crit

    readout_bottleneck_evidence = (
        core_gain >= 0.08
        and (dis_gain >= 0.08 or crit_gain >= 0.08)
    )

    return {
        "baseline": {
            "population": "dn_vpn_readout",
            "coreMacro": base_core,
            "disagreementRecovery": base_dis,
            "criticalCorrection": base_crit,
        },
        "bestBroaderCoreMacro": {
            "population": best_core_result["name"],
            "value": best_core_result["oof"]["coreMacroAccuracy"],
            "gainVsReadout": core_gain,
        },
        "bestBroaderDisagreementRecovery": {
            "population": best_dis_result["name"],
            "value": best_dis_result["disagreementRecovery"],
            "gainVsReadout": dis_gain,
        },
        "bestBroaderCriticalCorrection": {
            "population": best_crit_result["name"],
            "value": best_crit_result["criticalCorrection"],
            "gainVsReadout": crit_gain,
        },
        "readoutBottleneckEvidence": readout_bottleneck_evidence,
        "rule": (
            "Strong readout-bottleneck evidence requires >= +0.08 OOF coreMacro "
            "from a broader population and >= +0.08 on either disagreement recovery "
            "or immediately-fatal critical correction. Otherwise the audit does not "
            "justify replacing the readout; representation/input remains the leading suspect."
        ),
    }


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Structural MaleCNS V2 readout audit. Replays the exact DAgger R1 "
            "student trajectories and asks whether broader internal recurrent populations "
            "contain more recoverable expert-decision information than DN+VPN."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-features", type=int, default=DEFAULT_TOTAL_FEATURES)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=109)
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
        default=root / "reports" / "malecns-crossy-v2-brain-readout-audit.json",
    )
    args = parser.parse_args()

    if args.total_features <= 0 or args.total_features % 2 != 0:
        raise SystemExit("--total-features must be a positive even integer.")

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

    masks = population_masks(substrate)
    half = args.total_features // 2
    projectors = {
        name: make_projector(
            name=name,
            substrate=substrate,
            node_mask=mask,
            half_features=half,
            seed=args.seed + index * 10_003,
            device=device,
        )
        for index, (name, mask) in enumerate(masks.items())
    }

    print("=== MaleCNS Crossy V2 / Structural Brain Readout Audit ===", flush=True)
    print("No production weights are modified.", flush=True)
    print(
        f"states={len(labels.expert_action):,} episodes={len(labels.seeds)} "
        f"equalizedFeatures={args.total_features:,}",
        flush=True,
    )
    print("populations:", flush=True)
    for name, projector in projectors.items():
        print(
            f"  {name:16s}: {projector.cell_count:,} recurrent cells",
            flush=True,
        )

    started = time.perf_counter()
    print("\n[1/2] Exact R0 trajectory replay + internal-state collection...", flush=True)
    features, parity = replay_and_collect(
        labels=labels,
        projectors=projectors,
        frontend=frontend,
        brain=brain,
        decoder=decoder,
    )
    print(
        f"  replay parity rows={parity['rowsVerified']:,} "
        f"maxReadoutErr={parity['maxAbsReadoutFeatureError']:.6f}",
        flush=True,
    )

    folds = make_episode_folds(
        labels.episode,
        folds=args.folds,
        seed=args.seed,
    )

    print("\n[2/2] Equal-capacity grouped probes...", flush=True)
    results: dict[str, dict[str, object]] = {}

    for population_index, name in enumerate(
        ("dn_vpn_readout", "near_dn_hop1", "near_dn_hop2", "all_dynamic")
    ):
        print(f"\n  population: {name}", flush=True)
        result = evaluate_population_oof(
            name=name,
            features=features[name],
            labels=labels,
            folds=folds,
            device=device,
            seed=args.seed + 1_001 + population_index * 1_003,
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
            f"    disagreementRecovery={result['disagreementRecovery']:.3f} "
            f"criticalCorrection={result['criticalCorrection']:.3f}",
            flush=True,
        )

    conclusion = interpretation(results)

    report = {
        "version": 1,
        "auditVersion": AUDIT_VERSION,
        "purpose": (
            "Localize the MaleCNS V2 bottleneck: compare the current DN+VPN readout "
            "against increasingly broad recurrent populations on the exact same "
            "R0 student-state trajectories, labels, grouped folds, feature width "
            "and probe capacity."
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
            "projection": (
                "deterministic label-free CountSketch of current recurrent rates "
                "and first temporal delta into equal-width channels"
            ),
            "totalFeaturesPerPopulation": args.total_features,
            "probe": f"{args.total_features} -> 256 GELU LayerNorm -> 5",
            "sameExpertLabels": True,
            "sameR0Trajectories": True,
            "sameCorrectionWeightingAsDaggerLearnabilityProbe": True,
        },
        "populationCellCounts": {
            name: projector.cell_count
            for name, projector in projectors.items()
        },
        "replayParity": parity,
        "results": results,
        "interpretation": conclusion,
        "elapsedSeconds": time.perf_counter() - started,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== STRUCTURAL AUDIT SUMMARY ===", flush=True)
    for name, result in results.items():
        print(
            f"{name:16s} "
            f"coreMacro={result['oof']['coreMacroAccuracy']:.3f} "
            f"disagree={result['disagreementRecovery']:.3f} "
            f"critical={result['criticalCorrection']:.3f}",
            flush=True,
        )

    print()
    print(
        "best broader core gain: "
        f"{conclusion['bestBroaderCoreMacro']['population']} "
        f"{conclusion['bestBroaderCoreMacro']['gainVsReadout']:+.3f}",
        flush=True,
    )
    print(
        "best disagreement gain: "
        f"{conclusion['bestBroaderDisagreementRecovery']['population']} "
        f"{conclusion['bestBroaderDisagreementRecovery']['gainVsReadout']:+.3f}",
        flush=True,
    )
    print(
        "best critical gain: "
        f"{conclusion['bestBroaderCriticalCorrection']['population']} "
        f"{conclusion['bestBroaderCriticalCorrection']['gainVsReadout']:+.3f}",
        flush=True,
    )
    print(
        f"readout bottleneck evidence: "
        f"{conclusion['readoutBottleneckEvidence']}",
        flush=True,
    )
    print(f"report: {args.output.resolve()}", flush=True)

    if conclusion["readoutBottleneckEvidence"]:
        print("\nSTRUCTURAL READOUT AUDIT: READOUT BOTTLENECK SUPPORTED", flush=True)
        print(
            "Broader recurrent MaleCNS activity contains materially more recoverable "
            "expert-decision information than DN+VPN alone.",
            flush=True,
        )
    else:
        print("\nSTRUCTURAL READOUT AUDIT: READOUT BOTTLENECK NOT SUPPORTED", flush=True)
        print(
            "Do not replace the readout based on this evidence. The leading bottleneck "
            "moves upstream toward representation/input/state context.",
            flush=True,
        )


if __name__ == "__main__":
    main()
