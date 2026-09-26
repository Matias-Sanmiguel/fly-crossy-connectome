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

from fly_crossy.schema import ACTION_ORDER

from .dagger_probe import make_episode_folds, sample_weights
from .information_probe import metrics_from_predictions


ACTION_NAMES = tuple(action.value for action in ACTION_ORDER)
BRAIN_SOURCE_SIZE = 21_022
BRAIN_PROJECTED_SIZE = 8_192
CONTEXT_SIZE = 32
TOTAL_INPUT_SIZE = BRAIN_PROJECTED_SIZE + CONTEXT_SIZE
HISTORY_STEPS = 4
WORLD_HALF_WIDTH = 5.0
AUDIT_VERSION = "malecns-crossy-v2-context-audit-1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class ContextDataset:
    brain: np.ndarray
    expert_action: np.ndarray
    student_action: np.ndarray
    disagreement: np.ndarray
    critical_error: np.ndarray
    episode: np.ndarray
    step: np.ndarray
    fly_column: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "ContextDataset":
        with np.load(Path(path), allow_pickle=False) as z:
            required = (
                "features",
                "expert_action",
                "student_action",
                "disagreement",
                "critical_error",
                "episode",
                "step",
                "fly_column",
            )
            missing = [name for name in required if name not in z.files]
            if missing:
                raise ValueError(f"DAgger dataset missing arrays: {missing}")

            dataset = cls(
                brain=z["features"].astype(np.float16, copy=True),
                expert_action=z["expert_action"].astype(np.uint8, copy=True),
                student_action=z["student_action"].astype(np.uint8, copy=True),
                disagreement=z["disagreement"].astype(np.bool_, copy=True),
                critical_error=z["critical_error"].astype(np.bool_, copy=True),
                episode=z["episode"].astype(np.int16, copy=True),
                step=z["step"].astype(np.int16, copy=True),
                fly_column=z["fly_column"].astype(np.float32, copy=True),
            )
        dataset.validate()
        return dataset

    def validate(self) -> None:
        n = len(self.expert_action)
        if self.brain.shape != (n, BRAIN_SOURCE_SIZE):
            raise ValueError(
                f"Expected brain feature shape ({n}, {BRAIN_SOURCE_SIZE}), "
                f"got {self.brain.shape}."
            )
        for name in (
            "student_action",
            "disagreement",
            "critical_error",
            "episode",
            "step",
            "fly_column",
        ):
            value = getattr(self, name)
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},).")
        if not np.isfinite(self.brain).all():
            raise ValueError("Brain features contain non-finite values.")
        if not np.isfinite(self.fly_column).all():
            raise ValueError("fly_column contains non-finite values.")


def _mix64(values: np.ndarray, seed: int) -> np.ndarray:
    x = values.astype(np.uint64, copy=True)
    x ^= np.uint64(seed & 0xFFFF_FFFF_FFFF_FFFF)
    x += np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@dataclass(frozen=True, slots=True)
class Sketch:
    bucket: np.ndarray
    sign: np.ndarray
    scale: float

    def project(self, source: np.ndarray) -> np.ndarray:
        values = source.astype(np.float32, copy=False) * self.sign
        out = np.bincount(
            self.bucket,
            weights=values,
            minlength=BRAIN_PROJECTED_SIZE,
        ).astype(np.float32, copy=False)
        out *= self.scale
        return out


def make_sketch(seed: int) -> Sketch:
    source_index = np.arange(BRAIN_SOURCE_SIZE, dtype=np.uint64)
    hashed = _mix64(source_index, seed)
    bucket = (hashed % np.uint64(BRAIN_PROJECTED_SIZE)).astype(np.int64)
    sign = np.where(
        (hashed >> np.uint64(63)) == 0,
        1.0,
        -1.0,
    ).astype(np.float32)
    load = max(1.0, BRAIN_SOURCE_SIZE / float(BRAIN_PROJECTED_SIZE))
    return Sketch(
        bucket=bucket,
        sign=sign,
        scale=float(1.0 / np.sqrt(load)),
    )


def previous_rows(dataset: ContextDataset, row: int, max_lag: int) -> list[int | None]:
    """Current row followed by previous rows from the same episode."""
    result: list[int | None] = [row]
    episode = int(dataset.episode[row])
    step = int(dataset.step[row])

    for lag in range(1, max_lag):
        previous = row - lag
        if (
            previous >= 0
            and int(dataset.episode[previous]) == episode
            and int(dataset.step[previous]) == step - lag
        ):
            result.append(previous)
        else:
            result.append(None)
    return result


def motor_context(dataset: ContextDataset, row: int, *, include_column: bool) -> np.ndarray:
    """Biologically plausible diagnostic context: efference-copy history + body column.

    Slots:
      0..19  = previous 4 student actions, 5-way one-hot each
      20..23 = availability mask for those previous actions
      24     = normalized current lateral body position (optional)
      25     = normalized |column| / edge proximity (optional)
      26..31 = reserved zeros
    """
    context = np.zeros(CONTEXT_SIZE, dtype=np.float32)
    episode = int(dataset.episode[row])
    step = int(dataset.step[row])

    for lag in range(1, 5):
        previous = row - lag
        if (
            previous >= 0
            and int(dataset.episode[previous]) == episode
            and int(dataset.step[previous]) == step - lag
        ):
            action = int(dataset.student_action[previous])
            context[(lag - 1) * len(ACTION_ORDER) + action] = 1.0
            context[20 + lag - 1] = 1.0

    if include_column:
        normalized = float(
            np.clip(dataset.fly_column[row] / WORLD_HALF_WIDTH, -1.0, 1.0)
        )
        context[24] = normalized
        context[25] = abs(normalized)

    return context


def build_condition_features(
    dataset: ContextDataset,
    *,
    condition: str,
    seed: int,
) -> np.ndarray:
    if condition not in {
        "brain_now",
        "brain_history4",
        "brain_now_efference",
        "brain_now_efference_column",
        "brain_history4_efference_column",
        "context_only_efference_column",
    }:
        raise ValueError(f"Unknown context-audit condition: {condition}")

    n = len(dataset.expert_action)
    features = np.zeros((n, TOTAL_INPUT_SIZE), dtype=np.float32)

    # Distinct, label-free sketches preserve temporal-lag identity while every
    # condition remains exactly the same input width.
    sketches = [make_sketch(seed + lag * 10_007) for lag in range(HISTORY_STEPS)]
    history_condition = condition in {
        "brain_history4",
        "brain_history4_efference_column",
    }
    uses_brain = condition != "context_only_efference_column"
    uses_efference = condition in {
        "brain_now_efference",
        "brain_now_efference_column",
        "brain_history4_efference_column",
        "context_only_efference_column",
    }
    uses_column = condition in {
        "brain_now_efference_column",
        "brain_history4_efference_column",
        "context_only_efference_column",
    }

    for row in range(n):
        if uses_brain:
            rows = previous_rows(
                dataset,
                row,
                HISTORY_STEPS if history_condition else 1,
            )
            projected = np.zeros(BRAIN_PROJECTED_SIZE, dtype=np.float32)
            contributors = 0
            for lag, source_row in enumerate(rows):
                if source_row is None:
                    continue
                projected += sketches[lag].project(dataset.brain[source_row])
                contributors += 1
            if contributors:
                projected /= np.sqrt(float(contributors))
            features[row, :BRAIN_PROJECTED_SIZE] = projected

        if uses_efference or uses_column:
            context = motor_context(
                dataset,
                row,
                include_column=uses_column,
            )
            if not uses_efference:
                context[:24] = 0.0
            features[row, BRAIN_PROJECTED_SIZE:] = context

    return features.astype(np.float16)


class ContextProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(TOTAL_INPUT_SIZE, 256),
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
def predict(
    model: ContextProbe,
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
        batch = rows[start:start + batch_size]
        x = torch.as_tensor(
            features[batch],
            dtype=torch.float32,
            device=device,
        )
        logits = model((x - mean) / std)
        parts.append(
            logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
        )
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)


def train_fold(
    dataset: ContextDataset,
    features: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    *,
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

    model = ContextProbe().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    weights = sample_weights(
        dataset.expert_action[train_rows],
        dataset.disagreement[train_rows],
        dataset.critical_error[train_rows],
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
                dataset.expert_action[rows],
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

        pred = predict(
            model,
            features,
            validation_rows,
            mean,
            std,
            device=device,
            batch_size=batch_size,
        )
        metrics = metrics_from_predictions(
            dataset.expert_action[validation_rows],
            pred,
        )
        disagree = dataset.disagreement[validation_rows]
        disagreement_recovery = (
            float(
                (
                    pred[disagree]
                    == dataset.expert_action[validation_rows][disagree]
                ).mean()
            )
            if disagree.any()
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
        raise RuntimeError("Context audit fold produced no checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    pred = predict(
        model,
        features,
        validation_rows,
        mean,
        std,
        device=device,
        batch_size=batch_size,
    )

    disagree = dataset.disagreement[validation_rows]
    critical = dataset.critical_error[validation_rows]
    metrics = metrics_from_predictions(
        dataset.expert_action[validation_rows],
        pred,
    )

    return pred, {
        "seed": seed,
        "bestEpoch": best_epoch,
        "validation": metrics,
        "disagreementSupport": int(disagree.sum()),
        "disagreementRecovery": (
            float(
                (
                    pred[disagree]
                    == dataset.expert_action[validation_rows][disagree]
                ).mean()
            )
            if disagree.any()
            else None
        ),
        "criticalSupport": int(critical.sum()),
        "criticalCorrection": (
            float(
                (
                    pred[critical]
                    == dataset.expert_action[validation_rows][critical]
                ).mean()
            )
            if critical.any()
            else None
        ),
    }


def evaluate_condition(
    dataset: ContextDataset,
    features: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, object]:
    oof = np.full(len(dataset.expert_action), 255, dtype=np.uint8)
    fold_reports = []

    for fold_index, (train_rows, validation_rows) in enumerate(folds):
        pred, report = train_fold(
            dataset,
            features,
            train_rows,
            validation_rows,
            device=device,
            seed=seed + fold_index * 101,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        oof[validation_rows] = pred
        report["fold"] = fold_index
        fold_reports.append(report)

    if np.any(oof == 255):
        raise RuntimeError("OOF coverage incomplete.")

    metrics = metrics_from_predictions(dataset.expert_action, oof)
    disagree = dataset.disagreement
    critical = dataset.critical_error

    return {
        "folds": fold_reports,
        "oof": metrics,
        "disagreementRecovery": float(
            (oof[disagree] == dataset.expert_action[disagree]).mean()
        ),
        "criticalCorrection": float(
            (oof[critical] == dataset.expert_action[critical]).mean()
        ),
    }


def interpret(results: dict[str, dict[str, object]]) -> dict[str, object]:
    baseline = results["brain_now"]
    base_core = float(baseline["oof"]["coreMacroAccuracy"])
    base_dis = float(baseline["disagreementRecovery"])
    base_critical = float(baseline["criticalCorrection"])

    def gains(name: str) -> dict[str, float]:
        result = results[name]
        return {
            "coreMacro": float(result["oof"]["coreMacroAccuracy"]) - base_core,
            "disagreementRecovery": float(result["disagreementRecovery"]) - base_dis,
            "criticalCorrection": float(result["criticalCorrection"]) - base_critical,
        }

    history_gain = gains("brain_history4")
    efference_gain = gains("brain_now_efference")
    proprio_gain = gains("brain_now_efference_column")
    combined_gain = gains("brain_history4_efference_column")

    def strong(gain: dict[str, float]) -> bool:
        return (
            gain["coreMacro"] >= 0.08
            and (
                gain["disagreementRecovery"] >= 0.08
                or gain["criticalCorrection"] >= 0.08
            )
        )

    if strong(history_gain):
        diagnosis = "short_temporal_context_missing"
    elif strong(efference_gain):
        diagnosis = "motor_efference_context_missing"
    elif strong(proprio_gain):
        diagnosis = "motor_plus_lateral_self_state_missing"
    elif strong(combined_gain):
        diagnosis = "combined_temporal_and_self_state_context_missing"
    else:
        diagnosis = "context_hypothesis_not_supported"

    return {
        "baseline": {
            "coreMacro": base_core,
            "disagreementRecovery": base_dis,
            "criticalCorrection": base_critical,
        },
        "gainsVsBrainNow": {
            "brain_history4": history_gain,
            "brain_now_efference": efference_gain,
            "brain_now_efference_column": proprio_gain,
            "brain_history4_efference_column": combined_gain,
        },
        "diagnosis": diagnosis,
        "strongContextEvidence": diagnosis != "context_hypothesis_not_supported",
        "rule": (
            "Strong context evidence requires >= +0.08 OOF coreMacro and >= +0.08 "
            "on disagreement recovery or fatal critical correction, versus the same "
            "compressed brain-now baseline. If no condition reaches that, do not add "
            "context to production yet; move upstream to camera/retina sufficiency."
        ),
    }


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "MaleCNS Crossy V2 context sufficiency audit. Uses the exact stored DAgger "
            "student states to test whether short temporal history, motor efference copy, "
            "or lateral self-state restores expert-decision information."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "dagger-r1-student-states.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "malecns-crossy-v2-context-audit.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    dataset = ContextDataset.load(args.dataset)
    folds = make_episode_folds(
        dataset.episode,
        folds=args.folds,
        seed=args.seed,
    )

    conditions = (
        "brain_now",
        "brain_history4",
        "brain_now_efference",
        "brain_now_efference_column",
        "brain_history4_efference_column",
        "context_only_efference_column",
    )

    print("=== MaleCNS Crossy V2 / Context Sufficiency Audit ===", flush=True)
    print("No production weights are modified.", flush=True)
    print(
        f"states={len(dataset.expert_action):,} "
        f"episodes={len(np.unique(dataset.episode))} "
        f"inputWidth={TOTAL_INPUT_SIZE:,}",
        flush=True,
    )

    started = time.perf_counter()
    results: dict[str, dict[str, object]] = {}

    for index, condition in enumerate(conditions):
        print(f"\n[{index + 1}/{len(conditions)}] {condition}", flush=True)
        features = build_condition_features(
            dataset,
            condition=condition,
            seed=args.seed + index * 10_009,
        )
        result = evaluate_condition(
            dataset,
            features,
            folds,
            device=device,
            seed=args.seed + 1_000 + index * 1_003,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        results[condition] = result
        metrics = result["oof"]
        print(
            f"  OOF acc={metrics['accuracy']:.3f} "
            f"coreMacro={metrics['coreMacroAccuracy']:.3f} "
            f"L/R/W={metrics['perActionAccuracy']['left']}/"
            f"{metrics['perActionAccuracy']['right']}/"
            f"{metrics['perActionAccuracy']['wait']}",
            flush=True,
        )
        print(
            f"  disagree={result['disagreementRecovery']:.3f} "
            f"critical={result['criticalCorrection']:.3f}",
            flush=True,
        )

    conclusion = interpret(results)

    report = {
        "version": 1,
        "auditVersion": AUDIT_VERSION,
        "purpose": (
            "Discriminate missing short temporal context from missing motor/self-state "
            "context after the structural readout audit failed to support a broader "
            "MaleCNS readout."
        ),
        "productionWeightsChanged": False,
        "protocol": {
            "states": len(dataset.expert_action),
            "episodes": int(len(np.unique(dataset.episode))),
            "folds": args.folds,
            "splitUnit": "episode",
            "brainProjection": (
                "deterministic label-free CountSketch to 8192 channels; history lags "
                "use distinct fixed hashes and equalized sqrt contributor scaling"
            ),
            "contextSlots": CONTEXT_SIZE,
            "totalInputWidth": TOTAL_INPUT_SIZE,
            "probe": f"{TOTAL_INPUT_SIZE} -> 256 GELU LayerNorm -> 5",
            "history": "current plus up to previous 3 decision brain states (800 ms span)",
            "efference": "previous 4 student actions, one-hot plus availability masks",
            "lateralSelfState": "current fly column / world half width and absolute magnitude",
            "sameExpertLabels": True,
            "sameEpisodeGroupedFoldsForEveryCondition": True,
            "sameCorrectionWeightingAsDaggerProbe": True,
        },
        "results": results,
        "interpretation": conclusion,
        "elapsedSeconds": time.perf_counter() - started,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== CONTEXT AUDIT SUMMARY ===", flush=True)
    for condition in conditions:
        result = results[condition]
        print(
            f"{condition:34s} "
            f"core={result['oof']['coreMacroAccuracy']:.3f} "
            f"disagree={result['disagreementRecovery']:.3f} "
            f"critical={result['criticalCorrection']:.3f}",
            flush=True,
        )

    print(f"\ndiagnosis: {conclusion['diagnosis']}", flush=True)
    print(f"report: {args.output.resolve()}", flush=True)

    if conclusion["strongContextEvidence"]:
        print("\nCONTEXT SUFFICIENCY AUDIT: CONTEXT BOTTLENECK SUPPORTED", flush=True)
    else:
        print("\nCONTEXT SUFFICIENCY AUDIT: CONTEXT BOTTLENECK NOT SUPPORTED", flush=True)
        print(
            "Do not add state/context channels to production based on this audit. "
            "The next structural suspect is visual sensor/retina sufficiency.",
            flush=True,
        )


if __name__ == "__main__":
    main()
