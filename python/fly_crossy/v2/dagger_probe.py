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

from .decoder import EXPECTED_INPUT_SIZE, MaleCNSActionDecoder
from .information_probe import (
    class_weights,
    feature_stats,
    metrics_from_predictions,
)


ACTION_NAMES = tuple(action.value for action in ACTION_ORDER)
CORE_ACTION_NAMES = ("forward", "left", "right", "wait")
TERMINAL_NAMES = ("none", "vehicle", "train", "water", "bounds")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class DaggerProbeDataset:
    features: np.ndarray
    expert_action: np.ndarray
    student_action: np.ndarray
    disagreement: np.ndarray
    critical_error: np.ndarray
    student_terminal: np.ndarray
    episode_terminal: np.ndarray
    steps_to_terminal: np.ndarray
    episode: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "DaggerProbeDataset":
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            required = (
                "features",
                "expert_action",
                "student_action",
                "disagreement",
                "critical_error",
                "student_terminal",
                "episode_terminal",
                "steps_to_terminal",
                "episode",
            )
            missing = [name for name in required if name not in z.files]
            if missing:
                raise ValueError(f"DAgger R1 dataset is missing arrays: {missing}")

            dataset = cls(
                features=z["features"].astype(np.float16, copy=True),
                expert_action=z["expert_action"].astype(np.uint8, copy=True),
                student_action=z["student_action"].astype(np.uint8, copy=True),
                disagreement=z["disagreement"].astype(np.bool_, copy=True),
                critical_error=z["critical_error"].astype(np.bool_, copy=True),
                student_terminal=z["student_terminal"].astype(np.uint8, copy=True),
                episode_terminal=z["episode_terminal"].astype(np.uint8, copy=True),
                steps_to_terminal=z["steps_to_terminal"].astype(np.int16, copy=True),
                episode=z["episode"].astype(np.int16, copy=True),
            )
        dataset.validate()
        return dataset

    def validate(self) -> None:
        n = len(self.expert_action)
        if self.features.shape != (n, EXPECTED_INPUT_SIZE):
            raise ValueError(
                f"Expected features ({n}, {EXPECTED_INPUT_SIZE}), got {self.features.shape}."
            )
        for name in (
            "student_action",
            "disagreement",
            "critical_error",
            "student_terminal",
            "episode_terminal",
            "steps_to_terminal",
            "episode",
        ):
            value = getattr(self, name)
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {value.shape}.")
        if not np.isfinite(self.features).all():
            raise ValueError("DAgger features contain non-finite values.")
        if np.any(self.expert_action >= len(ACTION_ORDER)):
            raise ValueError("expert_action contains an invalid action.")
        if np.any(self.student_action >= len(ACTION_ORDER)):
            raise ValueError("student_action contains an invalid action.")


def make_episode_folds(
    episodes: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic train/validation row indices split strictly by episode."""
    unique = np.unique(episodes)
    if folds < 2 or folds > len(unique):
        raise ValueError("folds must be between 2 and the number of episodes.")

    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    validation_groups = np.array_split(shuffled, folds)

    result: list[tuple[np.ndarray, np.ndarray]] = []
    for group in validation_groups:
        validation_mask = np.isin(episodes, group)
        train_rows = np.flatnonzero(~validation_mask)
        validation_rows = np.flatnonzero(validation_mask)

        if len(set(episodes[train_rows].tolist()) & set(episodes[validation_rows].tolist())):
            raise RuntimeError("Episode leakage detected in DAgger probe fold.")
        result.append((train_rows, validation_rows))

    # Every row must be validation exactly once across all folds.
    coverage = np.zeros(len(episodes), dtype=np.int16)
    for _, validation_rows in result:
        coverage[validation_rows] += 1
    if not np.all(coverage == 1):
        raise RuntimeError("Episode-fold validation coverage is not exactly once per row.")

    return result


def sample_weights(
    labels: np.ndarray,
    disagreement: np.ndarray,
    critical_error: np.ndarray,
) -> np.ndarray:
    """Correction-focused but bounded weights for the diagnostic probe.

    This is intentionally not a new architecture. It asks whether the same
    21,022 -> 256 -> 5 readout can learn the expert correction when student
    states are emphasized.
    """
    base_class = class_weights(labels).astype(np.float32)
    weights = base_class[labels].astype(np.float32, copy=True)
    weights *= np.where(disagreement, 2.0, 1.0).astype(np.float32)
    weights *= np.where(critical_error, 2.0, 1.0).astype(np.float32)
    # Prevent a handful of rare samples from dominating an entire minibatch.
    return np.clip(weights, 0.25, 12.0)


@torch.no_grad()
def _predict(
    model: MaleCNSActionDecoder,
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

    return (
        np.concatenate(parts)
        if parts
        else np.zeros(0, dtype=np.uint8)
    )


def _action_recovery(
    labels: np.ndarray,
    predicted: np.ndarray,
    disagreement: np.ndarray,
) -> dict[str, object]:
    result = {}
    for index, name in enumerate(ACTION_NAMES):
        mask = disagreement & (labels == index)
        support = int(mask.sum())
        result[name] = {
            "support": support,
            "accuracy": (
                float((predicted[mask] == labels[mask]).mean())
                if support
                else None
            ),
        }
    return result


def _critical_by_reason(
    predicted: np.ndarray,
    dataset: DaggerProbeDataset,
) -> dict[str, object]:
    result = {}
    for terminal_index, name in enumerate(TERMINAL_NAMES[1:], start=1):
        mask = dataset.critical_error & (dataset.student_terminal == terminal_index)
        support = int(mask.sum())
        result[name] = {
            "support": support,
            "correctionRate": (
                float((predicted[mask] == dataset.expert_action[mask]).mean())
                if support
                else None
            ),
        }
    return result


def _predeath_accuracy(
    predicted: np.ndarray,
    dataset: DaggerProbeDataset,
) -> dict[str, object]:
    result = {}
    for terminal_index, terminal_name in enumerate(TERMINAL_NAMES[1:], start=1):
        reason = {}
        for window in (1, 3, 5):
            mask = (
                (dataset.episode_terminal == terminal_index)
                & (dataset.steps_to_terminal >= 0)
                & (dataset.steps_to_terminal < window)
            )
            support = int(mask.sum())
            reason[str(window)] = {
                "support": support,
                "expertActionAccuracy": (
                    float((predicted[mask] == dataset.expert_action[mask]).mean())
                    if support
                    else None
                ),
            }
        result[terminal_name] = reason
    return result


def evaluate_oof(
    predicted: np.ndarray,
    dataset: DaggerProbeDataset,
) -> dict[str, object]:
    if predicted.shape != dataset.expert_action.shape:
        raise ValueError("OOF prediction shape mismatch.")

    all_metrics = metrics_from_predictions(dataset.expert_action, predicted)

    disagreement_mask = dataset.disagreement
    critical_mask = dataset.critical_error

    disagreement_accuracy = (
        float(
            (
                predicted[disagreement_mask]
                == dataset.expert_action[disagreement_mask]
            ).mean()
        )
        if disagreement_mask.any()
        else 0.0
    )
    critical_correction_rate = (
        float(
            (
                predicted[critical_mask]
                == dataset.expert_action[critical_mask]
            ).mean()
        )
        if critical_mask.any()
        else 0.0
    )

    baseline_student = metrics_from_predictions(
        dataset.expert_action,
        dataset.student_action,
    )

    return {
        "allStates": all_metrics,
        "baselineR0OnSameStates": baseline_student,
        "coreMacroImprovementVsR0": (
            float(all_metrics["coreMacroAccuracy"])
            - float(baseline_student["coreMacroAccuracy"])
        ),
        "disagreementStates": {
            "support": int(disagreement_mask.sum()),
            "expertActionAccuracy": disagreement_accuracy,
            "byExpertAction": _action_recovery(
                dataset.expert_action,
                predicted,
                disagreement_mask,
            ),
        },
        "criticalErrors": {
            "support": int(critical_mask.sum()),
            "correctionRate": critical_correction_rate,
            "byDeathReason": _critical_by_reason(predicted, dataset),
        },
        "preDeathWindows": _predeath_accuracy(predicted, dataset),
    }


def train_fold(
    dataset: DaggerProbeDataset,
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

    indices = np.arange(EXPECTED_INPUT_SIZE, dtype=np.int64)
    mean_np, std_np = feature_stats(dataset.features[train_rows], indices)
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)

    model = MaleCNSActionDecoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    train_labels = dataset.expert_action[train_rows]
    per_sample = sample_weights(
        train_labels,
        dataset.disagreement[train_rows],
        dataset.critical_error[train_rows],
    )

    order = np.arange(len(train_rows), dtype=np.int64)
    best_state: dict[str, Tensor] | None = None
    best_score = (-1.0, -1.0)
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)
        loss_sum = 0.0
        seen_weight = 0.0

        for start in range(0, len(order), batch_size):
            local = order[start:start + batch_size]
            rows = train_rows[local]

            x = torch.as_tensor(
                dataset.features[rows],
                dtype=torch.float32,
                device=device,
            )
            y = torch.as_tensor(
                dataset.expert_action[rows],
                dtype=torch.long,
                device=device,
            )
            w = torch.as_tensor(
                per_sample[local],
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

            loss_sum += float((loss_each.detach() * w).sum())
            seen_weight += float(w.sum())

        validation_pred = _predict(
            model,
            dataset.features,
            validation_rows,
            mean,
            std,
            device=device,
            batch_size=batch_size,
        )
        validation_metrics = metrics_from_predictions(
            dataset.expert_action[validation_rows],
            validation_pred,
        )
        disagreement_mask = dataset.disagreement[validation_rows]
        disagreement_accuracy = (
            float(
                (
                    validation_pred[disagreement_mask]
                    == dataset.expert_action[validation_rows][disagreement_mask]
                ).mean()
            )
            if disagreement_mask.any()
            else 0.0
        )

        score = (
            float(validation_metrics["coreMacroAccuracy"]),
            disagreement_accuracy,
        )
        history.append(
            {
                "epoch": epoch,
                "trainWeightedLoss": loss_sum / max(seen_weight, 1e-9),
                "validationAccuracy": validation_metrics["accuracy"],
                "validationCoreMacro": validation_metrics["coreMacroAccuracy"],
                "validationDisagreementAccuracy": disagreement_accuracy,
            }
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
        raise RuntimeError("DAgger learnability fold produced no checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    predicted = _predict(
        model,
        dataset.features,
        validation_rows,
        mean,
        std,
        device=device,
        batch_size=batch_size,
    )

    fold_metrics = metrics_from_predictions(
        dataset.expert_action[validation_rows],
        predicted,
    )
    disagreement_mask = dataset.disagreement[validation_rows]
    critical_mask = dataset.critical_error[validation_rows]

    return predicted, {
        "seed": seed,
        "trainEpisodes": int(len(np.unique(dataset.episode[train_rows]))),
        "validationEpisodes": int(len(np.unique(dataset.episode[validation_rows]))),
        "trainSamples": int(len(train_rows)),
        "validationSamples": int(len(validation_rows)),
        "bestEpoch": best_epoch,
        "validation": fold_metrics,
        "validationDisagreementSupport": int(disagreement_mask.sum()),
        "validationDisagreementAccuracy": (
            float(
                (
                    predicted[disagreement_mask]
                    == dataset.expert_action[validation_rows][disagreement_mask]
                ).mean()
            )
            if disagreement_mask.any()
            else None
        ),
        "validationCriticalSupport": int(critical_mask.sum()),
        "validationCriticalCorrectionRate": (
            float(
                (
                    predicted[critical_mask]
                    == dataset.expert_action[validation_rows][critical_mask]
                ).mean()
            )
            if critical_mask.any()
            else None
        ),
        "history": history,
    }


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "One-shot DAgger-state learnability gate for MaleCNS Crossy V2. "
            "Five episode-grouped folds; same 21022->256->5 decoder architecture; "
            "no production model is modified."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "dagger-r1-student-states.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "malecns-crossy-v2-dagger-r1-learnability.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    dataset = DaggerProbeDataset.load(args.dataset)
    folds = make_episode_folds(
        dataset.episode,
        folds=args.folds,
        seed=args.seed,
    )

    print("=== MaleCNS Crossy V2 / DAgger R1 Learnability Probe ===", flush=True)
    print("Diagnostic only. Decoder R0 is NOT modified.", flush=True)
    print(
        f"samples={len(dataset.expert_action):,} "
        f"episodes={len(np.unique(dataset.episode))} "
        f"disagreements={int(dataset.disagreement.sum()):,} "
        f"critical={int(dataset.critical_error.sum()):,}",
        flush=True,
    )
    print(
        f"architecture={EXPECTED_INPUT_SIZE:,} -> 256 -> {len(ACTION_ORDER)}",
        flush=True,
    )

    oof = np.full(len(dataset.expert_action), 255, dtype=np.uint8)
    fold_reports: list[dict[str, object]] = []
    started = time.perf_counter()

    for fold_index, (train_rows, validation_rows) in enumerate(folds):
        print(
            f"\n[fold {fold_index + 1}/{len(folds)}] "
            f"train={len(train_rows):,} val={len(validation_rows):,}",
            flush=True,
        )
        predicted, report = train_fold(
            dataset,
            train_rows,
            validation_rows,
            device=device,
            seed=args.seed + fold_index * 101,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        oof[validation_rows] = predicted
        report["fold"] = fold_index
        fold_reports.append(report)

        vm = report["validation"]
        print(
            f"  bestEpoch={report['bestEpoch']} "
            f"acc={vm['accuracy']:.3f} "
            f"coreMacro={vm['coreMacroAccuracy']:.3f} "
            f"disagreeAcc={report['validationDisagreementAccuracy']} "
            f"criticalFix={report['validationCriticalCorrectionRate']}",
            flush=True,
        )

    if np.any(oof == 255):
        raise RuntimeError("OOF predictions did not cover every sample.")

    evaluation = evaluate_oof(oof, dataset)
    all_metrics = evaluation["allStates"]
    per_action = all_metrics["perActionAccuracy"]

    gates = {
        "episodeGroupedNoLeakage": True,
        "oofCoreMacroAtLeast060": (
            float(all_metrics["coreMacroAccuracy"]) >= 0.60
        ),
        "oofLeftAtLeast050": (
            per_action["left"] is not None
            and float(per_action["left"]) >= 0.50
        ),
        "oofRightAtLeast050": (
            per_action["right"] is not None
            and float(per_action["right"]) >= 0.50
        ),
        "oofWaitAtLeast040": (
            per_action["wait"] is not None
            and float(per_action["wait"]) >= 0.40
        ),
        "disagreementRecoveryAtLeast050": (
            float(
                evaluation["disagreementStates"]["expertActionAccuracy"]
            ) >= 0.50
        ),
        "criticalCorrectionAtLeast050": (
            float(
                evaluation["criticalErrors"]["correctionRate"]
            ) >= 0.50
        ),
        "coreMacroImprovesR0ByAtLeast005": (
            float(evaluation["coreMacroImprovementVsR0"]) >= 0.05
        ),
    }

    report = {
        "version": 1,
        "purpose": (
            "One-shot representation gate before the only allowed DAgger fine-tune. "
            "The production R0 decoder is untouched. Five grouped folds test whether "
            "the same 21022->256->5 architecture can learn expert corrections on "
            "student-visited states."
        ),
        "dataset": str(args.dataset.resolve()),
        "protocol": {
            "folds": args.folds,
            "splitUnit": "episode",
            "epochsMaximum": args.epochs,
            "patience": args.patience,
            "learningRate": args.learning_rate,
            "batchSize": args.batch_size,
            "lossWeighting": (
                "sqrt inverse expert-action frequency; x2 student disagreement; "
                "additional x2 immediately-fatal critical error; clipped [0.25,12]"
            ),
            "architecture": "21022 -> 256 GELU LayerNorm -> 5",
            "productionWeightsChanged": False,
        },
        "elapsedSeconds": time.perf_counter() - started,
        "folds": fold_reports,
        "outOfFold": evaluation,
        "gates": gates,
        "decisionRule": (
            "PASS => representation/readout is learnable on student states; consume the "
            "single bounded DAgger fine-tune round. REVIEW => do not fine-tune repeatedly; "
            "the bottleneck remains representation/readout."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n=== OOF SUMMARY ===", flush=True)
    print(
        f"accuracy={all_metrics['accuracy']:.3f} "
        f"coreMacro={all_metrics['coreMacroAccuracy']:.3f}",
        flush=True,
    )
    print(f"perAction={per_action}", flush=True)
    print(
        f"R0 coreMacro on same states="
        f"{evaluation['baselineR0OnSameStates']['coreMacroAccuracy']:.3f}",
        flush=True,
    )
    print(
        f"coreMacro improvement="
        f"{evaluation['coreMacroImprovementVsR0']:+.3f}",
        flush=True,
    )
    print(
        f"disagreement recovery="
        f"{evaluation['disagreementStates']['expertActionAccuracy']:.3f} "
        f"({evaluation['disagreementStates']['support']} states)",
        flush=True,
    )
    print(
        f"critical correction="
        f"{evaluation['criticalErrors']['correctionRate']:.3f} "
        f"({evaluation['criticalErrors']['support']} states)",
        flush=True,
    )
    print(f"report: {args.output.resolve()}", flush=True)

    if all(gates.values()):
        print("\nDAGGER R1 LEARNABILITY GATE: PASS", flush=True)
        print(
            "The frozen MaleCNS readout contains learnable expert-correction "
            "information on R0 student states.",
            flush=True,
        )
    else:
        print("\nDAGGER R1 LEARNABILITY GATE: REVIEW", flush=True)
        print(json.dumps(gates, indent=2), flush=True)
        print(
            "Do not consume the production DAgger fine-tune round yet.",
            flush=True,
        )


if __name__ == "__main__":
    main()
