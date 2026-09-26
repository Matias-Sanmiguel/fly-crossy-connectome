from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from fly_crossy.schema import ACTION_ORDER

from .decoder import (
    DECODER_VERSION,
    EXPECTED_INPUT_SIZE,
    FEATURE_KIND,
    MaleCNSActionDecoder,
    save_decoder,
)
from .information_probe import (
    ProbeDataset,
    class_weights,
    feature_stats,
    metrics_from_predictions,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@torch.no_grad()
def evaluate(
    model: MaleCNSActionDecoder,
    dataset: ProbeDataset,
    mean: Tensor,
    std: Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    predictions: list[np.ndarray] = []

    for start in range(0, len(dataset.labels), batch_size):
        end = min(len(dataset.labels), start + batch_size)
        x = torch.as_tensor(
            dataset.features[start:end],
            dtype=torch.float32,
            device=device,
        )
        logits = model((x - mean) / std)
        predictions.append(
            logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        )

    predicted = np.concatenate(predictions)
    return metrics_from_predictions(dataset.labels, predicted)


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Train the first final MaleCNS Crossy V2 action decoder. "
            "Only the 256-unit decoder is trainable."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "information-probe" / "train-features.npz",
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "information-probe" / "validation-features.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "decoder-r0.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "malecns-crossy-v2-decoder-r0-training.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    if args.restarts < 1:
        raise SystemExit("--restarts must be positive.")

    train = ProbeDataset.load(args.train_dataset)
    validation = ProbeDataset.load(args.validation_dataset)

    if train.features.shape[1] != EXPECTED_INPUT_SIZE:
        raise SystemExit(
            f"Train features have {train.features.shape[1]} columns; "
            f"expected {EXPECTED_INPUT_SIZE}."
        )
    if validation.features.shape[1] != EXPECTED_INPUT_SIZE:
        raise SystemExit("Validation feature width does not match decoder contract.")
    if set(train.seeds) & set(validation.seeds):
        raise SystemExit("Train/validation seed sets overlap.")

    mean_np, std_np = feature_stats(
        train.features,
        np.arange(EXPECTED_INPUT_SIZE, dtype=np.int64),
    )
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)

    weight = torch.as_tensor(
        class_weights(train.labels),
        dtype=torch.float32,
        device=device,
    )

    print("=== MaleCNS Crossy V2 / Decoder R0 Training ===", flush=True)
    print(f"device: {device}", flush=True)
    print(
        f"train={len(train.labels):,} validation={len(validation.labels):,} "
        f"features={EXPECTED_INPUT_SIZE:,}",
        flush=True,
    )
    print(
        "train labels: "
        + str({
            ACTION_ORDER[i].value: int((train.labels == i).sum())
            for i in range(len(ACTION_ORDER))
        }),
        flush=True,
    )

    restart_reports: list[dict[str, object]] = []
    global_best: dict[str, object] | None = None
    global_best_state: dict[str, Tensor] | None = None
    global_best_score: tuple[float, float] = (-1.0, -1.0)

    started_all = time.perf_counter()

    for restart in range(args.restarts):
        restart_seed = args.seed + restart * 101
        torch.manual_seed(restart_seed)
        rng = np.random.default_rng(restart_seed)

        model = MaleCNSActionDecoder().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-4,
        )

        order = np.arange(len(train.labels), dtype=np.int64)
        best_state: dict[str, Tensor] | None = None
        best_score: tuple[float, float] = (-1.0, -1.0)
        best_epoch = 0
        stale = 0
        history: list[dict[str, float | int]] = []

        print(f"\n[restart {restart + 1}/{args.restarts}] seed={restart_seed}", flush=True)

        for epoch in range(1, args.epochs + 1):
            model.train()
            rng.shuffle(order)
            loss_sum = 0.0
            seen = 0

            for start in range(0, len(order), args.batch_size):
                rows = order[start:start + args.batch_size]
                x = torch.as_tensor(
                    train.features[rows],
                    dtype=torch.float32,
                    device=device,
                )
                y = torch.as_tensor(
                    train.labels[rows],
                    dtype=torch.long,
                    device=device,
                )

                optimizer.zero_grad(set_to_none=True)
                logits = model((x - mean) / std)
                loss = F.cross_entropy(
                    logits,
                    y,
                    weight=weight,
                    label_smoothing=args.label_smoothing,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

                loss_sum += float(loss) * len(rows)
                seen += len(rows)

            validation_metrics = evaluate(
                model,
                validation,
                mean,
                std,
                device=device,
                batch_size=args.batch_size,
            )
            score = (
                float(validation_metrics["coreMacroAccuracy"]),
                float(validation_metrics["accuracy"]),
            )
            history.append(
                {
                    "epoch": epoch,
                    "loss": loss_sum / max(1, seen),
                    "validationAccuracy": score[1],
                    "validationCoreMacroAccuracy": score[0],
                }
            )

            print(
                f"  epoch {epoch:02d}: loss={history[-1]['loss']:.4f} "
                f"acc={score[1]:.3f} coreMacro={score[0]:.3f}",
                flush=True,
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

            if stale >= args.patience:
                print(f"  early stop after {epoch} epochs", flush=True)
                break

        if best_state is None:
            raise RuntimeError("Decoder restart produced no checkpoint.")

        model.load_state_dict(best_state)
        model.to(device)

        train_metrics = evaluate(
            model,
            train,
            mean,
            std,
            device=device,
            batch_size=args.batch_size,
        )
        validation_metrics = evaluate(
            model,
            validation,
            mean,
            std,
            device=device,
            batch_size=args.batch_size,
        )

        report = {
            "restart": restart,
            "seed": restart_seed,
            "bestEpoch": best_epoch,
            "selectionScore": {
                "coreMacroAccuracy": best_score[0],
                "accuracy": best_score[1],
            },
            "train": train_metrics,
            "validation": validation_metrics,
            "history": history,
        }
        restart_reports.append(report)

        print(
            f"  BEST: epoch={best_epoch} "
            f"acc={validation_metrics['accuracy']:.3f} "
            f"coreMacro={validation_metrics['coreMacroAccuracy']:.3f}",
            flush=True,
        )
        print(
            f"  perAction={validation_metrics['perActionAccuracy']}",
            flush=True,
        )

        if best_score > global_best_score:
            global_best_score = best_score
            global_best = report
            global_best_state = copy.deepcopy(best_state)

    assert global_best is not None and global_best_state is not None

    final_model = MaleCNSActionDecoder().to(device)
    final_model.load_state_dict(global_best_state)
    final_model.eval()

    metadata = {
        "decoderVersion": DECODER_VERSION,
        "featureKind": FEATURE_KIND,
        "trainDataset": str(args.train_dataset.resolve()),
        "validationDataset": str(args.validation_dataset.resolve()),
        "selectedRestart": global_best["restart"],
        "selectedSeed": global_best["seed"],
        "selectedEpoch": global_best["bestEpoch"],
        "validation": global_best["validation"],
        "architecture": {
            "input": EXPECTED_INPUT_SIZE,
            "hidden": 256,
            "output": len(ACTION_ORDER),
            "activation": "GELU",
            "normalization": "LayerNorm",
        },
        "training": {
            "learningRate": args.learning_rate,
            "labelSmoothing": args.label_smoothing,
            "batchSize": args.batch_size,
            "classWeighting": "sqrt-inverse-frequency clipped [0.35, 6.0]",
            "restarts": args.restarts,
        },
    }

    save_decoder(
        args.checkpoint,
        model=final_model,
        mean=mean_np,
        std=std_np,
        metadata=metadata,
    )

    final_report = {
        "version": 1,
        "decoderVersion": DECODER_VERSION,
        "featureKind": FEATURE_KIND,
        "checkpoint": str(args.checkpoint.resolve()),
        "elapsedSeconds": time.perf_counter() - started_all,
        "selected": global_best,
        "restarts": restart_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(final_report, indent=2) + "\n", encoding="utf-8")

    selected_validation = global_best["validation"]
    print()
    print("DECODER R0 TRAINING: COMPLETE", flush=True)
    print(f"checkpoint: {args.checkpoint.resolve()}", flush=True)
    print(f"report:     {args.output.resolve()}", flush=True)
    print(
        f"validation acc={selected_validation['accuracy']:.3f} "
        f"coreMacro={selected_validation['coreMacroAccuracy']:.3f}",
        flush=True,
    )
    print(f"perAction={selected_validation['perActionAccuracy']}", flush=True)
    print(
        "\nDo not promote this decoder yet. Closed-loop evaluation is the next gate.",
        flush=True,
    )


if __name__ == "__main__":
    main()
