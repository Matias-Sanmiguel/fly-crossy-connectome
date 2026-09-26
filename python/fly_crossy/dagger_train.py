from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .checkpoint import validate_checkpoint
from .env import WORLD_VERSION
from .export import export_policy
from .models import SettledPopulationFixedGraphPolicy
from .schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE, OBSERVATION_VERSION


TRAINING_VERSION = "dagger-hybrid-replay-v1"
MIN_BALANCED_SUPPORT = 64


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_dataset(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = directory / "manifest.json"
    dataset_path = directory / "dataset.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest["environmentVersion"] != WORLD_VERSION:
        raise ValueError("DAgger dataset environment version mismatch.")
    if manifest["observationVersion"] != OBSERVATION_VERSION:
        raise ValueError("DAgger dataset observation version mismatch.")
    if manifest["observationSize"] != OBSERVATION_INPUT_SIZE:
        raise ValueError("DAgger dataset observation width mismatch.")
    if manifest["actionOrder"] != [action.value for action in ACTION_ORDER]:
        raise ValueError("DAgger dataset action order mismatch.")

    expected_hash = manifest["dataset"]["sha256"]
    actual_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("DAgger dataset hash does not match manifest.")

    with np.load(dataset_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}

    required = {
        "observations",
        "actions",
        "episode_offsets",
        "seeds",
        "source",
        "disagreement",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"DAgger dataset is missing arrays: {sorted(missing)}")

    observations = arrays["observations"]
    actions = arrays["actions"]
    offsets = arrays["episode_offsets"]

    if observations.ndim != 2 or observations.shape[1] != OBSERVATION_INPUT_SIZE:
        raise ValueError("DAgger observations have an incompatible shape.")
    if actions.shape != (len(observations),):
        raise ValueError("DAgger actions have an incompatible shape.")
    if arrays["source"].shape != actions.shape:
        raise ValueError("DAgger source array has an incompatible shape.")
    if arrays["disagreement"].shape != actions.shape:
        raise ValueError("DAgger disagreement array has an incompatible shape.")
    if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(actions):
        raise ValueError("DAgger episode offsets are invalid.")
    if len(arrays["seeds"]) != len(offsets) - 1:
        raise ValueError("DAgger seed count does not match episode offsets.")

    return manifest, arrays


def _split_episodes(
    episode_count: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if episode_count < 2:
        raise ValueError("DAgger fine-tuning requires at least two episodes.")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")

    indices = list(range(episode_count))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(episode_count * validation_fraction))
    return sorted(indices[validation_count:]), sorted(indices[:validation_count])


def _episode_slice(offsets: np.ndarray, episode: int) -> slice:
    return slice(int(offsets[episode]), int(offsets[episode + 1]))


def _episode_start_lookup(offsets: np.ndarray) -> np.ndarray:
    lookup = np.empty(int(offsets[-1]), dtype=np.int64)
    for episode in range(len(offsets) - 1):
        start = int(offsets[episode])
        end = int(offsets[episode + 1])
        lookup[start:end] = start
    return lookup


def _indices_for_episodes(
    offsets: np.ndarray,
    episodes: list[int],
) -> np.ndarray:
    pieces = [
        np.arange(
            int(offsets[episode]),
            int(offsets[episode + 1]),
            dtype=np.int64,
        )
        for episode in episodes
    ]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def _action_pools(
    actions: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    pools: dict[str, np.ndarray] = {}
    for action_index, action in enumerate(ACTION_ORDER):
        pool = indices[actions[indices] == action_index]
        if len(pool) >= MIN_BALANCED_SUPPORT:
            pools[action.value] = pool
    if len(pools) < 2:
        raise RuntimeError("Not enough supported actions for DAgger balanced replay.")
    return pools


def _sample_hybrid_anchors(
    *,
    train_indices: np.ndarray,
    disagreement_indices: np.ndarray,
    action_pools: dict[str, np.ndarray],
    count: int,
    rng: np.random.Generator,
    natural_fraction: float,
    disagreement_fraction: float,
) -> np.ndarray:
    if count < 1:
        raise ValueError("anchor count must be positive.")
    if natural_fraction < 0 or disagreement_fraction < 0:
        raise ValueError("sampling fractions must be non-negative.")
    if natural_fraction + disagreement_fraction > 1:
        raise ValueError("natural + disagreement fractions cannot exceed 1.")

    natural_count = round(count * natural_fraction)
    disagreement_count = round(count * disagreement_fraction)
    balanced_count = count - natural_count - disagreement_count

    pieces: list[np.ndarray] = []

    if natural_count:
        pieces.append(
            rng.choice(
                train_indices,
                size=natural_count,
                replace=len(train_indices) < natural_count,
            )
        )

    if disagreement_count:
        pool = disagreement_indices
        if len(pool) == 0:
            pool = train_indices
        pieces.append(
            rng.choice(
                pool,
                size=disagreement_count,
                replace=len(pool) < disagreement_count,
            )
        )

    if balanced_count:
        names = list(action_pools)
        base = balanced_count // len(names)
        remainder = balanced_count % len(names)
        for index, name in enumerate(names):
            size = base + (1 if index < remainder else 0)
            if size == 0:
                continue
            pool = action_pools[name]
            pieces.append(
                rng.choice(
                    pool,
                    size=size,
                    replace=len(pool) < size,
                )
            )

    anchors = np.concatenate(pieces).astype(np.int64, copy=False)
    rng.shuffle(anchors)
    return anchors


def _window_batch(
    observations: np.ndarray,
    actions: np.ndarray,
    episode_start: np.ndarray,
    anchors: np.ndarray,
    *,
    window: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    batch = len(anchors)
    x = torch.zeros(
        window,
        batch,
        observations.shape[1],
        dtype=torch.float32,
        device=device,
    )
    active = torch.zeros(
        window,
        batch,
        dtype=torch.bool,
        device=device,
    )
    targets = torch.from_numpy(actions[anchors]).to(
        device=device,
        dtype=torch.long,
    )

    for column, anchor in enumerate(anchors.tolist()):
        start = max(
            int(episode_start[anchor]),
            anchor - window + 1,
        )
        segment = observations[start : anchor + 1]
        length = len(segment)
        x[window - length :, column] = torch.from_numpy(segment).to(
            device=device,
            dtype=torch.float32,
        )
        active[window - length :, column] = True

    return x, targets, active


def _train_batch(
    model: SettledPopulationFixedGraphPolicy,
    optimizer: torch.optim.Optimizer,
    observations: Tensor,
    targets: Tensor,
    active: Tensor,
) -> tuple[float, int]:
    model.train()
    hidden = torch.zeros(
        observations.shape[1],
        model.graph.node_count,
        dtype=torch.float32,
        device=observations.device,
    )
    final_logits: Tensor | None = None

    for timestep in range(observations.shape[0]):
        mask = active[timestep]
        logits, _, next_hidden = model(observations[timestep], hidden)
        hidden = torch.where(mask.unsqueeze(1), next_hidden, hidden)
        final_logits = logits

    if final_logits is None:
        raise RuntimeError("Empty DAgger recurrent window.")

    loss = F.cross_entropy(
        final_logits,
        targets,
        label_smoothing=0.01,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    correct = int(
        (torch.argmax(final_logits, dim=1) == targets)
        .sum()
        .item()
    )
    return float(loss.detach().cpu()), correct


def _empty_metric_state() -> dict[str, np.ndarray | int]:
    return {
        "total": np.zeros(len(ACTION_ORDER), dtype=np.int64),
        "correct": np.zeros(len(ACTION_ORDER), dtype=np.int64),
        "predicted": np.zeros(len(ACTION_ORDER), dtype=np.int64),
        "samples": 0,
    }


def _update_metric_state(
    state: dict[str, np.ndarray | int],
    target: int,
    prediction: int,
) -> None:
    total = state["total"]
    correct = state["correct"]
    predicted = state["predicted"]
    assert isinstance(total, np.ndarray)
    assert isinstance(correct, np.ndarray)
    assert isinstance(predicted, np.ndarray)
    total[target] += 1
    predicted[prediction] += 1
    if target == prediction:
        correct[target] += 1
    state["samples"] = int(state["samples"]) + 1


def _finalize_metric_state(
    state: dict[str, np.ndarray | int],
) -> dict[str, Any]:
    total = state["total"]
    correct = state["correct"]
    predicted = state["predicted"]
    assert isinstance(total, np.ndarray)
    assert isinstance(correct, np.ndarray)
    assert isinstance(predicted, np.ndarray)

    per_action: dict[str, float | None] = {}
    supported: list[float] = []
    for index, action in enumerate(ACTION_ORDER):
        if total[index] == 0:
            per_action[action.value] = None
        else:
            accuracy = float(correct[index] / total[index])
            per_action[action.value] = accuracy
            supported.append(accuracy)

    samples = int(state["samples"])
    return {
        "samples": samples,
        "accuracy": int(correct.sum()) / max(1, samples),
        "macroAccuracy": float(np.mean(supported)) if supported else 0.0,
        "perActionAccuracy": per_action,
        "targetActionCounts": {
            action.value: int(total[index])
            for index, action in enumerate(ACTION_ORDER)
        },
        "predictedActionCounts": {
            action.value: int(predicted[index])
            for index, action in enumerate(ACTION_ORDER)
        },
    }


@torch.no_grad()
def _evaluate(
    model: SettledPopulationFixedGraphPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    offsets: np.ndarray,
    source: np.ndarray,
    disagreement: np.ndarray,
    episodes: list[int],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    overall = _empty_metric_state()
    dagger = _empty_metric_state()
    prior_disagreement = _empty_metric_state()

    for episode in episodes:
        sl = _episode_slice(offsets, episode)
        episode_observations = torch.from_numpy(
            observations[sl]
        ).to(device=device, dtype=torch.float32)
        episode_actions = actions[sl]
        episode_source = source[sl]
        episode_disagreement = disagreement[sl]

        hidden = torch.zeros(
            1,
            model.graph.node_count,
            dtype=torch.float32,
            device=device,
        )

        for timestep, target in enumerate(episode_actions.tolist()):
            logits, _, hidden = model(
                episode_observations[timestep : timestep + 1],
                hidden,
            )
            prediction = int(torch.argmax(logits, dim=1).item())

            _update_metric_state(overall, target, prediction)
            if int(episode_source[timestep]) == 1:
                _update_metric_state(dagger, target, prediction)
            if int(episode_disagreement[timestep]) == 1:
                _update_metric_state(prior_disagreement, target, prediction)

    return {
        "overall": _finalize_metric_state(overall),
        "dagger": _finalize_metric_state(dagger),
        "previousStudentDisagreementStates": _finalize_metric_state(
            prior_disagreement
        ),
    }


def _load_initial_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[SettledPopulationFixedGraphPolicy, Any]:
    saved = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = validate_checkpoint(
        saved,
        expected_controller="connectome",
        expected_environment_version=WORLD_VERSION,
        expected_observation_size=OBSERVATION_INPUT_SIZE,
    )
    if checkpoint.connectome_interface != "population-settled":
        raise ValueError("Initial checkpoint must use population-settled.")
    if checkpoint.graph is None:
        raise ValueError("Initial checkpoint graph is missing.")

    model = SettledPopulationFixedGraphPolicy(
        checkpoint.graph,
        checkpoint.observation_size,
        checkpoint.actions,
    ).to(device)
    model.load_state_dict(checkpoint.state_dict)
    return model, checkpoint


def train_dagger(
    *,
    dataset: Path,
    init_checkpoint: Path,
    output: Path,
    epochs: int,
    window: int,
    anchors_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    seed: int,
    patience: int,
    natural_fraction: float,
    disagreement_fraction: float,
    device_name: str,
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if window < 1:
        raise ValueError("window must be positive.")
    if anchors_per_epoch < 1:
        raise ValueError("anchors_per_epoch must be positive.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if patience < 1:
        raise ValueError("patience must be positive.")

    manifest, arrays = _load_dataset(dataset)
    observations = arrays["observations"].astype(np.float32, copy=False)
    actions = arrays["actions"].astype(np.int64, copy=False)
    offsets = arrays["episode_offsets"].astype(np.int64, copy=False)
    source = arrays["source"].astype(np.uint8, copy=False)
    disagreement = arrays["disagreement"].astype(np.uint8, copy=False)
    seeds = [str(value) for value in arrays["seeds"].tolist()]

    device = _resolve_device(device_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_episodes, validation_episodes = _split_episodes(
        len(offsets) - 1,
        validation_fraction,
        seed,
    )
    train_indices = _indices_for_episodes(offsets, train_episodes)
    disagreement_indices = train_indices[
        disagreement[train_indices] == 1
    ]
    action_pools = _action_pools(actions, train_indices)
    episode_start = _episode_start_lookup(offsets)

    model, initial = _load_initial_model(init_checkpoint, device)

    dataset_graph_hash = manifest.get("aggregation", {}).get(
        "studentCheckpointSha256"
    )
    initial_hash = hashlib.sha256(init_checkpoint.read_bytes()).hexdigest()
    if dataset_graph_hash is not None and dataset_graph_hash != initial_hash:
        raise ValueError(
            "DAgger dataset was collected with a different student checkpoint."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    rng = np.random.default_rng(seed)

    output.mkdir(parents=True, exist_ok=True)
    curve: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0

    initial_validation = _evaluate(
        model,
        observations,
        actions,
        offsets,
        source,
        disagreement,
        validation_episodes,
        device,
    )
    print(
        "[dagger-train] initial "
        f"val={initial_validation['overall']['accuracy']:.3f} "
        f"macro={initial_validation['overall']['macroAccuracy']:.3f} "
        f"old-errors={initial_validation['previousStudentDisagreementStates']['accuracy']:.3f}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        anchors = _sample_hybrid_anchors(
            train_indices=train_indices,
            disagreement_indices=disagreement_indices,
            action_pools=action_pools,
            count=anchors_per_epoch,
            rng=rng,
            natural_fraction=natural_fraction,
            disagreement_fraction=disagreement_fraction,
        )

        losses: list[float] = []
        train_correct = 0
        for start in range(0, len(anchors), batch_size):
            batch_anchors = anchors[start : start + batch_size]
            x, targets, active = _window_batch(
                observations,
                actions,
                episode_start,
                batch_anchors,
                window=window,
                device=device,
            )
            loss, correct = _train_batch(
                model,
                optimizer,
                x,
                targets,
                active,
            )
            losses.append(loss)
            train_correct += correct

        validation = _evaluate(
            model,
            observations,
            actions,
            offsets,
            source,
            disagreement,
            validation_episodes,
            device,
        )

        overall = validation["overall"]
        old_errors = validation["previousStudentDisagreementStates"]

        # Preserve natural calibration while still rewarding recovery on the
        # states that the previous student actually got wrong.
        selection_score = (
            0.45 * float(overall["accuracy"])
            + 0.35 * float(overall["macroAccuracy"])
            + 0.20 * float(old_errors["accuracy"])
        )

        row = {
            "epoch": epoch,
            "trainLoss": float(np.mean(losses)),
            "sampledTrainAccuracy": train_correct / len(anchors),
            "selectionScore": selection_score,
            "validation": validation,
            "recurrentGain": float(model.recurrent_gain.detach().cpu()),
            "timeConstant": float(model.time_constant.detach().cpu()),
        }
        curve.append(row)

        print(
            f"[dagger-train] epoch={epoch:02d} "
            f"loss={row['trainLoss']:.4f} "
            f"train={row['sampledTrainAccuracy']:.3f} "
            f"val={overall['accuracy']:.3f} "
            f"macro={overall['macroAccuracy']:.3f} "
            f"old-errors={old_errors['accuracy']:.3f} "
            f"score={selection_score:.3f}",
            flush=True,
        )

        if selection_score > best_score + 1e-6:
            best_score = selection_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("DAgger fine-tuning failed to produce a model.")

    model.load_state_dict(best_state)
    model.to(device)

    final_validation = _evaluate(
        model,
        observations,
        actions,
        offsets,
        source,
        disagreement,
        validation_episodes,
        device,
    )

    checkpoint_path = output / "checkpoint.pt"
    root_seed = str(manifest["seedPrefix"])
    checkpoint = {
        "format_version": 1,
        "environment_version": WORLD_VERSION,
        "controller": "connectome",
        "model": {
            "observation_size": OBSERVATION_INPUT_SIZE,
            "actions": len(ACTION_ORDER),
            "graph": initial.graph.to_checkpoint(),
            "interface": "population-settled",
        },
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "training": {
            "seed": root_seed,
            "steps": int(len(actions)),
            "envs": 1,
            "learning_rate": learning_rate,
            "world_seeds": seeds,
            "connectome_graph": "1k",
            "connectome_interface": "population-settled",
            "training_mode": TRAINING_VERSION,
            "internal_steps": SettledPopulationFixedGraphPolicy.INTERNAL_STEPS,
            "optimizer_seed": seed,
            "dataset_sha256": manifest["dataset"]["sha256"],
            "init_checkpoint_sha256": initial_hash,
            "epochs_requested": epochs,
            "best_epoch": best_epoch,
            "window": window,
            "anchors_per_epoch": anchors_per_epoch,
            "batch_size": batch_size,
            "natural_fraction": natural_fraction,
            "disagreement_fraction": disagreement_fraction,
            "balanced_fraction": 1 - natural_fraction - disagreement_fraction,
            "balanced_actions": list(action_pools),
        },
    }
    torch.save(checkpoint, checkpoint_path)

    validated = validate_checkpoint(checkpoint)
    if validated.connectome_interface != "population-settled":
        raise RuntimeError("DAgger checkpoint lost settled interface metadata.")

    policy_path = output / "policy.json"
    export_policy(checkpoint_path, policy_path)

    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    metrics = {
        "version": 1,
        "trainingMode": TRAINING_VERSION,
        "bestEpoch": best_epoch,
        "bestSelectionScore": best_score,
        "initialValidation": initial_validation,
        "finalValidation": final_validation,
        "trainingCurve": curve,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "version": 1,
        "environmentVersion": WORLD_VERSION,
        "observationVersion": OBSERVATION_VERSION,
        "trainingMode": TRAINING_VERSION,
        "device": str(device),
        "dataset": str(dataset),
        "datasetSha256": manifest["dataset"]["sha256"],
        "initCheckpoint": str(init_checkpoint),
        "initCheckpointSha256": initial_hash,
        "balancedActionPools": {
            name: int(len(pool))
            for name, pool in action_pools.items()
        },
        "trainingSamples": int(len(train_indices)),
        "trainingDisagreementSamples": int(len(disagreement_indices)),
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": checkpoint_hash,
        },
        "policy": {"path": policy_path.name},
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "checkpoint": str(checkpoint_path),
        "policy": str(policy_path),
        "metrics": str(output / "metrics.json"),
        "metadata": str(output / "metadata.json"),
        "bestEpoch": best_epoch,
        "bestSelectionScore": best_score,
        "initialValidation": initial_validation,
        "finalValidation": final_validation,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the settled MaleCNS on aggregated DAgger data with "
            "hybrid natural, disagreement, and balanced replay."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--anchors-per-epoch", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--natural-fraction", type=float, default=0.50)
    parser.add_argument("--disagreement-fraction", type=float, default=0.25)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    result = train_dagger(
        dataset=args.dataset,
        init_checkpoint=args.init_checkpoint,
        output=args.output,
        epochs=args.epochs,
        window=args.window,
        anchors_per_epoch=args.anchors_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        patience=args.patience,
        natural_fraction=args.natural_fraction,
        disagreement_fraction=args.disagreement_fraction,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
