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
from .connectome import load_reduced_graph_variant
from .env import WORLD_VERSION
from .export import export_policy
from .models import SettledPopulationFixedGraphPolicy
from .schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE, OBSERVATION_VERSION
from .train import _build_controller_v2_graph


TRAINING_VERSION = "expert-imitation-settled-bptt-v2"
CORE_ACTIONS = ("forward", "left", "right", "wait")
ACTION_INDEX = {action.value: index for index, action in enumerate(ACTION_ORDER)}


def episode_slice(offsets: np.ndarray, episode: int) -> slice:
    return slice(int(offsets[episode]), int(offsets[episode + 1]))


def load_dataset(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    dataset_path = directory / "dataset.npz"
    if manifest["environmentVersion"] != WORLD_VERSION:
        raise ValueError("Dataset environment version mismatch.")
    if manifest["observationVersion"] != OBSERVATION_VERSION:
        raise ValueError("Dataset observation version mismatch.")
    if manifest["observationSize"] != OBSERVATION_INPUT_SIZE:
        raise ValueError("Dataset observation size mismatch.")
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != manifest["dataset"]["sha256"]:
        raise ValueError("Dataset hash mismatch.")
    with np.load(dataset_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    return manifest, arrays


def split_episodes(count: int, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(count * fraction))
    return sorted(indices[validation_count:]), sorted(indices[:validation_count])


def episode_start_lookup(offsets: np.ndarray) -> np.ndarray:
    lookup = np.empty(int(offsets[-1]), dtype=np.int64)
    for episode in range(len(offsets) - 1):
        start = int(offsets[episode])
        end = int(offsets[episode + 1])
        lookup[start:end] = start
    return lookup


def training_pools(actions: np.ndarray, offsets: np.ndarray, episodes: list[int]) -> dict[str, np.ndarray]:
    allowed = np.zeros(len(actions), dtype=bool)
    for episode in episodes:
        allowed[episode_slice(offsets, episode)] = True
    pools = {}
    for name in CORE_ACTIONS:
        pool = np.flatnonzero(allowed & (actions == ACTION_INDEX[name])).astype(np.int64)
        if len(pool) == 0:
            raise RuntimeError(f"No examples for {name}.")
        pools[name] = pool
    return pools


def sample_anchors(pools: dict[str, np.ndarray], per_class: int, rng: np.random.Generator) -> np.ndarray:
    pieces = []
    for name in CORE_ACTIONS:
        pool = pools[name]
        pieces.append(rng.choice(pool, size=per_class, replace=len(pool) < per_class))
    anchors = np.concatenate(pieces)
    rng.shuffle(anchors)
    return anchors


def window_batch(
    observations: np.ndarray,
    actions: np.ndarray,
    episode_start: np.ndarray,
    anchors: np.ndarray,
    window: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    x = torch.zeros(window, len(anchors), observations.shape[1], dtype=torch.float32, device=device)
    active = torch.zeros(window, len(anchors), dtype=torch.bool, device=device)
    targets = torch.from_numpy(actions[anchors]).to(device=device, dtype=torch.long)
    for column, anchor in enumerate(anchors.tolist()):
        start = max(int(episode_start[anchor]), anchor - window + 1)
        segment = observations[start : anchor + 1]
        length = len(segment)
        x[window - length :, column] = torch.from_numpy(segment).to(device=device, dtype=torch.float32)
        active[window - length :, column] = True
    return x, targets, active


def train_batch(model, optimizer, observations: Tensor, targets: Tensor, active: Tensor) -> tuple[float, int]:
    hidden = torch.zeros(observations.shape[1], model.graph.node_count, device=observations.device)
    final_logits = None
    for timestep in range(observations.shape[0]):
        mask = active[timestep]
        logits, _, next_hidden = model(observations[timestep], hidden)
        hidden = torch.where(mask.unsqueeze(1), next_hidden, hidden)
        final_logits = logits
    if final_logits is None:
        raise RuntimeError("Empty imitation window.")
    loss = F.cross_entropy(final_logits, targets, label_smoothing=0.02)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    correct = int((torch.argmax(final_logits, dim=1) == targets).sum().item())
    return float(loss.detach().cpu()), correct


@torch.no_grad()
def evaluate(model, observations, actions, offsets, episodes, device) -> dict[str, Any]:
    model.eval()
    totals = np.zeros(len(ACTION_ORDER), dtype=np.int64)
    correct = np.zeros(len(ACTION_ORDER), dtype=np.int64)
    for episode in episodes:
        sl = episode_slice(offsets, episode)
        obs = torch.from_numpy(observations[sl]).to(device=device, dtype=torch.float32)
        labels = actions[sl]
        hidden = torch.zeros(1, model.graph.node_count, device=device)
        for timestep, target in enumerate(labels.tolist()):
            logits, _, hidden = model(obs[timestep:timestep + 1], hidden)
            prediction = int(torch.argmax(logits, dim=1).item())
            totals[target] += 1
            if prediction == target:
                correct[target] += 1
    per_action = {}
    supported = []
    core_supported = []
    for index, action in enumerate(ACTION_ORDER):
        accuracy = float(correct[index] / totals[index]) if totals[index] else None
        per_action[action.value] = accuracy
        if accuracy is not None:
            supported.append(accuracy)
            if action.value in CORE_ACTIONS:
                core_supported.append(accuracy)
    return {
        "accuracy": int(correct.sum()) / max(1, int(totals.sum())),
        "macroAccuracy": float(np.mean(supported)) if supported else 0.0,
        "coreMacroAccuracy": float(np.mean(core_supported)) if core_supported else 0.0,
        "perActionAccuracy": per_action,
        "support": {action.value: int(totals[index]) for index, action in enumerate(ACTION_ORDER)},
    }


def train(args) -> dict[str, Any]:
    manifest, arrays = load_dataset(args.dataset)
    observations = arrays["observations"].astype(np.float32, copy=False)
    actions = arrays["actions"].astype(np.int64, copy=False)
    offsets = arrays["episode_offsets"].astype(np.int64, copy=False)
    seeds = [str(value) for value in arrays["seeds"].tolist()]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_episodes, val_episodes = split_episodes(len(offsets) - 1, args.validation_fraction, args.seed)
    episode_start = episode_start_lookup(offsets)
    pools = training_pools(actions, offsets, train_episodes)

    graph = _build_controller_v2_graph(load_reduced_graph_variant("1k"))
    model = SettledPopulationFixedGraphPolicy(graph, OBSERVATION_INPUT_SIZE, len(ACTION_ORDER)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)

    best_state = None
    best_core_macro = -math.inf
    best_epoch = 0
    stale = 0
    curve = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        anchors = sample_anchors(pools, args.anchors_per_class, rng)
        losses = []
        train_correct = 0
        for start in range(0, len(anchors), args.batch_size):
            batch = anchors[start:start + args.batch_size]
            x, targets, active = window_batch(observations, actions, episode_start, batch, args.window, device)
            loss, correct = train_batch(model, optimizer, x, targets, active)
            losses.append(loss)
            train_correct += correct

        val = evaluate(model, observations, actions, offsets, val_episodes, device)
        row = {
            "epoch": epoch,
            "trainLoss": float(np.mean(losses)),
            "balancedTrainAccuracy": train_correct / len(anchors),
            "validationAccuracy": val["accuracy"],
            "validationMacroAccuracy": val["macroAccuracy"],
            "validationCoreMacroAccuracy": val["coreMacroAccuracy"],
            "validationPerActionAccuracy": val["perActionAccuracy"],
        }
        curve.append(row)
        print(
            f"[imitation-v2] epoch={epoch:02d} loss={row['trainLoss']:.4f} "
            f"balanced={row['balancedTrainAccuracy']:.3f} "
            f"val={row['validationAccuracy']:.3f} coreMacro={row['validationCoreMacroAccuracy']:.3f}",
            flush=True,
        )
        if val["coreMacroAccuracy"] > best_core_macro + 1e-6:
            best_core_macro = float(val["coreMacroAccuracy"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No best model was produced.")
    model.load_state_dict(best_state)
    model.to(device)
    training_metrics = evaluate(model, observations, actions, offsets, train_episodes, device)
    validation_metrics = evaluate(model, observations, actions, offsets, val_episodes, device)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "checkpoint.pt"
    checkpoint = {
        "format_version": 1,
        "environment_version": WORLD_VERSION,
        "controller": "connectome",
        "model": {
            "observation_size": OBSERVATION_INPUT_SIZE,
            "actions": len(ACTION_ORDER),
            "graph": graph.to_checkpoint(),
            "interface": "population-settled",
        },
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training": {
            "seed": str(manifest["seedPrefix"]),
            "steps": int(len(actions)),
            "envs": 1,
            "learning_rate": args.learning_rate,
            "world_seeds": seeds,
            "connectome_graph": "1k",
            "connectome_interface": "population-settled",
            "training_mode": TRAINING_VERSION,
            "internal_steps": 2,
            "dataset_sha256": manifest["dataset"]["sha256"],
            "planner_version": manifest["plannerVersion"],
            "planner_depth": manifest["plannerDepth"],
            "window": args.window,
            "anchors_per_class": args.anchors_per_class,
            "batch_size": args.batch_size,
            "best_epoch": best_epoch,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    validated = validate_checkpoint(checkpoint)
    if validated.connectome_interface != "population-settled":
        raise RuntimeError("Checkpoint lost population-settled interface.")

    policy_path = args.output / "policy.json"
    export_policy(checkpoint_path, policy_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy["network"].get("interfaceMode") != "population-settled":
        raise RuntimeError("Exported policy lost settled interface.")
    if policy["network"].get("internalSteps") != 2:
        raise RuntimeError("Exported policy lost internalSteps=2.")
    if "riskWeights" in policy["network"] or "routeWeights" in policy["network"]:
        raise RuntimeError("Controller V2 auxiliary heads leaked into settled imitation policy.")

    metrics = {
        "version": 2,
        "trainingMode": TRAINING_VERSION,
        "bestEpoch": best_epoch,
        "epochsCompleted": len(curve),
        "trainingCurve": curve,
        "training": training_metrics,
        "validation": validation_metrics,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "version": 2,
        "environmentVersion": WORLD_VERSION,
        "observationVersion": OBSERVATION_VERSION,
        "trainingMode": TRAINING_VERSION,
        "internalSteps": 2,
        "datasetSha256": manifest["dataset"]["sha256"],
        "plannerVersion": manifest["plannerVersion"],
        "plannerDepth": manifest["plannerDepth"],
        "sensoryCells": len(graph.sensory_body_ids),
        "readoutCells": len(graph.readout_body_ids),
        "balancedTrainingPool": {key: int(len(value)) for key, value in pools.items()},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return {
        "checkpoint": str(checkpoint_path),
        "policy": str(policy_path),
        "metrics": str(args.output / "metrics.json"),
        "metadata": str(args.output / "metadata.json"),
        "bestEpoch": best_epoch,
        "validationAccuracy": validation_metrics["accuracy"],
        "validationMacroAccuracy": validation_metrics["macroAccuracy"],
        "validationCoreMacroAccuracy": validation_metrics["coreMacroAccuracy"],
        "validationPerActionAccuracy": validation_metrics["perActionAccuracy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train settled 1k MaleCNS by balanced expert imitation.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--anchors-per-class", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
