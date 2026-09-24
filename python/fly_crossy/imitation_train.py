from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
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
from .models import PopulationFixedGraphPolicy
from .schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE, OBSERVATION_VERSION
from .train import _build_controller_v2_graph


IMITATION_VERSION = "expert-imitation-bptt-v1"


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
        raise ValueError("Expert dataset environment version mismatch.")
    if manifest["observationVersion"] != OBSERVATION_VERSION:
        raise ValueError("Expert dataset observation version mismatch.")
    if manifest["observationSize"] != OBSERVATION_INPUT_SIZE:
        raise ValueError("Expert dataset observation width mismatch.")
    if manifest["actionOrder"] != [action.value for action in ACTION_ORDER]:
        raise ValueError("Expert dataset action order mismatch.")

    expected_hash = manifest["dataset"]["sha256"]
    actual_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("Expert dataset hash does not match manifest.")

    with np.load(dataset_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}

    observations = arrays["observations"]
    actions = arrays["actions"]
    offsets = arrays["episode_offsets"]
    seeds = arrays["seeds"]

    if observations.ndim != 2 or observations.shape[1] != OBSERVATION_INPUT_SIZE:
        raise ValueError("Expert observations have an incompatible shape.")
    if actions.shape != (observations.shape[0],):
        raise ValueError("Expert action array has an incompatible shape.")
    if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(actions):
        raise ValueError("Expert episode offsets are invalid.")
    if len(seeds) != len(offsets) - 1:
        raise ValueError("Expert seed count does not match episode offsets.")

    return manifest, arrays


def _split_episodes(
    episode_count: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if episode_count < 2:
        raise ValueError("Imitation training requires at least two episodes.")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")

    indices = list(range(episode_count))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(episode_count * validation_fraction))
    return sorted(indices[validation_count:]), sorted(indices[:validation_count])


def _episode_slice(offsets: np.ndarray, episode: int) -> slice:
    return slice(int(offsets[episode]), int(offsets[episode + 1]))


def _class_weights(
    actions: np.ndarray,
    offsets: np.ndarray,
    episodes: list[int],
    device: torch.device,
) -> Tensor:
    counts = np.zeros(len(ACTION_ORDER), dtype=np.int64)
    for episode in episodes:
        counts += np.bincount(
            actions[_episode_slice(offsets, episode)],
            minlength=len(ACTION_ORDER),
        )

    safe_counts = np.maximum(counts, 1)
    frequencies = safe_counts / safe_counts.sum()
    weights = np.sqrt(1.0 / (len(ACTION_ORDER) * frequencies))
    weights = np.clip(weights, 0.5, 3.0)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _batch_tensors(
    observations: np.ndarray,
    actions: np.ndarray,
    offsets: np.ndarray,
    episodes: list[int],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    lengths = [
        int(offsets[episode + 1] - offsets[episode])
        for episode in episodes
    ]
    maximum = max(lengths)
    batch = len(episodes)

    observation_batch = torch.zeros(
        maximum,
        batch,
        OBSERVATION_INPUT_SIZE,
        dtype=torch.float32,
        device=device,
    )
    action_batch = torch.zeros(
        maximum,
        batch,
        dtype=torch.long,
        device=device,
    )
    active = torch.zeros(
        maximum,
        batch,
        dtype=torch.bool,
        device=device,
    )

    for column, episode in enumerate(episodes):
        episode_slice = _episode_slice(offsets, episode)
        length = lengths[column]
        observation_batch[:length, column] = torch.from_numpy(
            observations[episode_slice]
        ).to(device=device, dtype=torch.float32)
        action_batch[:length, column] = torch.from_numpy(
            actions[episode_slice]
        ).to(device=device, dtype=torch.long)
        active[:length, column] = True

    return observation_batch, action_batch, active


def _train_batch(
    model: PopulationFixedGraphPolicy,
    optimizer: torch.optim.Optimizer,
    observations: Tensor,
    actions: Tensor,
    active: Tensor,
    class_weights: Tensor,
) -> tuple[float, int, int]:
    model.train()
    hidden = torch.zeros(
        observations.shape[1],
        model.graph.node_count,
        dtype=torch.float32,
        device=observations.device,
    )
    total_loss = torch.zeros((), device=observations.device)
    total_examples = 0
    total_correct = 0

    for timestep in range(observations.shape[0]):
        mask = active[timestep]
        if not bool(mask.any()):
            continue

        logits, _, next_hidden = model(
            observations[timestep],
            hidden,
        )
        per_example = F.cross_entropy(
            logits,
            actions[timestep],
            weight=class_weights,
            reduction="none",
            label_smoothing=0.02,
        )
        total_loss = total_loss + per_example[mask].sum()
        total_examples += int(mask.sum().item())
        total_correct += int(
            (torch.argmax(logits[mask], dim=1) == actions[timestep, mask])
            .sum()
            .item()
        )
        hidden = next_hidden * mask.unsqueeze(1)

    if total_examples <= 0:
        raise RuntimeError("Empty recurrent imitation batch.")

    loss = total_loss / total_examples
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return float(loss.detach().cpu()), total_correct, total_examples


@torch.no_grad()
def _evaluate(
    model: PopulationFixedGraphPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    offsets: np.ndarray,
    episodes: list[int],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    class_total = np.zeros(len(ACTION_ORDER), dtype=np.int64)
    class_correct = np.zeros(len(ACTION_ORDER), dtype=np.int64)

    for episode in episodes:
        episode_slice = _episode_slice(offsets, episode)
        episode_observations = torch.from_numpy(
            observations[episode_slice]
        ).to(device=device, dtype=torch.float32)
        episode_actions = torch.from_numpy(
            actions[episode_slice]
        ).to(device=device, dtype=torch.long)
        hidden = torch.zeros(
            1,
            model.graph.node_count,
            dtype=torch.float32,
            device=device,
        )

        for timestep in range(len(episode_actions)):
            logits, _, hidden = model(
                episode_observations[timestep : timestep + 1],
                hidden,
            )
            prediction = int(torch.argmax(logits, dim=1).item())
            target = int(episode_actions[timestep].item())
            total += 1
            class_total[target] += 1
            if prediction == target:
                correct += 1
                class_correct[target] += 1

    per_action: dict[str, float | None] = {}
    supported: list[float] = []
    for index, action in enumerate(ACTION_ORDER):
        if class_total[index] == 0:
            per_action[action.value] = None
        else:
            accuracy = float(class_correct[index] / class_total[index])
            per_action[action.value] = accuracy
            supported.append(accuracy)

    return {
        "accuracy": correct / max(1, total),
        "macroAccuracy": float(np.mean(supported)) if supported else 0.0,
        "perActionAccuracy": per_action,
        "samples": total,
    }


def train_imitation(
    *,
    dataset: Path,
    output: Path,
    epochs: int,
    batch_episodes: int,
    learning_rate: float,
    validation_fraction: float,
    seed: int,
    patience: int,
    device_name: str,
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if batch_episodes < 1:
        raise ValueError("batch_episodes must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if patience < 1:
        raise ValueError("patience must be positive.")

    manifest, arrays = _load_dataset(dataset)
    observations = arrays["observations"].astype(np.float32, copy=False)
    actions = arrays["actions"].astype(np.int64, copy=False)
    offsets = arrays["episode_offsets"].astype(np.int64, copy=False)
    seeds = [str(value) for value in arrays["seeds"].tolist()]

    device = _resolve_device(device_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    training_episodes, validation_episodes = _split_episodes(
        len(offsets) - 1,
        validation_fraction,
        seed,
    )

    graph = _build_controller_v2_graph(load_reduced_graph_variant("1k"))
    model = PopulationFixedGraphPolicy(
        graph,
        OBSERVATION_INPUT_SIZE,
        len(ACTION_ORDER),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    weights = _class_weights(
        actions,
        offsets,
        training_episodes,
        device,
    )

    output.mkdir(parents=True, exist_ok=True)
    training_curve: list[dict[str, Any]] = []
    best_macro = -math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    rng = random.Random(seed)

    for epoch in range(1, epochs + 1):
        shuffled = list(training_episodes)
        rng.shuffle(shuffled)
        losses: list[float] = []
        train_correct = 0
        train_total = 0

        for start in range(0, len(shuffled), batch_episodes):
            batch_ids = shuffled[start : start + batch_episodes]
            batch_observations, batch_actions, active = _batch_tensors(
                observations,
                actions,
                offsets,
                batch_ids,
                device,
            )
            loss, correct, total = _train_batch(
                model,
                optimizer,
                batch_observations,
                batch_actions,
                active,
                weights,
            )
            losses.append(loss)
            train_correct += correct
            train_total += total

        validation = _evaluate(
            model,
            observations,
            actions,
            offsets,
            validation_episodes,
            device,
        )
        epoch_metrics = {
            "epoch": epoch,
            "trainLoss": float(np.mean(losses)),
            "trainAccuracy": train_correct / max(1, train_total),
            "validationAccuracy": validation["accuracy"],
            "validationMacroAccuracy": validation["macroAccuracy"],
            "validationPerActionAccuracy": validation["perActionAccuracy"],
            "recurrentGain": float(model.recurrent_gain.detach().cpu()),
            "timeConstant": float(model.time_constant.detach().cpu()),
        }
        training_curve.append(epoch_metrics)

        print(
            f"[imitation] epoch={epoch:02d} "
            f"loss={epoch_metrics['trainLoss']:.4f} "
            f"train={epoch_metrics['trainAccuracy']:.3f} "
            f"val={epoch_metrics['validationAccuracy']:.3f} "
            f"macro={epoch_metrics['validationMacroAccuracy']:.3f}",
            flush=True,
        )

        macro = float(validation["macroAccuracy"])
        if macro > best_macro + 1e-6:
            best_macro = macro
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
        raise RuntimeError("Imitation trainer failed to produce a checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    final_training = _evaluate(
        model,
        observations,
        actions,
        offsets,
        training_episodes,
        device,
    )
    final_validation = _evaluate(
        model,
        observations,
        actions,
        offsets,
        validation_episodes,
        device,
    )

    checkpoint_path = output / "checkpoint.pt"
    checkpoint = {
        "format_version": 1,
        "environment_version": WORLD_VERSION,
        "controller": "connectome",
        "model": {
            "observation_size": OBSERVATION_INPUT_SIZE,
            "actions": len(ACTION_ORDER),
            "graph": graph.to_checkpoint(),
            "interface": "population",
        },
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "training": {
            "seed": str(manifest["seedPrefix"]),
            "steps": int(len(actions)),
            "envs": 1,
            "learning_rate": learning_rate,
            "world_seeds": seeds,
            "connectome_graph": "1k",
            "connectome_interface": "population",
            "training_mode": IMITATION_VERSION,
            "optimizer_seed": seed,
            "dataset_sha256": manifest["dataset"]["sha256"],
            "planner_version": manifest["plannerVersion"],
            "planner_depth": manifest["plannerDepth"],
            "epochs_requested": epochs,
            "best_epoch": best_epoch,
            "validation_fraction": validation_fraction,
            "batch_episodes": batch_episodes,
            "class_weights": [
                float(value)
                for value in weights.detach().cpu().tolist()
            ],
        },
    }
    torch.save(checkpoint, checkpoint_path)

    validated = validate_checkpoint(checkpoint)
    if validated.connectome_interface != "population":
        raise RuntimeError("Imitation checkpoint is not a plain population policy.")

    policy_path = output / "policy.json"
    export_policy(checkpoint_path, policy_path)

    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy["network"].get("interfaceMode") == "controller-v2":
        raise RuntimeError("Controller V2 heads leaked into imitation policy.")
    if "riskWeights" in policy["network"] or "routeWeights" in policy["network"]:
        raise RuntimeError("Auxiliary Controller V2 heads leaked into imitation policy.")

    metrics = {
        "version": 1,
        "trainingMode": IMITATION_VERSION,
        "bestEpoch": best_epoch,
        "epochsCompleted": len(training_curve),
        "trainingCurve": training_curve,
        "training": final_training,
        "validation": final_validation,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "version": 1,
        "environmentVersion": WORLD_VERSION,
        "observationVersion": OBSERVATION_VERSION,
        "trainingMode": IMITATION_VERSION,
        "device": str(device),
        "dataset": str(dataset),
        "datasetSha256": manifest["dataset"]["sha256"],
        "plannerVersion": manifest["plannerVersion"],
        "plannerDepth": manifest["plannerDepth"],
        "sensoryCells": len(graph.sensory_body_ids),
        "readoutCells": len(graph.readout_body_ids),
        "graphNodes": graph.node_count,
        "graphEdges": graph.edge_count,
        "graphArtifactHash": graph.artifact_sha256,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": checkpoint_hash,
        },
        "policy": {"path": policy_path.name},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cudaAvailable": torch.cuda.is_available(),
        },
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
        "validationAccuracy": final_validation["accuracy"],
        "validationMacroAccuracy": final_validation["macroAccuracy"],
        "validationPerActionAccuracy": final_validation["perActionAccuracy"],
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the recurrent 1k MaleCNS policy by imitation of offline "
            "expert trajectories."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    result = train_imitation(
        dataset=args.dataset,
        output=args.output,
        epochs=args.epochs,
        batch_episodes=args.batch_episodes,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        patience=args.patience,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
