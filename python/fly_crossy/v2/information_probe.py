from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from fly_crossy.env import GameState, create_game, step_game
from fly_crossy.expert_planner import PLANNER_VERSION, plan_action
from fly_crossy.schema import ACTION_ORDER, Action

from .core import FrozenMaleCNSCore
from .crossy_camera import render_crossy_neural_frame
from .substrate import MaleCNSV2Substrate
from .vision_frontend import FrozenVisualFrontEnd
from .vision_geometry import VisualGeometry


ACTION_NAMES = tuple(action.value for action in ACTION_ORDER)
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_ORDER)}
CORE_ACTION_INDICES = (
    ACTION_TO_INDEX[Action.FORWARD],
    ACTION_TO_INDEX[Action.LEFT],
    ACTION_TO_INDEX[Action.RIGHT],
    ACTION_TO_INDEX[Action.WAIT],
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class ProbeDataset:
    features: np.ndarray
    labels: np.ndarray
    episode: np.ndarray
    step: np.ndarray
    explored: np.ndarray
    seeds: tuple[str, ...]

    def validate(self) -> None:
        n = len(self.labels)
        if self.features.ndim != 2 or len(self.features) != n:
            raise ValueError("features/labels row count mismatch.")
        for name, value in {
            "episode": self.episode,
            "step": self.step,
            "explored": self.explored,
        }.items():
            if value.shape != (n,):
                raise ValueError(f"{name} shape does not match labels.")
        if self.labels.dtype != np.uint8:
            raise ValueError("labels must be uint8.")
        if np.any(self.labels >= len(ACTION_ORDER)):
            raise ValueError("labels contain an invalid action.")
        if not np.isfinite(self.features).all():
            raise ValueError("features contain non-finite values.")

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            features=self.features,
            labels=self.labels,
            episode=self.episode,
            step=self.step,
            explored=self.explored,
            seeds=np.asarray(self.seeds, dtype=f"<U{max(map(len, self.seeds), default=1)}"),
        )

    @classmethod
    def load(cls, path: Path) -> "ProbeDataset":
        with np.load(path, allow_pickle=False) as z:
            dataset = cls(
                features=z["features"].astype(np.float16, copy=True),
                labels=z["labels"].astype(np.uint8, copy=True),
                episode=z["episode"].astype(np.int16, copy=True),
                step=z["step"].astype(np.int16, copy=True),
                explored=z["explored"].astype(np.bool_, copy=True),
                seeds=tuple(str(item) for item in z["seeds"].tolist()),
            )
        dataset.validate()
        return dataset


def _safe_actions(state: GameState) -> tuple[Action, ...]:
    return tuple(
        action
        for action in ACTION_ORDER
        if step_game(state, action).state.terminal is None
    )


@torch.no_grad()
def collect_dataset(
    *,
    prefix: str,
    episodes: int,
    max_steps: int,
    explore_probability: float,
    frontend: FrozenVisualFrontEnd,
    brain: FrozenMaleCNSCore,
    rng: np.random.Generator,
) -> tuple[ProbeDataset, dict[str, object]]:
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive.")
    if not 0.0 <= explore_probability <= 1.0:
        raise ValueError("explore_probability must lie in [0,1].")

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    episode_rows: list[int] = []
    step_rows: list[int] = []
    explored_rows: list[bool] = []
    seeds: list[str] = []
    label_counts: Counter[str] = Counter()
    behaviour_counts: Counter[str] = Counter()
    terminal_counts: Counter[str] = Counter()
    planner_nodes = 0
    planner_cache_hits = 0
    actual_steps = 0

    started = time.perf_counter()

    for episode_index in range(episodes):
        seed = f"{prefix}:{episode_index:04d}"
        seeds.append(seed)
        state = create_game(seed)
        vision_state = frontend.init_state(1)
        brain_state = brain.init_state(1)

        for local_step in range(max_steps):
            frame = render_crossy_neural_frame(state)
            visual_rates, vision_state = frontend.process(frame[None], vision_state)
            brain_state = brain.step(brain_state, visual_rates)
            features = brain.decision_features(brain_state)[0]
            feature_rows.append(
                features.detach().to("cpu", dtype=torch.float16).numpy().copy()
            )

            decision = plan_action(state, depth=4)
            planner_nodes += decision.nodes_expanded
            planner_cache_hits += decision.cache_hits
            expert_action = decision.action
            label_counts[expert_action.value] += 1

            labels.append(ACTION_TO_INDEX[expert_action])
            episode_rows.append(episode_index)
            step_rows.append(local_step)

            behaviour_action = expert_action
            explored = False
            if rng.random() < explore_probability:
                safe = _safe_actions(state)
                alternatives = tuple(action for action in safe if action != expert_action)
                if alternatives:
                    behaviour_action = alternatives[int(rng.integers(0, len(alternatives)))]
                    explored = True

            explored_rows.append(explored)
            behaviour_counts[behaviour_action.value] += 1
            actual_steps += 1

            next_state = step_game(state, behaviour_action).state
            if next_state.terminal is not None:
                terminal_counts[next_state.terminal] += 1
                break
            state = next_state

    if not feature_rows:
        raise RuntimeError("Information probe collection produced no samples.")

    dataset = ProbeDataset(
        features=np.stack(feature_rows).astype(np.float16, copy=False),
        labels=np.asarray(labels, dtype=np.uint8),
        episode=np.asarray(episode_rows, dtype=np.int16),
        step=np.asarray(step_rows, dtype=np.int16),
        explored=np.asarray(explored_rows, dtype=np.bool_),
        seeds=tuple(seeds),
    )
    dataset.validate()

    elapsed = time.perf_counter() - started
    meta = {
        "prefix": prefix,
        "episodesRequested": episodes,
        "samples": len(dataset.labels),
        "featureSize": int(dataset.features.shape[1]),
        "exploreProbability": explore_probability,
        "exploredSamples": int(dataset.explored.sum()),
        "expertLabelCounts": dict(sorted(label_counts.items())),
        "behaviourActionCounts": dict(sorted(behaviour_counts.items())),
        "terminalCounts": dict(sorted(terminal_counts.items())),
        "meanPlannerNodesPerSample": planner_nodes / max(1, actual_steps),
        "meanPlannerCacheHitsPerSample": planner_cache_hits / max(1, actual_steps),
        "elapsedSeconds": elapsed,
        "samplesPerSecond": len(dataset.labels) / max(elapsed, 1e-9),
    }
    return dataset, meta


def readout_feature_groups(substrate: MaleCNSV2Substrate) -> dict[str, np.ndarray]:
    readout_nodes = np.flatnonzero(substrate.decision_readout_mask)
    n = len(readout_nodes)

    dn_positions = np.flatnonzero(substrate.is_output[readout_nodes])
    vpn_positions = np.flatnonzero(substrate.is_visual_projection[readout_nodes])

    def with_delta(position: np.ndarray) -> np.ndarray:
        return np.concatenate([position, position + n]).astype(np.int64)

    groups = {
        "dn": with_delta(dn_positions),
        "vpn": with_delta(vpn_positions),
        "dn_vpn": np.arange(2 * n, dtype=np.int64),
    }

    if len(dn_positions) != 1_312:
        raise ValueError(f"Expected 1,312 DN readouts, got {len(dn_positions):,}.")
    if len(vpn_positions) != 9_199:
        raise ValueError(f"Expected 9,199 VPN readouts, got {len(vpn_positions):,}.")
    if n != 10_511:
        raise ValueError(f"Expected 10,511 combined readouts, got {n:,}.")
    return groups


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(ACTION_ORDER)).astype(np.float64)
    supported = counts > 0
    weights = np.ones(len(ACTION_ORDER), dtype=np.float32)
    if supported.any():
        reference = counts[supported].mean()
        weights[supported] = np.sqrt(reference / counts[supported]).astype(np.float32)
        weights = np.clip(weights, 0.35, 6.0)
    return weights


def feature_stats(features: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Keep memory bounded: select once as float32, compute moments, release.
    selected = features[:, indices].astype(np.float32)
    mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    variance = selected.var(axis=0, dtype=np.float64).astype(np.float32)
    std = np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32)
    return mean, std


def metrics_from_predictions(
    labels: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:
    if labels.shape != predicted.shape:
        raise ValueError("labels/predicted shape mismatch.")

    confusion = np.zeros((len(ACTION_ORDER), len(ACTION_ORDER)), dtype=np.int64)
    for truth, pred in zip(labels.tolist(), predicted.tolist(), strict=True):
        confusion[int(truth), int(pred)] += 1

    support = confusion.sum(axis=1)
    correct = np.diag(confusion)
    per_action = {
        ACTION_NAMES[index]: (
            float(correct[index] / support[index])
            if support[index] > 0
            else None
        )
        for index in range(len(ACTION_ORDER))
    }

    supported_indices = np.flatnonzero(support > 0)
    supported_macro = (
        float(np.mean(correct[supported_indices] / support[supported_indices]))
        if len(supported_indices)
        else 0.0
    )

    core_supported = [
        index for index in CORE_ACTION_INDICES if support[index] > 0
    ]
    core_macro = (
        float(np.mean([correct[i] / support[i] for i in core_supported]))
        if core_supported
        else 0.0
    )

    return {
        "samples": int(len(labels)),
        "accuracy": float((labels == predicted).mean()) if len(labels) else 0.0,
        "macroAccuracySupported": supported_macro,
        "coreMacroAccuracy": core_macro,
        "support": {
            ACTION_NAMES[index]: int(support[index])
            for index in range(len(ACTION_ORDER))
        },
        "perActionAccuracy": per_action,
        "predictedCounts": {
            ACTION_NAMES[index]: int((predicted == index).sum())
            for index in range(len(ACTION_ORDER))
        },
        "confusion": {
            ACTION_NAMES[row]: {
                ACTION_NAMES[col]: int(confusion[row, col])
                for col in range(len(ACTION_ORDER))
            }
            for row in range(len(ACTION_ORDER))
        },
    }


class LinearProbe(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.actor = nn.Linear(input_size, len(ACTION_ORDER))

    def forward(self, x: Tensor) -> Tensor:
        return self.actor(x)


class MlpProbe(nn.Module):
    def __init__(self, input_size: int, hidden: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, len(ACTION_ORDER)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


def _evaluate(
    model: nn.Module,
    dataset: ProbeDataset,
    indices: np.ndarray,
    mean: Tensor,
    std: Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    predicted_parts: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(dataset.labels), batch_size):
            end = min(len(dataset.labels), start + batch_size)
            x = torch.as_tensor(
                dataset.features[start:end, indices],
                dtype=torch.float32,
                device=device,
            )
            x = (x - mean) / std
            logits = model(x)
            predicted_parts.append(
                logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
            )

    predicted = np.concatenate(predicted_parts) if predicted_parts else np.zeros(0, dtype=np.uint8)
    return metrics_from_predictions(dataset.labels, predicted)


def train_probe(
    *,
    train: ProbeDataset,
    validation: ProbeDataset,
    indices: np.ndarray,
    architecture: str,
    device: torch.device,
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    shuffle_labels: bool = False,
) -> dict[str, object]:
    if architecture not in {"linear", "mlp"}:
        raise ValueError("architecture must be linear or mlp.")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    mean_np, std_np = feature_stats(train.features, indices)
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)

    model: nn.Module
    if architecture == "linear":
        model = LinearProbe(len(indices))
    else:
        model = MlpProbe(len(indices), hidden=256)
    model.to(device)

    train_labels = train.labels.copy()
    if shuffle_labels:
        rng.shuffle(train_labels)

    weights = torch.as_tensor(
        class_weights(train_labels),
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    best_state: dict[str, Tensor] | None = None
    best_core_macro = -1.0
    best_epoch = 0
    history: list[dict[str, object]] = []

    order = np.arange(len(train.labels), dtype=np.int64)
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)
        loss_sum = 0.0
        seen = 0

        for start in range(0, len(order), batch_size):
            batch_rows = order[start:start + batch_size]
            x = torch.as_tensor(
                train.features[batch_rows][:, indices],
                dtype=torch.float32,
                device=device,
            )
            y = torch.as_tensor(
                train_labels[batch_rows],
                dtype=torch.long,
                device=device,
            )
            x = (x - mean) / std

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            loss_sum += float(loss) * len(batch_rows)
            seen += len(batch_rows)

        validation_metrics = _evaluate(
            model,
            validation,
            indices,
            mean,
            std,
            device=device,
            batch_size=batch_size,
        )
        core_macro = float(validation_metrics["coreMacroAccuracy"])

        history.append(
            {
                "epoch": epoch,
                "trainLoss": loss_sum / max(1, seen),
                "validationAccuracy": validation_metrics["accuracy"],
                "validationCoreMacroAccuracy": core_macro,
            }
        )

        if core_macro > best_core_macro:
            best_core_macro = core_macro
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Probe training produced no checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    train_metrics = _evaluate(
        model,
        train,
        indices,
        mean,
        std,
        device=device,
        batch_size=batch_size,
    )
    validation_metrics = _evaluate(
        model,
        validation,
        indices,
        mean,
        std,
        device=device,
        batch_size=batch_size,
    )

    return {
        "architecture": architecture,
        "inputFeatures": int(len(indices)),
        "trainableParameters": int(sum(p.numel() for p in model.parameters())),
        "shuffleLabels": bool(shuffle_labels),
        "bestEpoch": best_epoch,
        "epochs": epochs,
        "learningRate": learning_rate,
        "elapsedSeconds": time.perf_counter() - started,
        "train": train_metrics,
        "validation": validation_metrics,
        "history": history,
    }


def _minimum_supported_accuracy(
    metrics: dict[str, object],
    names: tuple[str, ...],
) -> float:
    per_action = metrics["perActionAccuracy"]
    support = metrics["support"]
    values = [
        float(per_action[name])
        for name in names
        if int(support[name]) > 0 and per_action[name] is not None
    ]
    return min(values) if values else 0.0


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "MaleCNS Crossy V2 information probe. "
            "No connectome parameter is trained and this probe is diagnostic only."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--explore-probability", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--linear-epochs", type=int, default=10)
    parser.add_argument("--mlp-epochs", type=int, default=14)
    parser.add_argument("--control-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
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
        "--artifact-dir",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "information-probe",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "malecns-crossy-v2-information-probe.json",
    )
    parser.add_argument(
        "--reuse-dataset",
        action="store_true",
        help="reuse existing train/validation NPZs instead of collecting again",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    substrate = MaleCNSV2Substrate.load(args.substrate, strict=True)
    geometry = VisualGeometry.load(args.geometry, strict=True)
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

    if sum(p.numel() for p in brain.parameters()) != 0:
        raise SystemExit("Frozen MaleCNS core unexpectedly has trainable parameters.")
    if frontend.trainable_parameters != 0:
        raise SystemExit("Frozen visual frontend unexpectedly has trainable parameters.")

    groups = readout_feature_groups(substrate)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.artifact_dir / "train-features.npz"
    validation_path = args.artifact_dir / "validation-features.npz"

    print("=== MaleCNS Crossy V2 / Information Probe ===", flush=True)
    print(f"device: {device}", flush=True)
    print(
        f"brain: {substrate.node_count:,} cells / {substrate.edge_count:,} synapses / "
        f"{substrate.decision_readout_count:,} DN+VPN",
        flush=True,
    )
    print(
        f"features: DN={len(groups['dn']):,} / VPN={len(groups['vpn']):,} / "
        f"combined={len(groups['dn_vpn']):,}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)

    if args.reuse_dataset:
        print("[1/5] Reusing saved feature datasets...", flush=True)
        train_data = ProbeDataset.load(train_path)
        validation_data = ProbeDataset.load(validation_path)
        train_meta = {"reused": True, "samples": len(train_data.labels)}
        validation_meta = {"reused": True, "samples": len(validation_data.labels)}
    else:
        print("[1/5] Collecting TRAIN trajectories...", flush=True)
        train_data, train_meta = collect_dataset(
            prefix="malecns-v2-info-train",
            episodes=args.train_episodes,
            max_steps=args.max_steps,
            explore_probability=args.explore_probability,
            frontend=frontend,
            brain=brain,
            rng=rng,
        )
        train_data.save(train_path)
        print(
            f"  samples={len(train_data.labels):,} "
            f"labels={train_meta['expertLabelCounts']}",
            flush=True,
        )

        print("[2/5] Collecting VALIDATION trajectories on unseen seeds...", flush=True)
        validation_data, validation_meta = collect_dataset(
            prefix="malecns-v2-info-validation",
            episodes=args.validation_episodes,
            max_steps=args.max_steps,
            explore_probability=args.explore_probability,
            frontend=frontend,
            brain=brain,
            rng=rng,
        )
        validation_data.save(validation_path)
        print(
            f"  samples={len(validation_data.labels):,} "
            f"labels={validation_meta['expertLabelCounts']}",
            flush=True,
        )

    train_seed_set = set(train_data.seeds)
    validation_seed_set = set(validation_data.seeds)
    seed_overlap = sorted(train_seed_set & validation_seed_set)

    print("[3/5] Linear probes: DN / VPN / DN+VPN...", flush=True)
    linear_results = {}
    for group_name in ("dn", "vpn", "dn_vpn"):
        result = train_probe(
            train=train_data,
            validation=validation_data,
            indices=groups[group_name],
            architecture="linear",
            device=device,
            seed=args.seed + {"dn": 1, "vpn": 2, "dn_vpn": 3}[group_name],
            epochs=args.linear_epochs,
            learning_rate=8e-4,
            batch_size=args.batch_size,
        )
        linear_results[group_name] = result
        vm = result["validation"]
        print(
            f"  {group_name:6s}: acc={vm['accuracy']:.3f} "
            f"coreMacro={vm['coreMacroAccuracy']:.3f} "
            f"perAction={vm['perActionAccuracy']}",
            flush=True,
        )

    print("[4/5] Nonlinear combined DN+VPN probe...", flush=True)
    mlp_result = train_probe(
        train=train_data,
        validation=validation_data,
        indices=groups["dn_vpn"],
        architecture="mlp",
        device=device,
        seed=args.seed + 10,
        epochs=args.mlp_epochs,
        learning_rate=2e-4,
        batch_size=args.batch_size,
    )
    mlp_validation = mlp_result["validation"]
    print(
        f"  acc={mlp_validation['accuracy']:.3f} "
        f"coreMacro={mlp_validation['coreMacroAccuracy']:.3f}",
        flush=True,
    )
    print(f"  perAction={mlp_validation['perActionAccuracy']}", flush=True)

    print("[5/5] Shuffled-label negative control...", flush=True)
    shuffled_result = train_probe(
        train=train_data,
        validation=validation_data,
        indices=groups["dn_vpn"],
        architecture="linear",
        device=device,
        seed=args.seed + 20,
        epochs=args.control_epochs,
        learning_rate=8e-4,
        batch_size=args.batch_size,
        shuffle_labels=True,
    )
    shuffled_validation = shuffled_result["validation"]
    print(
        f"  shuffled acc={shuffled_validation['accuracy']:.3f} "
        f"coreMacro={shuffled_validation['coreMacroAccuracy']:.3f}",
        flush=True,
    )

    validation_support = mlp_validation["support"]
    required_support = {
        name: int(validation_support[name])
        for name in ("forward", "left", "right", "wait")
    }
    all_core_supported = all(value >= 25 for value in required_support.values())

    min_lrw = _minimum_supported_accuracy(
        mlp_validation,
        ("left", "right", "wait"),
    )

    gates = {
        "frozenConnectomeHasZeroTrainableParameters": (
            sum(p.numel() for p in brain.parameters()) == 0
        ),
        "frozenVisualFrontendHasZeroTrainableParameters": (
            frontend.trainable_parameters == 0
        ),
        "trainValidationSeedsDisjoint": len(seed_overlap) == 0,
        "validationHasCoreActionSupport": all_core_supported,
        "combinedLinearCoreMacroAtLeast045": (
            float(linear_results["dn_vpn"]["validation"]["coreMacroAccuracy"]) >= 0.45
        ),
        "combinedMlpCoreMacroAtLeast055": (
            float(mlp_validation["coreMacroAccuracy"]) >= 0.55
        ),
        "leftRightWaitEachAtLeast040": min_lrw >= 0.40,
        "shuffledControlCoreMacroAtMost030": (
            float(shuffled_validation["coreMacroAccuracy"]) <= 0.30
        ),
    }

    report = {
        "version": 1,
        "purpose": (
            "Diagnostic information probe only: test whether frozen MaleCNS DN/VPN "
            "activity contains enough information to recover planner decisions on unseen seeds."
        ),
        "plannerVersion": PLANNER_VERSION,
        "plannerDepth": 4,
        "actionOrder": list(ACTION_NAMES),
        "camera": "fixed-perspective-crossy-neural-camera",
        "features": {
            "readoutCells": 10_511,
            "currentPlusDelta": True,
            "totalCombinedFeatures": int(len(groups["dn_vpn"])),
            "dnFeatures": int(len(groups["dn"])),
            "vpnFeatures": int(len(groups["vpn"])),
        },
        "frozenSystem": {
            "MaleCNSCells": substrate.node_count,
            "MaleCNSSynapses": substrate.edge_count,
            "visualCells": geometry.n_cells,
            "globalGain": gain,
            "connectomeTrainableParameters": sum(p.numel() for p in brain.parameters()),
            "frontendTrainableParameters": frontend.trainable_parameters,
        },
        "collection": {
            "train": train_meta,
            "validation": validation_meta,
            "trainArtifact": str(train_path.resolve()),
            "validationArtifact": str(validation_path.resolve()),
            "seedOverlap": seed_overlap,
        },
        "linear": linear_results,
        "mlpCombined": mlp_result,
        "shuffledControl": shuffled_result,
        "gates": gates,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"report: {args.output.resolve()}", flush=True)
    print(f"dataset: {args.artifact_dir.resolve()}", flush=True)

    if all(gates.values()):
        print("\nINFORMATION PROBE GATE: PASS", flush=True)
        print(
            "Interpretation: the frozen MaleCNS activity contains recoverable "
            "Crossy decision information on unseen seeds.",
            flush=True,
        )
    else:
        print("\nINFORMATION PROBE GATE: REVIEW", flush=True)
        print(json.dumps(gates, indent=2), flush=True)
        print(
            "Do not train the final decoder yet. Use DN/VPN probe results to "
            "localize where decision information is being lost.",
            flush=True,
        )


if __name__ == "__main__":
    main()
