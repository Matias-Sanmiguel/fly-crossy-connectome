from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from fly_crossy.env import GameState, create_game, step_game
from fly_crossy.expert_planner import PLANNER_ACTION_ORDER, PLANNER_VERSION, plan_action
from fly_crossy.schema import ACTION_ORDER, Action

from .core import FrozenMaleCNSCore
from .crossy_camera import render_crossy_neural_frame
from .decoder import EXPECTED_INPUT_SIZE, load_decoder, save_decoder
from .substrate import MaleCNSV2Substrate
from .vision_frontend import FrozenVisualFrontEnd
from .vision_geometry import VisualGeometry


PREFERENCE_VERSION = "malecns-crossy-v2-preference-distill-1"
VALUE_WIDTH = 6
PAIR_INDEX = tuple(
    (left, right)
    for left in range(len(ACTION_ORDER))
    for right in range(left + 1, len(ACTION_ORDER))
)
PAIR_COMPONENT_WEIGHTS = np.asarray(
    [4.0, 2.5, 2.0, 1.0, 0.5, 0.25], dtype=np.float32
)
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_ORDER)}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_action_count(state: GameState) -> int:
    if state.terminal is not None:
        return 0
    return sum(
        step_game(state, action).state.terminal is None
        for action in PLANNER_ACTION_ORDER
    )


def _leaf_value(
    state: GameState,
    *,
    root_score: float,
    root_row: int,
) -> tuple[float, ...]:
    score_gain = state.score - root_score
    row_gain = state.fly.row - root_row

    if state.terminal is not None:
        return (
            0.0,
            score_gain,
            row_gain,
            -abs(state.fly.column),
            0.0,
            0.0,
        )

    mobility = _safe_action_count(state)
    forward_safe = float(
        step_game(state, Action.FORWARD).state.terminal is None
    )
    return (
        1.0,
        score_gain,
        row_gain,
        float(mobility),
        forward_safe,
        -abs(state.fly.column),
    )


def _search_key(
    state: GameState,
    remaining: int,
) -> tuple[int, int, int, float, float, str | None]:
    return (
        remaining,
        state.step,
        state.fly.row,
        round(float(state.fly.column), 9),
        round(float(state.score), 9),
        state.terminal,
    )


def planner_action_preferences(
    state: GameState,
    *,
    depth: int = 4,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Return all five root-action values, in ACTION_ORDER."""
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if state.terminal is not None:
        raise ValueError("Cannot label a terminal state.")

    root_score = state.score
    root_row = state.fly.row
    cache: dict[
        tuple[int, int, int, float, float, str | None],
        tuple[float, ...],
    ] = {}

    def search(candidate: GameState, remaining: int) -> tuple[float, ...]:
        key = _search_key(candidate, remaining)
        cached = cache.get(key)
        if cached is not None:
            return cached

        if candidate.terminal is not None or remaining == 0:
            value = _leaf_value(
                candidate,
                root_score=root_score,
                root_row=root_row,
            )
            cache[key] = value
            return value

        best: tuple[float, ...] | None = None
        for action in PLANNER_ACTION_ORDER:
            next_state = step_game(candidate, action).state
            value = search(next_state, remaining - 1)
            if best is None or value > best:
                best = value

        assert best is not None
        cache[key] = best
        return best

    by_action: dict[Action, tuple[float, ...]] = {}
    best_action: Action | None = None
    best_value: tuple[float, ...] | None = None
    immediate_safe = np.zeros(len(ACTION_ORDER), dtype=np.bool_)

    for action in PLANNER_ACTION_ORDER:
        next_state = step_game(state, action).state
        immediate_safe[ACTION_TO_INDEX[action]] = next_state.terminal is None
        value = search(next_state, depth - 1)
        by_action[action] = value

        if best_value is None or value > best_value:
            best_value = value
            best_action = action

    assert best_action is not None
    values = np.asarray(
        [by_action[action] for action in ACTION_ORDER],
        dtype=np.float32,
    )
    return values, ACTION_TO_INDEX[best_action], immediate_safe


def primary_acceptable_mask(values: np.ndarray) -> np.ndarray:
    """Accept all actions tied on alive/score-gain/row-gain."""
    if values.shape != (len(ACTION_ORDER), VALUE_WIDTH):
        raise ValueError("Unexpected planner-value shape.")

    primary = [tuple(row[:3].tolist()) for row in values]
    best = max(primary)
    return np.asarray([value == best for value in primary], dtype=np.bool_)


def pairwise_targets(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    signs = np.zeros(len(PAIR_INDEX), dtype=np.int8)
    weights = np.zeros(len(PAIR_INDEX), dtype=np.float32)

    for pair_index, (left, right) in enumerate(PAIR_INDEX):
        a = tuple(float(x) for x in values[left])
        b = tuple(float(x) for x in values[right])

        if a == b:
            continue

        signs[pair_index] = 1 if a > b else -1
        first_difference = next(
            index
            for index, (x, y) in enumerate(zip(a, b, strict=True))
            if x != y
        )
        weights[pair_index] = PAIR_COMPONENT_WEIGHTS[first_difference]

    return signs, weights


@dataclass(frozen=True, slots=True)
class PreferenceDataset:
    features: np.ndarray
    values: np.ndarray
    acceptable: np.ndarray
    immediate_safe: np.ndarray
    best_action: np.ndarray
    pair_sign: np.ndarray
    pair_weight: np.ndarray
    sample_weight: np.ndarray
    episode: np.ndarray
    source: np.ndarray

    def validate(self) -> None:
        n = len(self.best_action)
        expected = {
            "features": (n, EXPECTED_INPUT_SIZE),
            "values": (n, len(ACTION_ORDER), VALUE_WIDTH),
            "acceptable": (n, len(ACTION_ORDER)),
            "immediate_safe": (n, len(ACTION_ORDER)),
            "pair_sign": (n, len(PAIR_INDEX)),
            "pair_weight": (n, len(PAIR_INDEX)),
            "sample_weight": (n,),
            "episode": (n,),
            "source": (n,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(
                    f"{name} shape {getattr(self, name).shape} != {shape}"
                )

        if not np.isfinite(self.features).all():
            raise ValueError("features contain non-finite values")
        if not np.isfinite(self.values).all():
            raise ValueError("values contain non-finite values")
        if not np.all(self.acceptable.any(axis=1)):
            raise ValueError("Every state must have an acceptable action")

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            features=self.features,
            values=self.values,
            acceptable=self.acceptable,
            immediate_safe=self.immediate_safe,
            best_action=self.best_action,
            pair_sign=self.pair_sign,
            pair_weight=self.pair_weight,
            sample_weight=self.sample_weight,
            episode=self.episode,
            source=self.source,
        )

    @classmethod
    def concatenate(
        cls,
        datasets: list["PreferenceDataset"],
    ) -> "PreferenceDataset":
        if not datasets:
            raise ValueError("Need at least one dataset")
        result = cls(
            features=np.concatenate([x.features for x in datasets]),
            values=np.concatenate([x.values for x in datasets]),
            acceptable=np.concatenate([x.acceptable for x in datasets]),
            immediate_safe=np.concatenate([x.immediate_safe for x in datasets]),
            best_action=np.concatenate([x.best_action for x in datasets]),
            pair_sign=np.concatenate([x.pair_sign for x in datasets]),
            pair_weight=np.concatenate([x.pair_weight for x in datasets]),
            sample_weight=np.concatenate([x.sample_weight for x in datasets]),
            episode=np.concatenate([x.episode for x in datasets]),
            source=np.concatenate([x.source for x in datasets]),
        )
        result.validate()
        return result


def _make_dataset(
    features,
    values,
    acceptable,
    immediate_safe,
    best_action,
    pair_sign,
    pair_weight,
    sample_weight,
    episode,
    source,
) -> PreferenceDataset:
    dataset = PreferenceDataset(
        features=np.stack(features).astype(np.float16, copy=False),
        values=np.stack(values).astype(np.float32, copy=False),
        acceptable=np.stack(acceptable).astype(np.bool_, copy=False),
        immediate_safe=np.stack(immediate_safe).astype(np.bool_, copy=False),
        best_action=np.asarray(best_action, dtype=np.uint8),
        pair_sign=np.stack(pair_sign).astype(np.int8, copy=False),
        pair_weight=np.stack(pair_weight).astype(np.float16, copy=False),
        sample_weight=np.asarray(sample_weight, dtype=np.float32),
        episode=np.asarray(episode, dtype=np.int32),
        source=np.asarray(source, dtype=np.uint8),
    )
    dataset.validate()
    return dataset


@torch.no_grad()
def collect_fresh_preferences(
    *,
    episodes: int,
    max_steps: int,
    seed_prefix: str,
    exploration: float,
    source_code: int,
    episode_offset: int,
    frontend: FrozenVisualFrontEnd,
    brain: FrozenMaleCNSCore,
    rng: np.random.Generator,
) -> PreferenceDataset:
    Ftrs, Vals, Acc, Safe, Best = [], [], [], [], []
    PS, PW, SW, Eps, Src = [], [], [], [], []

    for episode_index in range(episodes):
        state = create_game(f"{seed_prefix}:{episode_index:04d}")
        vision_state = frontend.init_state(1)
        brain_state = brain.init_state(1)

        for _ in range(max_steps):
            frame = render_crossy_neural_frame(state)
            visual_rates, vision_state = frontend.process(frame[None], vision_state)
            brain_state = brain.step(brain_state, visual_rates)
            feature = brain.decision_features(brain_state)[0]

            values, best_index, immediate_safe = planner_action_preferences(
                state, depth=4
            )
            acceptable = primary_acceptable_mask(values)
            signs, weights = pairwise_targets(values)

            Ftrs.append(
                feature.detach().to("cpu", dtype=torch.float16).numpy().copy()
            )
            Vals.append(values)
            Acc.append(acceptable)
            Safe.append(immediate_safe)
            Best.append(best_index)
            PS.append(signs)
            PW.append(weights)
            SW.append(1.0)
            Eps.append(episode_offset + episode_index)
            Src.append(source_code)

            behavior_index = best_index
            if rng.random() < exploration:
                alternatives = np.flatnonzero(immediate_safe)
                alternatives = alternatives[alternatives != best_index]
                if len(alternatives):
                    behavior_index = int(rng.choice(alternatives))

            state = step_game(state, ACTION_ORDER[behavior_index]).state
            if state.terminal is not None:
                break

        if (episode_index + 1) % 16 == 0 or episode_index + 1 == episodes:
            print(
                f"  {seed_prefix}: {episode_index + 1}/{episodes} "
                f"samples={len(Ftrs):,}",
                flush=True,
            )

    return _make_dataset(Ftrs, Vals, Acc, Safe, Best, PS, PW, SW, Eps, Src)


def replay_dagger_preferences(
    path: Path,
    *,
    episode_offset: int,
) -> PreferenceDataset | None:
    if not path.is_file():
        return None

    with np.load(path, allow_pickle=False) as z:
        required = ("features", "student_action", "episode", "step", "seeds")
        missing = [name for name in required if name not in z.files]
        if missing:
            raise ValueError(f"DAgger dataset missing arrays: {missing}")

        stored_features = z["features"].astype(np.float16, copy=True)
        student_action = z["student_action"].astype(np.uint8, copy=True)
        episode = z["episode"].astype(np.int32, copy=True)
        step = z["step"].astype(np.int32, copy=True)
        seeds = tuple(str(item) for item in z["seeds"].tolist())
        disagreement = (
            z["disagreement"].astype(np.bool_, copy=True)
            if "disagreement" in z.files
            else np.zeros(len(student_action), dtype=np.bool_)
        )
        critical = (
            z["critical_error"].astype(np.bool_, copy=True)
            if "critical_error" in z.files
            else np.zeros(len(student_action), dtype=np.bool_)
        )

    Ftrs, Vals, Acc, Safe, Best = [], [], [], [], []
    PS, PW, SW, Eps, Src = [], [], [], [], []

    for episode_index, seed in enumerate(seeds):
        rows = np.flatnonzero(episode == episode_index)
        if not len(rows):
            continue
        if not np.array_equal(step[rows], np.arange(len(rows))):
            raise RuntimeError(f"DAgger episode {episode_index} is not contiguous")

        state = create_game(seed)

        for row in rows.tolist():
            values, best_index, immediate_safe = planner_action_preferences(
                state, depth=4
            )
            acceptable = primary_acceptable_mask(values)
            signs, weights = pairwise_targets(values)

            Ftrs.append(stored_features[row])
            Vals.append(values)
            Acc.append(acceptable)
            Safe.append(immediate_safe)
            Best.append(best_index)
            PS.append(signs)
            PW.append(weights)

            weight = 1.0
            if disagreement[row]:
                weight += 0.5
            if critical[row]:
                weight += 1.5
            SW.append(weight)
            Eps.append(episode_offset + episode_index)
            Src.append(1)

            state = step_game(
                state,
                ACTION_ORDER[int(student_action[row])],
            ).state

        if (episode_index + 1) % 20 == 0 or episode_index + 1 == len(seeds):
            print(
                f"  dagger replay: {episode_index + 1}/{len(seeds)} "
                f"samples={len(Ftrs):,}",
                flush=True,
            )

    return _make_dataset(Ftrs, Vals, Acc, Safe, Best, PS, PW, SW, Eps, Src)


def preference_loss(
    logits: Tensor,
    acceptable: Tensor,
    immediate_safe: Tensor,
    pair_sign: Tensor,
    pair_weight: Tensor,
    sample_weight: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    masked = logits.masked_fill(~acceptable, -1e9)
    set_loss_each = (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(masked, dim=1)
    )

    pair_losses, pair_weights = [], []
    for pair_index, (left, right) in enumerate(PAIR_INDEX):
        sign = pair_sign[:, pair_index]
        weight = pair_weight[:, pair_index]
        margin = logits[:, left] - logits[:, right]
        pair_losses.append(F.softplus(-sign * margin) * weight)
        pair_weights.append(weight)

    pair_loss_matrix = torch.stack(pair_losses, dim=1)
    pair_weight_matrix = torch.stack(pair_weights, dim=1)
    pair_loss_each = pair_loss_matrix.sum(dim=1) / (
        pair_weight_matrix.sum(dim=1).clamp_min(1e-6)
    )

    probabilities = torch.softmax(logits, dim=1)
    fatal_mass = (
        probabilities * (~immediate_safe).to(probabilities.dtype)
    ).sum(dim=1)

    normalized_weight = sample_weight / sample_weight.mean().clamp_min(1e-6)
    set_loss = (set_loss_each * normalized_weight).mean()
    pair_loss = (pair_loss_each * normalized_weight).mean()
    fatal_loss = (fatal_mass * normalized_weight).mean()

    total = set_loss + 0.50 * pair_loss + 1.00 * fatal_loss
    return total, {
        "set": set_loss.detach(),
        "pair": pair_loss.detach(),
        "fatal": fatal_loss.detach(),
    }


@torch.no_grad()
def preference_metrics(
    logits: Tensor,
    dataset: PreferenceDataset,
) -> dict[str, object]:
    predicted = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
    rows = np.arange(len(predicted))

    acceptable_accuracy = float(dataset.acceptable[rows, predicted].mean())
    safe_rate = float(dataset.immediate_safe[rows, predicted].mean())
    exact_best = float((predicted == dataset.best_action).mean())

    signs = dataset.pair_sign
    valid = signs != 0
    logits_np = logits.cpu().numpy()
    correct = np.zeros_like(valid, dtype=np.bool_)
    for pair_index, (left, right) in enumerate(PAIR_INDEX):
        model_sign = np.sign(
            logits_np[:, left] - logits_np[:, right]
        ).astype(np.int8)
        correct[:, pair_index] = model_sign == signs[:, pair_index]

    pairwise = float(correct[valid].mean()) if valid.any() else 1.0
    counts = {
        ACTION_ORDER[index].value: int((predicted == index).sum())
        for index in range(len(ACTION_ORDER))
    }

    return {
        "samples": len(predicted),
        "acceptableSetAccuracy": acceptable_accuracy,
        "immediateSafeRate": safe_rate,
        "immediateFatalRate": 1.0 - safe_rate,
        "exactBestAccuracy": exact_best,
        "pairwiseAgreement": pairwise,
        "predictedActions": counts,
    }


@torch.no_grad()
def predict_all(
    model,
    dataset: PreferenceDataset,
    mean: Tensor,
    std: Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    model.eval()
    chunks = []

    for start in range(0, len(dataset.best_action), batch_size):
        x = torch.as_tensor(
            dataset.features[start:start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        chunks.append(model((x - mean) / std).detach().cpu())

    return torch.cat(chunks, dim=0)


@torch.no_grad()
def evaluate_closed_loop(
    model,
    *,
    mean: Tensor,
    std: Tensor,
    frontend: FrozenVisualFrontEnd,
    brain: FrozenMaleCNSCore,
    episodes: int,
    max_steps: int,
    seed_prefix: str,
) -> dict[str, object]:
    model.eval()
    scores, lengths = [], []
    terminal_reasons: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    reached_limit = 0

    for episode in range(episodes):
        state = create_game(f"{seed_prefix}:{episode:04d}")
        vision_state = frontend.init_state(1)
        brain_state = brain.init_state(1)
        steps = 0

        while state.terminal is None and steps < max_steps:
            frame = render_crossy_neural_frame(state)
            visual_rates, vision_state = frontend.process(frame[None], vision_state)
            brain_state = brain.step(brain_state, visual_rates)
            feature = brain.decision_features(brain_state)[0]
            logits = model(((feature - mean) / std)[None])[0]
            action_index = int(torch.argmax(logits).item())
            action = ACTION_ORDER[action_index]
            actions[action.value] += 1
            state = step_game(state, action).state
            steps += 1

        scores.append(float(state.score))
        lengths.append(steps)
        if state.terminal is None:
            reached_limit += 1
        else:
            terminal_reasons[state.terminal] += 1

    ordered = sorted(scores)
    return {
        "episodes": episodes,
        "maxSteps": max_steps,
        "meanScore": float(np.mean(scores)),
        "medianScore": float(ordered[len(ordered) // 2]),
        "bestScore": float(max(scores)),
        "meanLength": float(np.mean(lengths)),
        "reachedStepLimit": reached_limit,
        "terminalReasons": dict(sorted(terminal_reasons.items())),
        "actions": dict(sorted(actions.items())),
    }


def dev_key(metrics: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(metrics["meanScore"]),
        float(metrics["reachedStepLimit"]),
        float(metrics["medianScore"]),
        float(metrics["meanLength"]),
    )


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Planner preference/value distillation for MaleCNS Crossy V2. "
            "Planner is offline teacher only."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--exploration", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dev-every", type=int, default=4)
    parser.add_argument("--dev-episodes", type=int, default=12)
    parser.add_argument("--dev-max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "decoder-r0.pt",
    )
    parser.add_argument(
        "--dagger-dataset",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "dagger-r1-student-states.npz",
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
        "--train-dataset",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "preference-r1-train.npz",
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "preference-r1-validation.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "decoder-preference-r1.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "malecns-crossy-v2-preference-r1-training.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    substrate = MaleCNSV2Substrate.load(args.substrate, strict=True)
    geometry = VisualGeometry.load(args.geometry, strict=True)
    initial = load_decoder(args.initial_checkpoint, device=device)

    if not np.array_equal(
        geometry.cell_body_id,
        substrate.body_ids[substrate.is_clamped],
    ):
        raise SystemExit("Visual geometry/substrate clamped order mismatch.")

    frontend = FrozenVisualFrontEnd(
        geometry, decision_ms=200.0, internal_steps=10, device=device
    )
    brain = FrozenMaleCNSCore(
        substrate,
        decision_ms=200.0,
        internal_steps=10,
        tau_ms=20.0,
        device=device,
    )
    gain = brain.match_rest_gain(target=0.95, iterations=80)

    if sum(parameter.numel() for parameter in brain.parameters()) != 0:
        raise SystemExit("MaleCNS unexpectedly trainable.")
    if frontend.trainable_parameters != 0:
        raise SystemExit("Visual frontend unexpectedly trainable.")

    parity_state = create_game("preference-parity")
    for _ in range(8):
        _, best_index, _ = planner_action_preferences(parity_state, depth=4)
        planner = plan_action(parity_state, depth=4)
        if ACTION_ORDER[best_index] != planner.action:
            raise SystemExit("Preference search does not match canonical planner.")
        parity_state = step_game(parity_state, planner.action).state
        if parity_state.terminal is not None:
            break

    print("=== MaleCNS Crossy V2 / Preference Distillation R1 ===", flush=True)
    print("planner: OFFLINE TEACHER ONLY", flush=True)
    print("camera / retina / MaleCNS: FROZEN", flush=True)
    print("decoder: same 21022 -> 256 -> 5", flush=True)

    collection_started = time.perf_counter()

    print("\n[1/4] fresh train preferences...", flush=True)
    fresh_train = collect_fresh_preferences(
        episodes=args.train_episodes,
        max_steps=args.max_steps,
        seed_prefix="malecns-v2-pref-train",
        exploration=args.exploration,
        source_code=0,
        episode_offset=0,
        frontend=frontend,
        brain=brain,
        rng=rng,
    )

    print("\n[2/4] R0 student-state preference replay...", flush=True)
    dagger = replay_dagger_preferences(
        args.dagger_dataset,
        episode_offset=args.train_episodes,
    )
    train = (
        PreferenceDataset.concatenate([fresh_train, dagger])
        if dagger is not None
        else fresh_train
    )

    print("\n[3/4] held-out preference validation...", flush=True)
    validation = collect_fresh_preferences(
        episodes=args.validation_episodes,
        max_steps=args.max_steps,
        seed_prefix="malecns-v2-pref-validation",
        exploration=args.exploration,
        source_code=2,
        episode_offset=10_000,
        frontend=frontend,
        brain=brain,
        rng=np.random.default_rng(args.seed + 1),
    )

    train.save(args.train_dataset)
    validation.save(args.validation_dataset)
    print(
        f"  train={len(train.best_action):,} "
        f"validation={len(validation.best_action):,}",
        flush=True,
    )

    print("\n[4/4] train unchanged decoder with preference objective...", flush=True)

    model = initial.model
    mean = initial.mean.detach()
    std = initial.std.detach()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )

    order = np.arange(len(train.best_action), dtype=np.int64)

    initial_validation = preference_metrics(
        predict_all(
            model,
            validation,
            mean,
            std,
            device=device,
            batch_size=args.batch_size,
        ),
        validation,
    )
    initial_dev = evaluate_closed_loop(
        model,
        mean=mean,
        std=std,
        frontend=frontend,
        brain=brain,
        episodes=args.dev_episodes,
        max_steps=args.dev_max_steps,
        seed_prefix="malecns-v2-pref-dev",
    )

    print(
        f"  R0 acceptable={initial_validation['acceptableSetAccuracy']:.3f} "
        f"safe={initial_validation['immediateSafeRate']:.3f} "
        f"pair={initial_validation['pairwiseAgreement']:.3f}",
        flush=True,
    )
    print(
        f"  R0 dev mean={initial_dev['meanScore']:.2f} "
        f"median={initial_dev['medianScore']:.2f} "
        f"limit={initial_dev['reachedStepLimit']}/{args.dev_episodes}",
        flush=True,
    )

    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_dev = initial_dev
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    train_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(order)
        total_losses, set_losses, pair_losses, fatal_losses = [], [], [], []

        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]

            x = torch.as_tensor(
                train.features[rows], dtype=torch.float32, device=device
            )
            acceptable = torch.as_tensor(
                train.acceptable[rows], dtype=torch.bool, device=device
            )
            immediate_safe = torch.as_tensor(
                train.immediate_safe[rows], dtype=torch.bool, device=device
            )
            pair_sign = torch.as_tensor(
                train.pair_sign[rows], dtype=torch.float32, device=device
            )
            pair_weight = torch.as_tensor(
                train.pair_weight[rows], dtype=torch.float32, device=device
            )
            sample_weight = torch.as_tensor(
                train.sample_weight[rows], dtype=torch.float32, device=device
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model((x - mean) / std)
            loss, parts = preference_loss(
                logits,
                acceptable,
                immediate_safe,
                pair_sign,
                pair_weight,
                sample_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_losses.append(float(loss.detach().item()))
            set_losses.append(float(parts["set"].item()))
            pair_losses.append(float(parts["pair"].item()))
            fatal_losses.append(float(parts["fatal"].item()))

        validation_metrics = preference_metrics(
            predict_all(
                model,
                validation,
                mean,
                std,
                device=device,
                batch_size=args.batch_size,
            ),
            validation,
        )

        epoch_report: dict[str, object] = {
            "epoch": epoch,
            "loss": float(np.mean(total_losses)),
            "setLoss": float(np.mean(set_losses)),
            "pairLoss": float(np.mean(pair_losses)),
            "fatalMass": float(np.mean(fatal_losses)),
            "validation": validation_metrics,
        }

        should_dev = (
            epoch == 1
            or epoch % args.dev_every == 0
            or epoch == args.epochs
        )

        if should_dev:
            dev = evaluate_closed_loop(
                model,
                mean=mean,
                std=std,
                frontend=frontend,
                brain=brain,
                episodes=args.dev_episodes,
                max_steps=args.dev_max_steps,
                seed_prefix="malecns-v2-pref-dev",
            )
            epoch_report["dev"] = dev

            if dev_key(dev) > dev_key(best_dev):
                best_dev = dev
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1

            print(
                f"  epoch {epoch:02d}: "
                f"acceptable={validation_metrics['acceptableSetAccuracy']:.3f} "
                f"safe={validation_metrics['immediateSafeRate']:.3f} "
                f"pair={validation_metrics['pairwiseAgreement']:.3f} | "
                f"dev mean={dev['meanScore']:.2f} "
                f"median={dev['medianScore']:.2f} "
                f"limit={dev['reachedStepLimit']}/{args.dev_episodes} "
                f"bestEpoch={best_epoch}",
                flush=True,
            )

            if stale >= args.patience:
                history.append(epoch_report)
                print("  early stopping on closed-loop dev.", flush=True)
                break
        else:
            print(
                f"  epoch {epoch:02d}: "
                f"acceptable={validation_metrics['acceptableSetAccuracy']:.3f} "
                f"safe={validation_metrics['immediateSafeRate']:.3f} "
                f"pair={validation_metrics['pairwiseAgreement']:.3f}",
                flush=True,
            )

        history.append(epoch_report)

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    final_validation = preference_metrics(
        predict_all(
            model,
            validation,
            mean,
            std,
            device=device,
            batch_size=args.batch_size,
        ),
        validation,
    )

    metadata = {
        "preferenceVersion": PREFERENCE_VERSION,
        "plannerVersion": PLANNER_VERSION,
        "plannerDepth": 4,
        "plannerUsedAtInference": False,
        "initialCheckpoint": str(args.initial_checkpoint.resolve()),
        "bestEpoch": best_epoch,
        "bestDev": best_dev,
        "cameraFrozen": True,
        "visualFrontendFrozen": True,
        "MaleCNSFrozen": True,
        "decoderArchitecture": "21022 -> 256 GELU LayerNorm -> 5",
    }
    save_decoder(
        args.checkpoint,
        model=model,
        mean=mean.detach().cpu().numpy(),
        std=std.detach().cpu().numpy(),
        metadata=metadata,
    )

    report = {
        "version": 1,
        "preferenceVersion": PREFERENCE_VERSION,
        "checkpoint": str(args.checkpoint.resolve()),
        "initialCheckpoint": str(args.initial_checkpoint.resolve()),
        "frozenSystem": {
            "camera": True,
            "visualFrontend": True,
            "MaleCNS": True,
            "globalGain": gain,
        },
        "teacher": {
            "plannerVersion": PLANNER_VERSION,
            "depth": 4,
            "role": "offline only",
            "allFiveRootActionValuesPreserved": True,
            "oneHotPlannerImitation": False,
        },
        "datasets": {
            "trainSamples": len(train.best_action),
            "validationSamples": len(validation.best_action),
            "daggerStudentStatesIncluded": dagger is not None,
            "trainArtifact": str(args.train_dataset.resolve()),
            "validationArtifact": str(args.validation_dataset.resolve()),
        },
        "loss": {
            "acceptableSet": 1.0,
            "pairwiseRanking": 0.5,
            "immediateFatalMass": 1.0,
            "primaryAcceptableDimensions": [
                "alive_at_depth",
                "score_gain",
                "row_gain",
            ],
            "secondaryPreferenceDimensions": [
                "mobility",
                "forward_safe",
                "centering",
            ],
        },
        "initialValidation": initial_validation,
        "finalValidation": final_validation,
        "initialDev": initial_dev,
        "bestEpoch": best_epoch,
        "bestDev": best_dev,
        "history": history,
        "timing": {
            "collectionSeconds": train_started - collection_started,
            "trainingSeconds": time.perf_counter() - train_started,
        },
        "nextStep": (
            "Run existing decoder_eval with decoder-preference-r1.pt. "
            "Stop after the 50-seed closed-loop result."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nPREFERENCE DISTILLATION R1: COMPLETE", flush=True)
    print(f"best epoch: {best_epoch}", flush=True)
    print(
        f"best dev mean={best_dev['meanScore']:.2f} "
        f"median={best_dev['medianScore']:.2f} "
        f"limit={best_dev['reachedStepLimit']}/{args.dev_episodes}",
        flush=True,
    )
    print(
        f"validation acceptable={final_validation['acceptableSetAccuracy']:.3f} "
        f"safe={final_validation['immediateSafeRate']:.3f} "
        f"pair={final_validation['pairwiseAgreement']:.3f}",
        flush=True,
    )
    print(f"checkpoint: {args.checkpoint.resolve()}", flush=True)
    print(f"report:     {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
