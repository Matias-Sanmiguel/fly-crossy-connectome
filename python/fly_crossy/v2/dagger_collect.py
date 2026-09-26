from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from fly_crossy.env import GameState, create_game, generate_rows, step_game
from fly_crossy.expert_planner import PLANNER_VERSION, plan_action
from fly_crossy.schema import ACTION_ORDER, Action

from .core import FrozenMaleCNSCore
from .crossy_camera import render_crossy_neural_frame
from .decoder import EXPECTED_INPUT_SIZE, decoder_logits, load_decoder
from .substrate import MaleCNSV2Substrate
from .vision_frontend import FrozenVisualFrontEnd
from .vision_geometry import VisualGeometry


ACTION_NAMES = tuple(action.value for action in ACTION_ORDER)
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_ORDER)}
LANE_NAMES = ("grass", "road", "rail", "river")
LANE_TO_INDEX = {name: index for index, name in enumerate(LANE_NAMES)}
TERMINAL_NAMES = ("none", "vehicle", "train", "water", "bounds")
TERMINAL_TO_INDEX = {name: index for index, name in enumerate(TERMINAL_NAMES)}

CORE_ACTIONS = ("forward", "left", "right", "wait")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _lane_kind(state: GameState, row: int) -> str:
    lane = next((candidate for candidate in state.lanes if candidate.row == row), None)
    if lane is None:
        lane = generate_rows(state.seed, row, 1)[0]
    return lane.kind


def _destination_row(state: GameState, action: Action) -> int:
    if action == Action.FORWARD:
        return state.fly.row + 1
    if action == Action.BACKWARD:
        return state.fly.row - 1
    return state.fly.row


def _softmax_stats(logits: np.ndarray) -> tuple[float, float, float]:
    x = logits.astype(np.float64)
    x -= x.max()
    p = np.exp(x)
    p /= p.sum()
    ordered = np.sort(p)
    max_probability = float(ordered[-1])
    margin = float(ordered[-1] - ordered[-2])
    entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    return max_probability, margin, entropy


@dataclass(frozen=True, slots=True)
class DaggerRolloutDataset:
    features: np.ndarray
    logits: np.ndarray
    student_action: np.ndarray
    expert_action: np.ndarray
    disagreement: np.ndarray
    critical_error: np.ndarray
    student_terminal: np.ndarray
    expert_terminal: np.ndarray
    episode_terminal: np.ndarray
    steps_to_terminal: np.ndarray
    episode: np.ndarray
    step: np.ndarray
    score: np.ndarray
    fly_row: np.ndarray
    fly_column: np.ndarray
    current_lane: np.ndarray
    forward_lane: np.ndarray
    student_destination_lane: np.ndarray
    expert_destination_lane: np.ndarray
    max_probability: np.ndarray
    probability_margin: np.ndarray
    entropy: np.ndarray
    seeds: tuple[str, ...]

    def validate(self) -> None:
        n = len(self.student_action)
        if self.features.shape != (n, EXPECTED_INPUT_SIZE):
            raise ValueError(
                f"features shape {self.features.shape} != ({n}, {EXPECTED_INPUT_SIZE})"
            )
        if self.logits.shape != (n, len(ACTION_ORDER)):
            raise ValueError("logits shape does not match actions.")

        one_dimensional = (
            "expert_action",
            "disagreement",
            "critical_error",
            "student_terminal",
            "expert_terminal",
            "episode_terminal",
            "steps_to_terminal",
            "episode",
            "step",
            "score",
            "fly_row",
            "fly_column",
            "current_lane",
            "forward_lane",
            "student_destination_lane",
            "expert_destination_lane",
            "max_probability",
            "probability_margin",
            "entropy",
        )
        for name in one_dimensional:
            value = getattr(self, name)
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {value.shape}.")

        if not np.isfinite(self.features).all():
            raise ValueError("features contain non-finite values.")
        if not np.isfinite(self.logits).all():
            raise ValueError("logits contain non-finite values.")
        if np.any(self.student_action >= len(ACTION_ORDER)):
            raise ValueError("student_action contains an invalid action.")
        if np.any(self.expert_action >= len(ACTION_ORDER)):
            raise ValueError("expert_action contains an invalid action.")
        if np.any(self.current_lane >= len(LANE_NAMES)):
            raise ValueError("current_lane contains an invalid lane.")
        if np.any(self.forward_lane >= len(LANE_NAMES)):
            raise ValueError("forward_lane contains an invalid lane.")
        if np.any(self.student_terminal >= len(TERMINAL_NAMES)):
            raise ValueError("student_terminal contains an invalid terminal code.")
        if np.any(self.expert_terminal >= len(TERMINAL_NAMES)):
            raise ValueError("expert_terminal contains an invalid terminal code.")

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            features=self.features,
            logits=self.logits,
            student_action=self.student_action,
            expert_action=self.expert_action,
            disagreement=self.disagreement,
            critical_error=self.critical_error,
            student_terminal=self.student_terminal,
            expert_terminal=self.expert_terminal,
            episode_terminal=self.episode_terminal,
            steps_to_terminal=self.steps_to_terminal,
            episode=self.episode,
            step=self.step,
            score=self.score,
            fly_row=self.fly_row,
            fly_column=self.fly_column,
            current_lane=self.current_lane,
            forward_lane=self.forward_lane,
            student_destination_lane=self.student_destination_lane,
            expert_destination_lane=self.expert_destination_lane,
            max_probability=self.max_probability,
            probability_margin=self.probability_margin,
            entropy=self.entropy,
            seeds=np.asarray(
                self.seeds,
                dtype=f"<U{max((len(seed) for seed in self.seeds), default=1)}",
            ),
        )


def _confusion(
    expert_action: np.ndarray,
    student_action: np.ndarray,
) -> dict[str, dict[str, int]]:
    matrix = np.zeros((len(ACTION_ORDER), len(ACTION_ORDER)), dtype=np.int64)
    for expert, student in zip(
        expert_action.tolist(),
        student_action.tolist(),
        strict=True,
    ):
        matrix[int(expert), int(student)] += 1

    return {
        ACTION_NAMES[row]: {
            ACTION_NAMES[column]: int(matrix[row, column])
            for column in range(len(ACTION_ORDER))
        }
        for row in range(len(ACTION_ORDER))
    }


def _agreement_metrics(dataset: DaggerRolloutDataset) -> dict[str, object]:
    result: dict[str, object] = {}
    supported_core: list[float] = []

    for index, name in enumerate(ACTION_NAMES):
        mask = dataset.expert_action == index
        support = int(mask.sum())
        accuracy = (
            float((dataset.student_action[mask] == index).mean())
            if support
            else None
        )
        result[name] = {
            "support": support,
            "agreement": accuracy,
        }
        if name in CORE_ACTIONS and accuracy is not None:
            supported_core.append(accuracy)

    result["coreMacroAgreement"] = (
        float(np.mean(supported_core))
        if supported_core
        else 0.0
    )
    return result


def _lane_disagreement(dataset: DaggerRolloutDataset, field: str) -> dict[str, object]:
    values = getattr(dataset, field)
    result = {}
    for index, name in enumerate(LANE_NAMES):
        mask = values == index
        support = int(mask.sum())
        result[name] = {
            "support": support,
            "disagreementRate": (
                float(dataset.disagreement[mask].mean())
                if support
                else None
            ),
            "criticalErrors": int(dataset.critical_error[mask].sum()),
        }
    return result


def _confidence_group(
    dataset: DaggerRolloutDataset,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    support = int(mask.sum())
    if support == 0:
        return {
            "support": 0,
            "meanMaxProbability": None,
            "meanProbabilityMargin": None,
            "meanEntropy": None,
        }
    return {
        "support": support,
        "meanMaxProbability": float(dataset.max_probability[mask].mean()),
        "meanProbabilityMargin": float(dataset.probability_margin[mask].mean()),
        "meanEntropy": float(dataset.entropy[mask].mean()),
    }


def _predeath_windows(dataset: DaggerRolloutDataset) -> dict[str, object]:
    result = {}
    for terminal_index, terminal_name in enumerate(TERMINAL_NAMES[1:], start=1):
        reason_mask = dataset.episode_terminal == terminal_index
        per_window = {}
        for window in (1, 3, 5):
            mask = (
                reason_mask
                & (dataset.steps_to_terminal >= 0)
                & (dataset.steps_to_terminal < window)
            )
            support = int(mask.sum())
            per_window[str(window)] = {
                "samples": support,
                "disagreementRate": (
                    float(dataset.disagreement[mask].mean())
                    if support
                    else None
                ),
                "criticalErrors": int(dataset.critical_error[mask].sum()),
                "meanMaxProbability": (
                    float(dataset.max_probability[mask].mean())
                    if support
                    else None
                ),
            }
        result[terminal_name] = per_window
    return result


def summarize_dataset(
    dataset: DaggerRolloutDataset,
    *,
    episode_scores: list[float],
    episode_lengths: list[int],
    terminals: Counter[str],
    baseline_validation: dict[str, object] | None,
) -> dict[str, object]:
    dataset.validate()

    disagreement_rate = float(dataset.disagreement.mean())
    critical_count = int(dataset.critical_error.sum())

    student_counts = {
        name: int((dataset.student_action == index).sum())
        for index, name in enumerate(ACTION_NAMES)
    }
    expert_counts = {
        name: int((dataset.expert_action == index).sum())
        for index, name in enumerate(ACTION_NAMES)
    }

    critical_by_reason = {}
    for index, name in enumerate(TERMINAL_NAMES[1:], start=1):
        critical_by_reason[name] = int(
            (dataset.critical_error & (dataset.student_terminal == index)).sum()
        )

    immediate_student_deaths = int((dataset.student_terminal != 0).sum())
    immediate_expert_deaths = int((dataset.expert_terminal != 0).sum())

    agreement = _agreement_metrics(dataset)

    comparison = None
    if baseline_validation is not None:
        baseline_per_action = baseline_validation.get("perActionAccuracy", {})
        per_action_delta = {}
        for name in ACTION_NAMES:
            current = agreement[name]["agreement"]
            baseline = baseline_per_action.get(name)
            per_action_delta[name] = (
                None
                if current is None or baseline is None
                else float(current) - float(baseline)
            )
        comparison = {
            "baselineValidationAccuracy": baseline_validation.get("accuracy"),
            "baselineValidationCoreMacro": baseline_validation.get("coreMacroAccuracy"),
            "onPolicyCoreMacroAgreement": agreement["coreMacroAgreement"],
            "perActionAgreementDelta": per_action_delta,
        }

    return {
        "samples": len(dataset.student_action),
        "episodes": len(episode_scores),
        "meanScore": float(np.mean(episode_scores)) if episode_scores else 0.0,
        "medianScore": float(np.median(episode_scores)) if episode_scores else 0.0,
        "bestScore": float(max(episode_scores)) if episode_scores else 0.0,
        "meanLength": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "terminalReasons": dict(sorted(terminals.items())),
        "studentActionCounts": student_counts,
        "expertActionCounts": expert_counts,
        "disagreementRate": disagreement_rate,
        "criticalErrors": critical_count,
        "criticalErrorRate": critical_count / max(1, len(dataset.student_action)),
        "criticalErrorsByStudentTerminal": critical_by_reason,
        "studentImmediateTerminalActions": immediate_student_deaths,
        "expertImmediateTerminalActions": immediate_expert_deaths,
        "agreementByExpertAction": agreement,
        "confusionExpertRowsStudentColumns": _confusion(
            dataset.expert_action,
            dataset.student_action,
        ),
        "currentLane": _lane_disagreement(dataset, "current_lane"),
        "studentDestinationLane": _lane_disagreement(
            dataset,
            "student_destination_lane",
        ),
        "preDeathWindows": _predeath_windows(dataset),
        "confidence": {
            "agreement": _confidence_group(dataset, ~dataset.disagreement),
            "disagreement": _confidence_group(dataset, dataset.disagreement),
            "criticalError": _confidence_group(dataset, dataset.critical_error),
        },
        "baselineValidationComparison": comparison,
    }


@torch.no_grad()
def collect(
    *,
    episodes: int,
    max_steps: int,
    seed_prefix: str,
    frontend: FrozenVisualFrontEnd,
    brain: FrozenMaleCNSCore,
    decoder,
) -> tuple[DaggerRolloutDataset, list[float], list[int], Counter[str]]:
    features: list[np.ndarray] = []
    logits_rows: list[np.ndarray] = []
    student_actions: list[int] = []
    expert_actions: list[int] = []
    disagreements: list[bool] = []
    critical_errors: list[bool] = []
    student_terminals: list[int] = []
    expert_terminals: list[int] = []
    episode_terminals: list[int] = []
    steps_to_terminal: list[int] = []
    episode_rows: list[int] = []
    step_rows: list[int] = []
    scores: list[float] = []
    fly_rows: list[int] = []
    fly_columns: list[float] = []
    current_lanes: list[int] = []
    forward_lanes: list[int] = []
    student_destination_lanes: list[int] = []
    expert_destination_lanes: list[int] = []
    max_probabilities: list[float] = []
    probability_margins: list[float] = []
    entropies: list[float] = []
    seeds: list[str] = []

    episode_scores: list[float] = []
    episode_lengths: list[int] = []
    terminals: Counter[str] = Counter()

    for episode_index in range(episodes):
        seed = f"{seed_prefix}:{episode_index:04d}"
        seeds.append(seed)

        state = create_game(seed)
        vision_state = frontend.init_state(1)
        brain_state = brain.init_state(1)

        episode_start = len(student_actions)
        episode_terminal_name = "none"
        local_steps = 0

        for local_step in range(max_steps):
            frame = render_crossy_neural_frame(state)
            visual_rates, vision_state = frontend.process(frame[None], vision_state)
            brain_state = brain.step(brain_state, visual_rates)
            feature = brain.decision_features(brain_state)
            logits = decoder_logits(decoder, feature)[0].detach().cpu().numpy()

            student_index = int(np.argmax(logits))
            student_action = ACTION_ORDER[student_index]

            expert_decision = plan_action(state, depth=4)
            expert_action = expert_decision.action
            expert_index = ACTION_TO_INDEX[expert_action]

            student_next = step_game(state, student_action).state
            expert_next = step_game(state, expert_action).state

            student_terminal_name = student_next.terminal or "none"
            expert_terminal_name = expert_next.terminal or "none"

            disagreement = student_action != expert_action
            critical = (
                student_next.terminal is not None
                and expert_next.terminal is None
            )

            max_probability, probability_margin, entropy = _softmax_stats(logits)

            features.append(
                feature[0].detach().to("cpu", dtype=torch.float16).numpy().copy()
            )
            logits_rows.append(logits.astype(np.float16, copy=True))
            student_actions.append(student_index)
            expert_actions.append(expert_index)
            disagreements.append(disagreement)
            critical_errors.append(critical)
            student_terminals.append(TERMINAL_TO_INDEX[student_terminal_name])
            expert_terminals.append(TERMINAL_TO_INDEX[expert_terminal_name])
            episode_terminals.append(0)
            steps_to_terminal.append(-1)
            episode_rows.append(episode_index)
            step_rows.append(local_step)
            scores.append(float(state.score))
            fly_rows.append(int(state.fly.row))
            fly_columns.append(float(state.fly.column))
            current_lanes.append(LANE_TO_INDEX[_lane_kind(state, state.fly.row)])
            forward_lanes.append(LANE_TO_INDEX[_lane_kind(state, state.fly.row + 1)])
            student_destination_lanes.append(
                LANE_TO_INDEX[
                    _lane_kind(
                        state,
                        _destination_row(state, student_action),
                    )
                ]
            )
            expert_destination_lanes.append(
                LANE_TO_INDEX[
                    _lane_kind(
                        state,
                        _destination_row(state, expert_action),
                    )
                ]
            )
            max_probabilities.append(max_probability)
            probability_margins.append(probability_margin)
            entropies.append(entropy)

            state = student_next
            local_steps += 1

            if state.terminal is not None:
                episode_terminal_name = state.terminal
                terminals[state.terminal] += 1
                break

        episode_end = len(student_actions)
        terminal_code = TERMINAL_TO_INDEX[episode_terminal_name]

        if terminal_code != 0:
            terminal_sample = episode_end - 1
            for global_index in range(episode_start, episode_end):
                episode_terminals[global_index] = terminal_code
                steps_to_terminal[global_index] = terminal_sample - global_index

        episode_scores.append(float(state.score))
        episode_lengths.append(local_steps)

        if (episode_index + 1) % 10 == 0 or episode_index + 1 == episodes:
            print(
                f"  episode {episode_index + 1:3d}/{episodes}: "
                f"samples={len(student_actions):,} "
                f"meanScore={np.mean(episode_scores):.2f} "
                f"disagreement={np.mean(disagreements):.3f} "
                f"critical={sum(critical_errors)}",
                flush=True,
            )

    dataset = DaggerRolloutDataset(
        features=np.stack(features).astype(np.float16, copy=False),
        logits=np.stack(logits_rows).astype(np.float16, copy=False),
        student_action=np.asarray(student_actions, dtype=np.uint8),
        expert_action=np.asarray(expert_actions, dtype=np.uint8),
        disagreement=np.asarray(disagreements, dtype=np.bool_),
        critical_error=np.asarray(critical_errors, dtype=np.bool_),
        student_terminal=np.asarray(student_terminals, dtype=np.uint8),
        expert_terminal=np.asarray(expert_terminals, dtype=np.uint8),
        episode_terminal=np.asarray(episode_terminals, dtype=np.uint8),
        steps_to_terminal=np.asarray(steps_to_terminal, dtype=np.int16),
        episode=np.asarray(episode_rows, dtype=np.int16),
        step=np.asarray(step_rows, dtype=np.int16),
        score=np.asarray(scores, dtype=np.float32),
        fly_row=np.asarray(fly_rows, dtype=np.int16),
        fly_column=np.asarray(fly_columns, dtype=np.float32),
        current_lane=np.asarray(current_lanes, dtype=np.uint8),
        forward_lane=np.asarray(forward_lanes, dtype=np.uint8),
        student_destination_lane=np.asarray(
            student_destination_lanes,
            dtype=np.uint8,
        ),
        expert_destination_lane=np.asarray(
            expert_destination_lanes,
            dtype=np.uint8,
        ),
        max_probability=np.asarray(max_probabilities, dtype=np.float32),
        probability_margin=np.asarray(probability_margins, dtype=np.float32),
        entropy=np.asarray(entropies, dtype=np.float32),
        seeds=tuple(seeds),
    )
    dataset.validate()
    return dataset, episode_scores, episode_lengths, terminals


def _load_baseline_validation(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data.get("selected")
    if not isinstance(selected, dict):
        return None
    validation = selected.get("validation")
    return validation if isinstance(validation, dict) else None


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Collect ONE bounded MaleCNS Crossy V2 student-state DAgger dataset. "
            "The frozen R0 decoder drives every actual game step; the depth-4 planner "
            "labels the same pre-action state offline. This command does not train anything."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed-prefix", default="malecns-v2-dagger-r1")
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
        "--baseline-training-report",
        type=Path,
        default=root / "reports" / "malecns-crossy-v2-decoder-r0-training.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "dagger-r1-student-states.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "malecns-crossy-v2-dagger-r1-collection.json",
    )
    args = parser.parse_args()

    if args.episodes <= 0 or args.max_steps <= 0:
        raise SystemExit("episodes and max-steps must be positive.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

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

    if sum(parameter.numel() for parameter in brain.parameters()) != 0:
        raise SystemExit("MaleCNS core unexpectedly has trainable parameters.")
    if frontend.trainable_parameters != 0:
        raise SystemExit("Visual frontend unexpectedly has trainable parameters.")

    print("=== MaleCNS Crossy V2 / DAgger R1 Student-State Collection ===", flush=True)
    print("NO TRAINING in this command.", flush=True)
    print(f"controller: decoder R0 / argmax", flush=True)
    print(f"episodes:   {args.episodes}", flush=True)
    print(f"max steps:  {args.max_steps}", flush=True)
    print(f"seed prefix:{args.seed_prefix}", flush=True)

    started = time.perf_counter()
    dataset, episode_scores, episode_lengths, terminals = collect(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed_prefix=args.seed_prefix,
        frontend=frontend,
        brain=brain,
        decoder=decoder,
    )
    elapsed = time.perf_counter() - started

    baseline_validation = _load_baseline_validation(args.baseline_training_report)
    summary = summarize_dataset(
        dataset,
        episode_scores=episode_scores,
        episode_lengths=episode_lengths,
        terminals=terminals,
        baseline_validation=baseline_validation,
    )

    dataset.save(args.dataset)

    report = {
        "version": 1,
        "purpose": (
            "Bounded DAgger R1 diagnosis only. R0 argmax drives every real step; "
            "planner depth-4 only labels the same pre-action states offline."
        ),
        "plannerVersion": PLANNER_VERSION,
        "plannerDepth": 4,
        "controller": "MaleCNS-V2-decoder-R0-argmax",
        "architectureFrozen": {
            "camera": True,
            "visualFrontend": True,
            "MaleCNS": True,
            "globalGain": gain,
            "decoderArchitecture": "21022 -> 256 -> 5",
        },
        "collection": {
            "episodes": args.episodes,
            "maxSteps": args.max_steps,
            "seedPrefix": args.seed_prefix,
            "elapsedSeconds": elapsed,
            "samplesPerSecond": len(dataset.student_action) / max(elapsed, 1e-9),
        },
        "dataset": str(args.dataset.resolve()),
        "summary": summary,
        "nextStepPolicy": (
            "Do not train from this command. Inspect disagreement, fatal critical errors, "
            "per-action agreement, lane breakdown and pre-death windows first. "
            "At most one decoder DAgger fine-tune round is allowed afterward."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"samples:          {summary['samples']:,}", flush=True)
    print(f"mean score:       {summary['meanScore']:.2f}", flush=True)
    print(f"mean length:      {summary['meanLength']:.2f}", flush=True)
    print(f"deaths:           {summary['terminalReasons']}", flush=True)
    print(f"disagreement:     {summary['disagreementRate']:.3f}", flush=True)
    print(
        f"critical errors:  {summary['criticalErrors']} "
        f"({100*summary['criticalErrorRate']:.2f}%)",
        flush=True,
    )
    print(
        f"on-policy coreMacro agreement: "
        f"{summary['agreementByExpertAction']['coreMacroAgreement']:.3f}",
        flush=True,
    )
    print(
        f"per expert action: "
        f"{summary['agreementByExpertAction']}",
        flush=True,
    )
    print(
        f"critical by death: "
        f"{summary['criticalErrorsByStudentTerminal']}",
        flush=True,
    )
    print(f"dataset: {args.dataset.resolve()}", flush=True)
    print(f"report:  {args.output.resolve()}", flush=True)
    print()
    print("DAGGER R1 COLLECTION: COMPLETE", flush=True)
    print("No model weights were changed.", flush=True)


if __name__ == "__main__":
    main()
