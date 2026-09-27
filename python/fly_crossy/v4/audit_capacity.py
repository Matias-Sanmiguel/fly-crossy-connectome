from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pyarrow.feather as feather
import torch
from torch import nn
import torch.nn.functional as F

from fly_crossy.env import step_game
from fly_crossy.schema import ACTION_ORDER
from fly_crossy.v2.crossy_camera import render_crossy_neural_frame
from fly_crossy.v2.preference_distill import (
    planner_action_preferences,
    primary_acceptable_mask,
)
from fly_crossy.v4.policy import FullMaleCNSRGBPolicy
from fly_crossy.v4.train_expo_specialist import (
    EXPO_SEED,
    LANE_INDEX,
    SEMANTIC_DIM,
    default_flyhard_root,
    resize_rgb,
    repo_root,
    robustness_configs,
    semantic_target,
    variant_state,
)

ACTION_NAMES = [action.value for action in ACTION_ORDER]
LANE_NAMES = [None] * len(LANE_INDEX)
for _lane_name, _lane_index in LANE_INDEX.items():
    LANE_NAMES[_lane_index] = _lane_name


def _json_float(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return value


def _load_bundle(
    *,
    checkpoint_path: Path,
    flyhard_root: Path,
    device: torch.device,
) -> tuple[FullMaleCNSRGBPolicy, nn.Linear, dict[str, Any]]:
    graph_root = flyhard_root / "data" / "graph-traced-v1"
    graph_path = graph_root / "graph.npz"
    nodes_path = graph_root / "nodes.feather"
    if not graph_path.is_file() or not nodes_path.is_file():
        raise FileNotFoundError(
            "Full traced MaleCNS graph missing. Expected "
            "flyhard/data/graph-traced-v1/{graph.npz,nodes.feather}."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V4 checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    graph = dict(np.load(graph_path))
    nodes = feather.read_table(nodes_path)
    classes = np.asarray(nodes["superclass"].fill_null("").to_pylist())
    sensory_ids = np.flatnonzero(classes == "ol_sensory")
    motor_ids = np.flatnonzero(classes == "vnc_motor")

    config = checkpoint.get("config", {})
    rng_seed = int(config.get("rngSeed", 404)) if isinstance(config, dict) else 404
    policy = FullMaleCNSRGBPolicy(
        graph,
        sensory_ids,
        motor_ids,
        seed=rng_seed,
    ).to(device)
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.eval()

    semantic_head = nn.Linear(len(motor_ids), SEMANTIC_DIM).to(device)
    if "semanticHead" in checkpoint:
        semantic_head.load_state_dict(checkpoint["semanticHead"], strict=True)
    semantic_head.eval()

    metadata = {
        "checkpointStage": checkpoint.get("stage"),
        "rngSeed": rng_seed,
        "neurons": int(len(graph["crow"]) - 1),
        "edges": int(len(graph["col"])),
        "sensory": int(len(sensory_ids)),
        "motor": int(len(motor_ids)),
    }
    return policy, semantic_head, metadata


def _action_names(mask: np.ndarray) -> list[str]:
    return [ACTION_NAMES[index] for index in np.flatnonzero(mask)]


def _semantic_row(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    current_true = int(np.argmax(target[:4]))
    current_pred = int(np.argmax(prediction[:4]))
    next_true = int(np.argmax(target[4:8]))
    next_pred = int(np.argmax(prediction[4:8]))
    support_true = bool(target[8] >= 0.5)
    support_pred = bool(prediction[8] >= 0.5)
    traffic_delta = prediction[9:] - target[9:]
    return {
        "currentLaneTrue": LANE_NAMES[current_true],
        "currentLanePred": LANE_NAMES[current_pred],
        "currentLaneCorrect": current_true == current_pred,
        "nextLaneTrue": LANE_NAMES[next_true],
        "nextLanePred": LANE_NAMES[next_pred],
        "nextLaneCorrect": next_true == next_pred,
        "supportTrue": support_true,
        "supportPred": support_pred,
        "supportCorrect": support_true == support_pred,
        "trafficMse": _json_float(np.mean(traffic_delta * traffic_delta)),
        "semanticMse": _json_float(np.mean((prediction - target) ** 2)),
    }


def _summarize_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(trace))
    exact = sum(row["studentAction"] == row["teacherAction"] for row in trace)
    acceptable = sum(bool(row["studentAcceptable"]) for row in trace)
    safe = sum(bool(row["studentImmediateSafe"]) for row in trace)
    fatal = sum(bool(row["studentImmediateFatal"]) for row in trace)

    first_exact = next(
        (row for row in trace if row["studentAction"] != row["teacherAction"]),
        None,
    )
    first_unacceptable = next(
        (row for row in trace if not row["studentAcceptable"]),
        None,
    )
    first_unsafe = next(
        (row for row in trace if not row["studentImmediateSafe"]),
        None,
    )

    by_lane: dict[str, dict[str, float | int]] = {}
    lanes = sorted({str(row["currentLane"]) for row in trace})
    for lane in lanes:
        rows = [row for row in trace if row["currentLane"] == lane]
        n = len(rows)
        by_lane[lane] = {
            "frames": n,
            "exactRate": sum(
                row["studentAction"] == row["teacherAction"] for row in rows
            )
            / max(1, n),
            "acceptableRate": sum(bool(row["studentAcceptable"]) for row in rows)
            / max(1, n),
            "immediateSafeRate": sum(
                bool(row["studentImmediateSafe"]) for row in rows
            )
            / max(1, n),
        }

    semantic_rows = [row["semantic"] for row in trace]
    semantic = {
        "currentLaneAccuracy": sum(row["currentLaneCorrect"] for row in semantic_rows)
        / total,
        "nextLaneAccuracy": sum(row["nextLaneCorrect"] for row in semantic_rows)
        / total,
        "supportAccuracy": sum(row["supportCorrect"] for row in semantic_rows)
        / total,
        "trafficMse": float(np.mean([row["trafficMse"] for row in semantic_rows]))
        if semantic_rows
        else 0.0,
        "overallMse": float(np.mean([row["semanticMse"] for row in semantic_rows]))
        if semantic_rows
        else 0.0,
    }

    student_actions = Counter(row["studentAction"] for row in trace)
    teacher_actions = Counter(row["teacherAction"] for row in trace)

    def compact(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "decision": row["decision"],
            "gameStep": row["gameStep"],
            "time": row["time"],
            "row": row["row"],
            "column": row["column"],
            "score": row["score"],
            "currentLane": row["currentLane"],
            "nextLane": row["nextLane"],
            "teacherAction": row["teacherAction"],
            "studentAction": row["studentAction"],
            "acceptableActions": row["acceptableActions"],
            "immediateSafeActions": row["immediateSafeActions"],
        }

    return {
        "frames": len(trace),
        "exactActionRate": exact / total,
        "acceptableActionRate": acceptable / total,
        "immediateSafeActionRate": safe / total,
        "immediateFatalChoiceRate": fatal / total,
        "studentActions": dict(sorted(student_actions.items())),
        "teacherActions": dict(sorted(teacher_actions.items())),
        "firstExactDivergence": compact(first_exact),
        "firstUnacceptableAction": compact(first_unacceptable),
        "firstImmediateUnsafeAction": compact(first_unsafe),
        "byCurrentLane": by_lane,
        "semanticHead": semantic,
    }


@torch.no_grad()
def teacher_forced_audit(
    *,
    policy: FullMaleCNSRGBPolicy,
    semantic_head: nn.Linear,
    device: torch.device,
    seed: str,
    phase_offset: float,
    initial_column: int,
    max_steps: int,
    planner_depth: int,
    keep_trace: bool = True,
) -> dict[str, Any]:
    """Feed the student the exact teacher trajectory and measure divergence.

    The environment always advances with the planner's best action. Therefore a
    wrong student prediction cannot corrupt later states in this audit.
    """
    state = variant_state(
        seed,
        phase_offset=phase_offset,
        initial_column=initial_column,
    )
    neural = policy.zero_state(1, device=device)
    trace: list[dict[str, Any]] = []

    for decision in range(max_steps):
        if state.terminal is not None:
            break
        image = torch.as_tensor(
            resize_rgb(render_crossy_neural_frame(state))[None],
            dtype=torch.float32,
            device=device,
        )
        neural = policy.step_state(neural, image)
        logits = policy.readout(neural)[0]
        student_index = int(logits.argmax().item())

        values, best_index, safe = planner_action_preferences(
            state,
            depth=planner_depth,
        )
        acceptable = primary_acceptable_mask(values)
        student_next = step_game(state, ACTION_ORDER[student_index]).state
        teacher_next = step_game(state, ACTION_ORDER[best_index]).state
        immediate_fatal = bool(
            student_next.terminal is not None and teacher_next.terminal is None
        )

        semantic_pred = semantic_head(policy.motor_state(neural))[0].detach().cpu().numpy()
        semantic_true = semantic_target(state).astype(np.float32)
        semantic = _semantic_row(semantic_pred, semantic_true)

        target_lane = semantic["currentLaneTrue"]
        next_lane = semantic["nextLaneTrue"]
        trace.append(
            {
                "decision": decision,
                "gameStep": int(state.step),
                "time": float(state.time),
                "row": int(state.fly.row),
                "column": float(state.fly.column),
                "score": float(state.score),
                "currentLane": target_lane,
                "nextLane": next_lane,
                "teacherAction": ACTION_NAMES[best_index],
                "studentAction": ACTION_NAMES[student_index],
                "studentAcceptable": bool(acceptable[student_index]),
                "studentImmediateSafe": bool(safe[student_index]),
                "studentImmediateFatal": immediate_fatal,
                "acceptableActions": _action_names(acceptable),
                "immediateSafeActions": _action_names(safe),
                "logits": [float(value) for value in logits.detach().cpu().tolist()],
                "plannerValues": np.asarray(values, dtype=np.float32).tolist(),
                "semantic": semantic,
            }
        )

        # Critical distinction: advance with teacher, not student.
        state = teacher_next

    return {
        "mode": "teacher-forced",
        "phaseOffset": float(phase_offset),
        "initialColumn": int(initial_column),
        "teacherTerminal": state.terminal,
        "teacherFinalScore": float(state.score),
        "summary": _summarize_trace(trace),
        "trace": trace if keep_trace else None,
    }


@torch.no_grad()
def closed_loop_audit(
    *,
    policy: FullMaleCNSRGBPolicy,
    semantic_head: nn.Linear,
    device: torch.device,
    seed: str,
    phase_offset: float,
    initial_column: int,
    max_steps: int,
    planner_depth: int,
) -> dict[str, Any]:
    """Let the fly act, while the teacher labels each visited state."""
    state = variant_state(
        seed,
        phase_offset=phase_offset,
        initial_column=initial_column,
    )
    neural = policy.zero_state(1, device=device)
    trace: list[dict[str, Any]] = []

    for decision in range(max_steps):
        if state.terminal is not None:
            break
        image = torch.as_tensor(
            resize_rgb(render_crossy_neural_frame(state))[None],
            dtype=torch.float32,
            device=device,
        )
        neural = policy.step_state(neural, image)
        logits = policy.readout(neural)[0]
        student_index = int(logits.argmax().item())
        values, best_index, safe = planner_action_preferences(
            state,
            depth=planner_depth,
        )
        acceptable = primary_acceptable_mask(values)
        student_next = step_game(state, ACTION_ORDER[student_index]).state
        teacher_next = step_game(state, ACTION_ORDER[best_index]).state
        immediate_fatal = bool(
            student_next.terminal is not None and teacher_next.terminal is None
        )

        semantic_pred = semantic_head(policy.motor_state(neural))[0].detach().cpu().numpy()
        semantic_true = semantic_target(state).astype(np.float32)
        semantic = _semantic_row(semantic_pred, semantic_true)

        trace.append(
            {
                "decision": decision,
                "gameStep": int(state.step),
                "time": float(state.time),
                "row": int(state.fly.row),
                "column": float(state.fly.column),
                "score": float(state.score),
                "currentLane": semantic["currentLaneTrue"],
                "nextLane": semantic["nextLaneTrue"],
                "teacherAction": ACTION_NAMES[best_index],
                "studentAction": ACTION_NAMES[student_index],
                "studentAcceptable": bool(acceptable[student_index]),
                "studentImmediateSafe": bool(safe[student_index]),
                "studentImmediateFatal": immediate_fatal,
                "acceptableActions": _action_names(acceptable),
                "immediateSafeActions": _action_names(safe),
                "logits": [float(value) for value in logits.detach().cpu().tolist()],
                "plannerValues": np.asarray(values, dtype=np.float32).tolist(),
                "semantic": semantic,
            }
        )
        state = student_next

    return {
        "mode": "closed-loop",
        "phaseOffset": float(phase_offset),
        "initialColumn": int(initial_column),
        "steps": len(trace),
        "score": float(state.score),
        "terminalReason": state.terminal,
        "reachedStepLimit": bool(state.terminal is None and len(trace) >= max_steps),
        "summary": _summarize_trace(trace),
        "trace": trace,
    }


@torch.no_grad()
def collect_probe_variant(
    *,
    policy: FullMaleCNSRGBPolicy,
    device: torch.device,
    seed: str,
    phase_offset: float,
    initial_column: int,
    max_steps: int,
    planner_depth: int,
) -> dict[str, np.ndarray]:
    state = variant_state(
        seed,
        phase_offset=phase_offset,
        initial_column=initial_column,
    )
    neural = policy.zero_state(1, device=device)
    motor, labels, acceptable, safe, semantic = [], [], [], [], []

    for _ in range(max_steps):
        if state.terminal is not None:
            break
        image = torch.as_tensor(
            resize_rgb(render_crossy_neural_frame(state))[None],
            dtype=torch.float32,
            device=device,
        )
        neural = policy.step_state(neural, image)
        motor.append(policy.motor_state(neural)[0].detach().cpu().numpy())

        values, best_index, safe_row = planner_action_preferences(
            state,
            depth=planner_depth,
        )
        labels.append(best_index)
        acceptable.append(primary_acceptable_mask(values))
        safe.append(safe_row)
        semantic.append(semantic_target(state))
        state = step_game(state, ACTION_ORDER[best_index]).state

    return {
        "motor": np.asarray(motor, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "acceptable": np.asarray(acceptable, dtype=np.bool_),
        "safe": np.asarray(safe, dtype=np.bool_),
        "semantic": np.asarray(semantic, dtype=np.float32),
    }


def _concat_probe(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("No probe data to concatenate.")
    return {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in parts[0]
    }


def _probe_action_metrics(
    logits: torch.Tensor,
    labels: np.ndarray,
    acceptable: np.ndarray,
    safe: np.ndarray,
) -> dict[str, Any]:
    predictions = logits.argmax(dim=1).detach().cpu().numpy()
    rows = np.arange(len(predictions))
    return {
        "frames": int(len(predictions)),
        "exactActionRate": float(np.mean(predictions == labels)),
        "acceptableActionRate": float(np.mean(acceptable[rows, predictions])),
        "immediateSafeActionRate": float(np.mean(safe[rows, predictions])),
        "actions": dict(
            sorted(Counter(ACTION_NAMES[index] for index in predictions).items())
        ),
    }


def fit_motor_decoder_probe(
    *,
    train_data: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> tuple[nn.Linear, dict[str, Any], torch.Tensor, torch.Tensor]:
    """Fit ONLY a 708->5 linear readout on frozen MaleCNS motor states."""
    x_train = torch.as_tensor(train_data["motor"], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_data["labels"], dtype=torch.long, device=device)
    x_test = torch.as_tensor(test_data["motor"], dtype=torch.float32, device=device)

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-4)
    x_train_n = (x_train - mean) / std
    x_test_n = (x_test - mean) / std

    decoder = nn.Linear(x_train.shape[1], len(ACTION_ORDER), bias=True).to(device)
    counts = np.bincount(train_data["labels"], minlength=len(ACTION_ORDER)).astype(np.float32)
    total = max(1.0, float(counts.sum()))
    class_weights = np.sqrt(total / (len(ACTION_ORDER) * np.maximum(counts, 1.0)))
    class_weights[counts == 0] = 0.5
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = decoder(x_train_n)
        loss = F.cross_entropy(logits, y_train, weight=weights)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"  motor probe epoch {epoch}/{epochs} loss={history[-1]:.4f}", flush=True)

    with torch.no_grad():
        train_logits = decoder(x_train_n)
        test_logits = decoder(x_test_n)
    report = {
        "train": _probe_action_metrics(
            train_logits,
            train_data["labels"],
            train_data["acceptable"],
            train_data["safe"],
        ),
        "heldoutExact": _probe_action_metrics(
            test_logits,
            test_data["labels"],
            test_data["acceptable"],
            test_data["safe"],
        ),
        "classCounts": {
            ACTION_NAMES[index]: int(counts[index])
            for index in range(len(ACTION_ORDER))
        },
        "firstLoss": history[0],
        "lastLoss": history[-1],
    }
    return decoder, report, mean.detach(), std.detach()


def _semantic_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if len(pred) == 0:
        return {
            "frames": 0,
            "currentLaneAccuracy": 0.0,
            "nextLaneAccuracy": 0.0,
            "supportAccuracy": 0.0,
            "trafficMse": 0.0,
            "overallMse": 0.0,
        }
    return {
        "frames": int(len(pred)),
        "currentLaneAccuracy": float(
            np.mean(np.argmax(pred[:, :4], axis=1) == np.argmax(target[:, :4], axis=1))
        ),
        "nextLaneAccuracy": float(
            np.mean(
                np.argmax(pred[:, 4:8], axis=1)
                == np.argmax(target[:, 4:8], axis=1)
            )
        ),
        "supportAccuracy": float(
            np.mean((pred[:, 8] >= 0.5) == (target[:, 8] >= 0.5))
        ),
        "trafficMse": float(np.mean((pred[:, 9:] - target[:, 9:]) ** 2)),
        "overallMse": float(np.mean((pred - target) ** 2)),
    }


def fit_semantic_probe(
    *,
    train_data: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Fresh linear semantic probe; the MaleCNS stays completely frozen."""
    x_train = torch.as_tensor(train_data["motor"], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_data["semantic"], dtype=torch.float32, device=device)
    x_test = torch.as_tensor(test_data["motor"], dtype=torch.float32, device=device)

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-4)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    probe = nn.Linear(x_train.shape[1], SEMANTIC_DIM).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = probe(x_train)
        loss = F.mse_loss(prediction, y_train)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"  semantic probe epoch {epoch}/{epochs} loss={history[-1]:.4f}", flush=True)

    with torch.no_grad():
        train_pred = probe(x_train).cpu().numpy()
        test_pred = probe(x_test).cpu().numpy()
    return {
        "train": _semantic_metrics(train_pred, train_data["semantic"]),
        "heldoutExact": _semantic_metrics(test_pred, test_data["semantic"]),
        "firstLoss": history[0],
        "lastLoss": history[-1],
    }


@torch.no_grad()
def evaluate_with_probe_decoder(
    *,
    policy: FullMaleCNSRGBPolicy,
    decoder: nn.Linear,
    motor_mean: torch.Tensor,
    motor_std: torch.Tensor,
    device: torch.device,
    seed: str,
    phase_offset: float,
    initial_column: int,
    max_steps: int,
) -> dict[str, Any]:
    state = variant_state(
        seed,
        phase_offset=phase_offset,
        initial_column=initial_column,
    )
    neural = policy.zero_state(1, device=device)
    actions: Counter[str] = Counter()
    steps = 0
    while state.terminal is None and steps < max_steps:
        image = torch.as_tensor(
            resize_rgb(render_crossy_neural_frame(state))[None],
            dtype=torch.float32,
            device=device,
        )
        neural = policy.step_state(neural, image)
        motor = policy.motor_state(neural)
        logits = decoder((motor - motor_mean) / motor_std)
        action_index = int(logits[0].argmax().item())
        action = ACTION_ORDER[action_index]
        actions[action.value] += 1
        state = step_game(state, action).state
        steps += 1
    return {
        "phaseOffset": float(phase_offset),
        "initialColumn": int(initial_column),
        "steps": steps,
        "score": float(state.score),
        "reachedStepLimit": bool(state.terminal is None and steps >= max_steps),
        "terminalReason": state.terminal,
        "actions": dict(sorted(actions.items())),
    }


def evaluate_probe_suite(
    *,
    policy: FullMaleCNSRGBPolicy,
    decoder: nn.Linear,
    motor_mean: torch.Tensor,
    motor_std: torch.Tensor,
    device: torch.device,
    seed: str,
    max_steps: int,
    smoke: bool,
) -> dict[str, Any]:
    episodes = [
        evaluate_with_probe_decoder(
            policy=policy,
            decoder=decoder,
            motor_mean=motor_mean,
            motor_std=motor_std,
            device=device,
            seed=seed,
            phase_offset=phase,
            initial_column=column,
            max_steps=max_steps,
        )
        for phase, column in robustness_configs(smoke)
    ]
    exact = next(
        item
        for item in episodes
        if item["phaseOffset"] == 0.0 and item["initialColumn"] == 0
    )
    terminals = Counter(
        item["terminalReason"]
        for item in episodes
        if item["terminalReason"] is not None
    )
    scores = [item["score"] for item in episodes]
    actions: Counter[str] = Counter()
    for item in episodes:
        actions.update(item["actions"])
    return {
        "exactExpo": exact,
        "robust": {
            "episodes": len(episodes),
            "reachedStepLimit": sum(item["reachedStepLimit"] for item in episodes),
            "meanScore": float(np.mean(scores)),
            "medianScore": float(np.median(scores)),
            "terminalReasons": dict(sorted(terminals.items())),
            "actions": dict(sorted(actions.items())),
        },
        "perVariant": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V4 capacity audit. Does not retrain the 25.7M-parameter MaleCNS. "
            "Measures teacher-forced divergence and fits tiny frozen-brain "
            "linear probes to isolate readout vs representation bottlenecks."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=EXPO_SEED)
    parser.add_argument("--planner-depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=0.03)
    parser.add_argument("--flyhard-root", type=Path, default=default_flyhard_root())
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root() / "runs" / "crossy-v4-expo-specialist" / "best.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root() / "reports" / "crossy-v4-capacity-audit.json",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_steps = min(args.max_steps, 24)
        args.probe_epochs = min(args.probe_epochs, 20)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    started = time.perf_counter()
    policy, semantic_head, metadata = _load_bundle(
        checkpoint_path=args.checkpoint,
        flyhard_root=args.flyhard_root,
        device=device,
    )

    print("=== CROSSY V4 / CAPACITY AUDIT ===", flush=True)
    print(f"checkpoint: {args.checkpoint.resolve()}", flush=True)
    print(f"stage: {metadata['checkpointStage']}", flush=True)
    print("MaleCNS weights: FROZEN (no full-connectome backprop)", flush=True)

    print("\n[1] teacher-forced exact trajectory...", flush=True)
    teacher_forced = teacher_forced_audit(
        policy=policy,
        semantic_head=semantic_head,
        device=device,
        seed=args.seed,
        phase_offset=0.0,
        initial_column=0,
        max_steps=args.max_steps,
        planner_depth=args.planner_depth,
        keep_trace=True,
    )
    tf = teacher_forced["summary"]
    print(
        "  exact={:.1%} acceptable={:.1%} safe={:.1%} semantic lane={:.1%}/{:.1%}".format(
            tf["exactActionRate"],
            tf["acceptableActionRate"],
            tf["immediateSafeActionRate"],
            tf["semanticHead"]["currentLaneAccuracy"],
            tf["semanticHead"]["nextLaneAccuracy"],
        ),
        flush=True,
    )
    print(
        f"  first unacceptable: {tf['firstUnacceptableAction']}",
        flush=True,
    )

    print("\n[2] closed-loop exact trajectory with teacher labels...", flush=True)
    closed_loop = closed_loop_audit(
        policy=policy,
        semantic_head=semantic_head,
        device=device,
        seed=args.seed,
        phase_offset=0.0,
        initial_column=0,
        max_steps=args.max_steps,
        planner_depth=args.planner_depth,
    )
    print(
        f"  steps={closed_loop['steps']}/{args.max_steps} "
        f"score={closed_loop['score']:.1f} terminal={closed_loop['terminalReason']}",
        flush=True,
    )

    print("\n[3] collect frozen motor states for probe...", flush=True)
    if args.smoke:
        train_offsets = [0.4, 0.8]
    else:
        train_offsets = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    train_parts = []
    for offset in train_offsets:
        part = collect_probe_variant(
            policy=policy,
            device=device,
            seed=args.seed,
            phase_offset=offset,
            initial_column=0,
            max_steps=args.max_steps,
            planner_depth=args.planner_depth,
        )
        train_parts.append(part)
        print(f"  phase={offset:.1f}s samples={len(part['labels'])}", flush=True)
    train_data = _concat_probe(train_parts)
    test_data = collect_probe_variant(
        policy=policy,
        device=device,
        seed=args.seed,
        phase_offset=0.0,
        initial_column=0,
        max_steps=args.max_steps,
        planner_depth=args.planner_depth,
    )

    print("\n[4] tiny trainable motor decoder probe (MaleCNS frozen)...", flush=True)
    decoder, decoder_probe, motor_mean, motor_std = fit_motor_decoder_probe(
        train_data=train_data,
        test_data=test_data,
        device=device,
        epochs=args.probe_epochs,
        learning_rate=args.probe_lr,
    )
    heldout = decoder_probe["heldoutExact"]
    print(
        "  heldout exact={:.1%} acceptable={:.1%} safe={:.1%}".format(
            heldout["exactActionRate"],
            heldout["acceptableActionRate"],
            heldout["immediateSafeActionRate"],
        ),
        flush=True,
    )

    print("\n[5] fresh semantic probe (MaleCNS frozen)...", flush=True)
    semantic_probe = fit_semantic_probe(
        train_data=train_data,
        test_data=test_data,
        device=device,
        epochs=args.probe_epochs,
        learning_rate=args.probe_lr,
    )
    sem = semantic_probe["heldoutExact"]
    print(
        "  heldout currentLane={:.1%} nextLane={:.1%} support={:.1%} trafficMSE={:.4f}".format(
            sem["currentLaneAccuracy"],
            sem["nextLaneAccuracy"],
            sem["supportAccuracy"],
            sem["trafficMse"],
        ),
        flush=True,
    )

    print("\n[6] closed-loop with ONLY the fitted 708->5 decoder changed...", flush=True)
    probe_suite = evaluate_probe_suite(
        policy=policy,
        decoder=decoder,
        motor_mean=motor_mean,
        motor_std=motor_std,
        device=device,
        seed=args.seed,
        max_steps=args.max_steps,
        smoke=args.smoke,
    )
    print(
        f"  exact={probe_suite['exactExpo']['steps']}/{args.max_steps} "
        f"score={probe_suite['exactExpo']['score']:.1f} "
        f"robust={probe_suite['robust']['reachedStepLimit']}/{probe_suite['robust']['episodes']} "
        f"mean={probe_suite['robust']['meanScore']:.1f}",
        flush=True,
    )

    current_acceptable = float(tf["acceptableActionRate"])
    probe_acceptable = float(heldout["acceptableActionRate"])
    hints: list[str] = []
    if probe_acceptable >= current_acceptable + 0.15:
        hints.append(
            "A tiny learned motor decoder recovers substantially more teacher-acceptable "
            "actions from the same frozen MaleCNS state; the frozen random motor readout "
            "is a strong bottleneck candidate."
        )
    if probe_acceptable < 0.80:
        hints.append(
            "Even a learned linear motor decoder cannot recover teacher-acceptable "
            "actions reliably on the held-out exact trajectory; inspect sensory/core "
            "representation before another long policy training run."
        )
    if sem["currentLaneAccuracy"] < 0.90 or sem["nextLaneAccuracy"] < 0.90:
        hints.append(
            "A fresh held-out semantic probe does not decode lane identity reliably; "
            "the random sensory interface/core representation is a bottleneck candidate."
        )
    if not hints:
        hints.append(
            "The frozen MaleCNS state is linearly informative on these probes. If "
            "closed-loop still fails, prioritize covariate-shift/recovery training rather "
            "than changing the connectome core."
        )

    report = {
        "version": "crossy-v4-capacity-audit-1",
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "metadata": metadata,
        "config": {
            "plannerDepth": args.planner_depth,
            "maxSteps": args.max_steps,
            "probeEpochs": args.probe_epochs,
            "probeLearningRate": args.probe_lr,
            "probeTrainPhaseOffsets": train_offsets,
            "probeHeldout": {"phaseOffset": 0.0, "initialColumn": 0},
            "MaleCNSRetrained": False,
        },
        "teacherForcedExact": teacher_forced,
        "closedLoopExact": closed_loop,
        "frozenMotorDecoderProbe": decoder_probe,
        "freshSemanticProbe": semantic_probe,
        "probeDecoderClosedLoop": probe_suite,
        "diagnosticHints": hints,
        "elapsedSeconds": time.perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== AUDIT COMPLETE ===", flush=True)
    for hint in hints:
        print(f"- {hint}", flush=True)
    print(f"report: {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
