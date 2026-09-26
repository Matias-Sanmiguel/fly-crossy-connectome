from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from fly_crossy.env import create_game, step_game
from fly_crossy.expert_planner import plan_action
from fly_crossy.schema import ACTION_ORDER

from .core import FrozenMaleCNSCore
from .crossy_camera import render_crossy_neural_frame
from .decoder import decoder_logits, load_decoder
from .substrate import MaleCNSV2Substrate
from .vision_frontend import FrozenVisualFrontEnd
from .vision_geometry import VisualGeometry


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@torch.no_grad()
def evaluate_argmax(checkpoint: Path, substrate_path: Path, geometry_path: Path,
                    device: torch.device, episodes: int, max_steps: int,
                    seed_prefix: str) -> tuple[dict[str, object], dict[str, object]]:
    substrate = MaleCNSV2Substrate.load(substrate_path, strict=True)
    geometry = VisualGeometry.load(geometry_path, strict=True)
    decoder = load_decoder(checkpoint, device=device)

    if not np.array_equal(geometry.cell_body_id, substrate.body_ids[substrate.is_clamped]):
        raise RuntimeError("Visual geometry and substrate clamped order do not match.")

    frontend = FrozenVisualFrontEnd(geometry, decision_ms=200.0, internal_steps=10, device=device)
    brain = FrozenMaleCNSCore(substrate, decision_ms=200.0, internal_steps=10, tau_ms=20.0, device=device)
    gain = brain.match_rest_gain(target=0.95, iterations=80)

    scores: list[float] = []
    lengths: list[int] = []
    terminals: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    inference_ms: list[float] = []
    reached_limit = 0

    for episode in range(episodes):
        state = create_game(f"{seed_prefix}:{episode:04d}")
        vision_state = frontend.init_state(1)
        brain_state = brain.init_state(1)
        steps = 0

        while state.terminal is None and steps < max_steps:
            started = time.perf_counter()
            frame = render_crossy_neural_frame(state)
            visual_rates, vision_state = frontend.process(frame[None], vision_state)
            brain_state = brain.step(brain_state, visual_rates)
            logits = decoder_logits(decoder, brain.decision_features(brain_state))[0]
            action_index = int(torch.argmax(logits).item())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_ms.append((time.perf_counter() - started) * 1000.0)

            action = ACTION_ORDER[action_index]
            actions[action.value] += 1
            state = step_game(state, action).state
            steps += 1

        scores.append(float(state.score))
        lengths.append(steps)
        if state.terminal is None:
            reached_limit += 1
        else:
            terminals[state.terminal] += 1

    ordered = sorted(scores)
    neural = {
        "controller": "MaleCNS-V2-frozen-brain-plus-256-decoder",
        "mode": "argmax",
        "episodes": episodes,
        "maxSteps": max_steps,
        "meanScore": float(np.mean(scores)),
        "medianScore": float(ordered[len(ordered) // 2]),
        "bestScore": float(max(scores)),
        "meanLength": float(np.mean(lengths)),
        "reachedStepLimit": reached_limit,
        "terminalReasons": dict(sorted(terminals.items())),
        "actions": dict(sorted(actions.items())),
        "meanInferenceMs": float(np.mean(inference_ms)),
    }
    frozen = {"camera": True, "visualFrontend": True, "MaleCNS": True, "globalGain": gain}
    return neural, frozen


def evaluate_planner(episodes: int, max_steps: int, seed_prefix: str) -> dict[str, object]:
    scores: list[float] = []
    lengths: list[int] = []
    terminals: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    reached_limit = 0

    for episode in range(episodes):
        state = create_game(f"{seed_prefix}:{episode:04d}")
        steps = 0
        while state.terminal is None and steps < max_steps:
            decision = plan_action(state, depth=4)
            actions[decision.action.value] += 1
            state = step_game(state, decision.action).state
            steps += 1
        scores.append(float(state.score))
        lengths.append(steps)
        if state.terminal is None:
            reached_limit += 1
        else:
            terminals[state.terminal] += 1

    ordered = sorted(scores)
    return {
        "controller": "planner-reference-depth4",
        "episodes": episodes,
        "maxSteps": max_steps,
        "meanScore": float(np.mean(scores)),
        "medianScore": float(ordered[len(ordered) // 2]),
        "bestScore": float(max(scores)),
        "meanLength": float(np.mean(lengths)),
        "reachedStepLimit": reached_limit,
        "terminalReasons": dict(sorted(terminals.items())),
        "actions": dict(sorted(actions.items())),
    }


def ratios(neural: dict[str, object], planner: dict[str, object]) -> dict[str, float]:
    return {
        "meanScoreRatio": float(neural["meanScore"]) / max(float(planner["meanScore"]), 1e-9),
        "medianScoreRatio": float(neural["medianScore"]) / max(float(planner["medianScore"]), 1e-9),
        "meanLengthRatio": float(neural["meanLength"]) / max(float(planner["meanLength"]), 1e-9),
        "stepLimitRatio": float(neural["reachedStepLimit"]) / max(float(planner["reachedStepLimit"]), 1.0),
    }


def gates(relative: dict[str, float], neural: dict[str, object]) -> dict[str, bool]:
    return {
        "meanScoreAtLeast70PercentOfPlanner": relative["meanScoreRatio"] >= 0.70,
        "medianScoreAtLeast65PercentOfPlanner": relative["medianScoreRatio"] >= 0.65,
        "meanLengthAtLeast60PercentOfPlanner": relative["meanLengthRatio"] >= 0.60,
        "stepLimitAtLeast50PercentOfPlanner": relative["stepLimitRatio"] >= 0.50,
        "neuralInferenceUnder100ms": float(neural["meanInferenceMs"]) < 100.0,
    }


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Argmax-only final audit for preference R1. No sampling, mode selection, or training.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed-prefix", default="malecns-v2-decoder-final")
    parser.add_argument("--checkpoint", type=Path, default=root / "artifacts" / "malecns-crossy-v2" / "decoder-preference-r1.pt")
    parser.add_argument("--substrate", type=Path, default=root / "artifacts" / "malecns-crossy-v2" / "substrate-w5-h4.npz")
    parser.add_argument("--geometry", type=Path, default=root / "artifacts" / "malecns-crossy-v2" / "visual-geometry-160x120.npz")
    parser.add_argument("--output", type=Path, default=root / "reports" / "malecns-crossy-v2-preference-r1-argmax-final.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    print("=== Preference R1 / ARGMAX-ONLY Final Audit ===", flush=True)
    print("No sampling. No mode selection. No training.", flush=True)

    neural, frozen = evaluate_argmax(args.checkpoint, args.substrate, args.geometry, device, args.episodes, args.max_steps, args.seed_prefix)
    planner = evaluate_planner(args.episodes, args.max_steps, args.seed_prefix)
    relative = ratios(neural, planner)
    result_gates = gates(relative, neural)

    report = {
        "version": 1,
        "purpose": "Deterministic argmax audit on the exact same final seeds, bypassing survival-first mode selection.",
        "checkpoint": str(args.checkpoint.resolve()),
        "seedPrefix": args.seed_prefix,
        "frozenSystem": frozen,
        "finalArgmax": neural,
        "plannerReference": planner,
        "relativeToPlanner": relative,
        "gates": result_gates,
        "productionWeightsChanged": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"argmax: mean={neural['meanScore']:.2f} median={neural['medianScore']:.2f} length={neural['meanLength']:.2f} limit={neural['reachedStepLimit']}/{args.episodes}")
    print(f"planner: mean={planner['meanScore']:.2f} median={planner['medianScore']:.2f} length={planner['meanLength']:.2f} limit={planner['reachedStepLimit']}/{args.episodes}")
    print(f"relative: {json.dumps(relative)}")
    print(f"gates: {json.dumps(result_gates)}")
    print(f"report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
