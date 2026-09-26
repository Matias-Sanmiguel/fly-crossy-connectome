from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import time

import numpy as np
import torch

from fly_crossy.env import create_game, step_game
from fly_crossy.expert_planner import plan_action
from fly_crossy.schema import ACTION_ORDER, Action

from .core import FrozenMaleCNSCore
from .crossy_camera import render_crossy_neural_frame
from .decoder import DecoderBundle, decoder_logits, load_decoder
from .substrate import MaleCNSV2Substrate
from .vision_frontend import FrozenVisualFrontEnd
from .vision_geometry import VisualGeometry


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def stable_rng(seed: str) -> np.random.Generator:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def choose_action(
    logits: np.ndarray,
    *,
    mode: str,
    rng: np.random.Generator,
) -> Action:
    if mode == "argmax":
        return ACTION_ORDER[int(np.argmax(logits))]

    if not mode.startswith("sample:"):
        raise ValueError(f"Unknown decoder action mode: {mode}")
    temperature = float(mode.split(":", 1)[1])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Sampling temperature must be positive.")

    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max()
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    return ACTION_ORDER[int(rng.choice(len(ACTION_ORDER), p=probabilities))]


@torch.no_grad()
def evaluate_decoder(
    *,
    prefix: str,
    episodes: int,
    max_steps: int,
    mode: str,
    frontend: FrozenVisualFrontEnd,
    brain: FrozenMaleCNSCore,
    decoder: DecoderBundle,
) -> dict[str, object]:
    scores: list[float] = []
    lengths: list[int] = []
    terminals: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    reached_step_limit = 0
    inference_ms: list[float] = []

    for episode in range(episodes):
        seed = f"{prefix}:{episode:04d}"
        rng = stable_rng(f"{seed}:{mode}")
        state = create_game(seed)
        vision_state = frontend.init_state(1)
        brain_state = brain.init_state(1)
        steps = 0

        while state.terminal is None and steps < max_steps:
            started = time.perf_counter()

            frame = render_crossy_neural_frame(state)
            visual_rates, vision_state = frontend.process(frame[None], vision_state)
            brain_state = brain.step(brain_state, visual_rates)
            features = brain.decision_features(brain_state)
            logits = decoder_logits(decoder, features)[0].detach().cpu().numpy()

            if frontend.device.type == "cuda":
                torch.cuda.synchronize(frontend.device)
            inference_ms.append((time.perf_counter() - started) * 1000.0)

            action = choose_action(logits, mode=mode, rng=rng)
            actions[action.value] += 1
            state = step_game(state, action).state
            steps += 1

        scores.append(float(state.score))
        lengths.append(steps)
        if state.terminal is None:
            reached_step_limit += 1
        else:
            terminals[state.terminal] += 1

    ordered_scores = sorted(scores)
    return {
        "controller": "MaleCNS-V2-frozen-brain-plus-256-decoder",
        "mode": mode,
        "episodes": episodes,
        "maxSteps": max_steps,
        "meanScore": mean(scores),
        "medianScore": ordered_scores[len(ordered_scores) // 2],
        "bestScore": max(scores),
        "meanLength": mean(lengths),
        "reachedStepLimit": reached_step_limit,
        "terminalReasons": dict(sorted(terminals.items())),
        "actions": dict(sorted(actions.items())),
        "meanInferenceMs": mean(inference_ms) if inference_ms else 0.0,
    }


def evaluate_planner(
    *,
    prefix: str,
    episodes: int,
    max_steps: int,
) -> dict[str, object]:
    scores: list[float] = []
    lengths: list[int] = []
    terminals: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    reached_step_limit = 0

    for episode in range(episodes):
        state = create_game(f"{prefix}:{episode:04d}")
        steps = 0

        while state.terminal is None and steps < max_steps:
            decision = plan_action(state, depth=4)
            actions[decision.action.value] += 1
            state = step_game(state, decision.action).state
            steps += 1

        scores.append(float(state.score))
        lengths.append(steps)
        if state.terminal is None:
            reached_step_limit += 1
        else:
            terminals[state.terminal] += 1

    ordered_scores = sorted(scores)
    return {
        "controller": "planner-reference-depth4",
        "episodes": episodes,
        "maxSteps": max_steps,
        "meanScore": mean(scores),
        "medianScore": ordered_scores[len(ordered_scores) // 2],
        "bestScore": max(scores),
        "meanLength": mean(lengths),
        "reachedStepLimit": reached_step_limit,
        "terminalReasons": dict(sorted(terminals.items())),
        "actions": dict(sorted(actions.items())),
    }


def selection_key(metrics: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(metrics["reachedStepLimit"]),
        float(metrics["medianScore"]),
        float(metrics["meanScore"]),
        float(metrics["meanLength"]),
    )


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Closed-loop evaluation for MaleCNS Crossy V2 decoder R0. "
            "Action-mode selection and final testing use disjoint seed prefixes."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selection-episodes", type=int, default=16)
    parser.add_argument("--selection-max-steps", type=int, default=160)
    parser.add_argument("--final-episodes", type=int, default=50)
    parser.add_argument("--final-max-steps", type=int, default=200)
    parser.add_argument(
        "--modes",
        default="argmax,sample:0.8,sample:1.0,sample:1.25",
    )
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
        "--output",
        type=Path,
        default=root / "reports" / "malecns-crossy-v2-decoder-r0-closed-loop.json",
    )
    args = parser.parse_args()

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
    brain.match_rest_gain(target=0.95, iterations=80)

    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    if not modes:
        raise SystemExit("At least one action mode is required.")

    print("=== MaleCNS Crossy V2 / Decoder R0 Closed Loop ===", flush=True)
    print(f"checkpoint: {args.checkpoint.resolve()}", flush=True)
    print(f"modes: {modes}", flush=True)

    print("\n[1/3] Action-mode selection on held-out seeds...", flush=True)
    selection = {}
    for mode in modes:
        metrics = evaluate_decoder(
            prefix="malecns-v2-decoder-selection",
            episodes=args.selection_episodes,
            max_steps=args.selection_max_steps,
            mode=mode,
            frontend=frontend,
            brain=brain,
            decoder=decoder,
        )
        selection[mode] = metrics
        print(
            f"  {mode:11s}: mean={metrics['meanScore']:.2f} "
            f"median={metrics['medianScore']:.2f} "
            f"limit={metrics['reachedStepLimit']}/{args.selection_episodes} "
            f"deaths={metrics['terminalReasons']}",
            flush=True,
        )

    selected_mode = max(modes, key=lambda mode: selection_key(selection[mode]))
    print(f"  selected mode: {selected_mode}", flush=True)

    final_prefix = "malecns-v2-decoder-final"

    print("\n[2/3] Final decoder evaluation on unseen seeds...", flush=True)
    final = evaluate_decoder(
        prefix=final_prefix,
        episodes=args.final_episodes,
        max_steps=args.final_max_steps,
        mode=selected_mode,
        frontend=frontend,
        brain=brain,
        decoder=decoder,
    )
    print(json.dumps(final, indent=2), flush=True)

    print("\n[3/3] Planner reference on the EXACT same final worlds...", flush=True)
    planner = evaluate_planner(
        prefix=final_prefix,
        episodes=args.final_episodes,
        max_steps=args.final_max_steps,
    )
    print(json.dumps(planner, indent=2), flush=True)

    score_ratio = final["meanScore"] / max(float(planner["meanScore"]), 1e-9)
    median_ratio = final["medianScore"] / max(float(planner["medianScore"]), 1e-9)
    length_ratio = final["meanLength"] / max(float(planner["meanLength"]), 1e-9)
    planner_limit = max(1, int(planner["reachedStepLimit"]))
    step_limit_ratio = final["reachedStepLimit"] / planner_limit

    gates = {
        "meanScoreAtLeast70PercentOfPlanner": score_ratio >= 0.70,
        "medianScoreAtLeast65PercentOfPlanner": median_ratio >= 0.65,
        "meanLengthAtLeast60PercentOfPlanner": length_ratio >= 0.60,
        "stepLimitAtLeast50PercentOfPlanner": step_limit_ratio >= 0.50,
        "neuralInferenceUnder100ms": float(final["meanInferenceMs"]) < 100.0,
    }

    report = {
        "version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "selectionSeedPrefix": "malecns-v2-decoder-selection",
        "finalSeedPrefix": final_prefix,
        "selection": selection,
        "selectedMode": selected_mode,
        "final": final,
        "plannerReference": planner,
        "relativeToPlanner": {
            "meanScoreRatio": score_ratio,
            "medianScoreRatio": median_ratio,
            "meanLengthRatio": length_ratio,
            "stepLimitRatio": step_limit_ratio,
        },
        "gates": gates,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    print(
        f"relative to planner: score={score_ratio:.3f} "
        f"median={median_ratio:.3f} length={length_ratio:.3f} "
        f"stepLimit={step_limit_ratio:.3f}",
        flush=True,
    )
    print(f"report: {args.output.resolve()}", flush=True)

    if all(gates.values()):
        print("\nDECODER R0 CLOSED-LOOP GATE: PASS", flush=True)
        print(
            "R0 is strong enough to freeze before browser/backend integration.",
            flush=True,
        )
    else:
        print("\nDECODER R0 CLOSED-LOOP GATE: REVIEW", flush=True)
        print(json.dumps(gates, indent=2), flush=True)
        print(
            "Do not patch the architecture. If needed, the next and only "
            "learning intervention is one student-state DAgger round.",
            flush=True,
        )


if __name__ == "__main__":
    main()
