from __future__ import annotations

import argparse, json, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from fly_crossy.env import create_game, step_game
from fly_crossy.schema import ACTION_ORDER
from .core import FrozenMaleCNSCore
from .crossy_camera import render_crossy_neural_frame
from .decoder import load_decoder
from .substrate import MaleCNSV2Substrate
from .vision_frontend import FrozenVisualFrontEnd
from .vision_geometry import VisualGeometry


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@torch.no_grad()
def evaluate(checkpoint: Path, substrate_path: Path, geometry_path: Path, device: torch.device,
             episodes: int, max_steps: int, seed_prefix: str) -> dict[str, object]:
    substrate = MaleCNSV2Substrate.load(substrate_path, strict=True)
    geometry = VisualGeometry.load(geometry_path, strict=True)
    decoder = load_decoder(checkpoint, device=device)

    frontend = FrozenVisualFrontEnd(geometry, decision_ms=200.0, internal_steps=10, device=device)
    brain = FrozenMaleCNSCore(substrate, decision_ms=200.0, internal_steps=10, tau_ms=20.0, device=device)
    brain.match_rest_gain(target=0.95, iterations=80)

    scores, lengths, latency = [], [], []
    terminals, actions = Counter(), Counter()
    reached = 0

    for ep in range(episodes):
        state = create_game(f"{seed_prefix}:{ep:04d}")
        vision_state = frontend.init_state(1)
        brain_state = brain.init_state(1)
        steps = 0
        while state.terminal is None and steps < max_steps:
            t0 = time.perf_counter()
            frame = render_crossy_neural_frame(state)
            visual_rates, vision_state = frontend.process(frame[None], vision_state)
            brain_state = brain.step(brain_state, visual_rates)
            feature = brain.decision_features(brain_state)[0]
            logits = decoder.model(((feature - decoder.mean) / decoder.std)[None])[0]
            idx = int(torch.argmax(logits).item())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency.append((time.perf_counter() - t0) * 1000.0)
            action = ACTION_ORDER[idx]
            actions[action.value] += 1
            state = step_game(state, action).state
            steps += 1
        scores.append(float(state.score)); lengths.append(steps)
        if state.terminal is None: reached += 1
        else: terminals[state.terminal] += 1

    ordered = sorted(scores)
    return {
        "episodes": episodes, "maxSteps": max_steps, "mode": "argmax-forced",
        "seedPrefix": seed_prefix, "meanScore": float(np.mean(scores)),
        "medianScore": float(ordered[len(ordered)//2]), "bestScore": float(max(scores)),
        "meanLength": float(np.mean(lengths)), "reachedStepLimit": reached,
        "terminalReasons": dict(sorted(terminals.items())),
        "actions": dict(sorted(actions.items())), "meanInferenceMs": float(np.mean(latency)),
    }


def main() -> None:
    root = repo_root()
    p = argparse.ArgumentParser(description="Force argmax on the exact 50 final preference-R1 worlds.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed-prefix", default="malecns-v2-decoder-final")
    p.add_argument("--checkpoint", type=Path, default=root/"artifacts"/"malecns-crossy-v2"/"decoder-preference-r1.pt")
    p.add_argument("--substrate", type=Path, default=root/"artifacts"/"malecns-crossy-v2"/"substrate-w5-h4.npz")
    p.add_argument("--geometry", type=Path, default=root/"artifacts"/"malecns-crossy-v2"/"visual-geometry-160x120.npz")
    p.add_argument("--output", type=Path, default=root/"reports"/"malecns-crossy-v2-preference-r1-final-argmax.json")
    a = p.parse_args()
    device = torch.device(a.device)
    result = evaluate(a.checkpoint, a.substrate, a.geometry, device, a.episodes, a.max_steps, a.seed_prefix)
    report = {"version":1,"purpose":"Force argmax on the same final 50 seeds; no training.","checkpoint":str(a.checkpoint.resolve()),"final":result}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2)); print(f"report: {a.output.resolve()}")

if __name__ == "__main__":
    main()
