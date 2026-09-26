from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from fly_crossy.expert_planner import evaluate_expert
from .core import FrozenMaleCNSCore
from .decoder import load_decoder
from .preference_distill import evaluate_closed_loop
from .substrate import MaleCNSV2Substrate
from .vision_frontend import FrozenVisualFrontEnd
from .vision_geometry import VisualGeometry

def repo_root():
    return Path(__file__).resolve().parents[3]

def ratios(controller, planner):
    return {
        "meanScoreRatio": float(controller["meanScore"]) / float(planner["meanScore"]),
        "medianScoreRatio": float(controller["medianScore"]) / float(planner["medianScore"]),
        "meanLengthRatio": float(controller["meanLength"]) / float(planner["meanLength"]),
        "stepLimitRatio": float(controller["reachedStepLimit"]) / max(1.0, float(planner["reachedStepLimit"])),
    }

def main():
    root = repo_root()
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint", type=Path, default=root/"artifacts"/"malecns-crossy-v2"/"decoder-preference-r1.pt")
    p.add_argument("--substrate", type=Path, default=root/"artifacts"/"malecns-crossy-v2"/"substrate-w5-h4.npz")
    p.add_argument("--geometry", type=Path, default=root/"artifacts"/"malecns-crossy-v2"/"visual-geometry-160x120.npz")
    p.add_argument("--output", type=Path, default=root/"reports"/"malecns-crossy-v2-preference-r1-argmax-final.json")
    args = p.parse_args()

    device = torch.device(args.device)
    substrate = MaleCNSV2Substrate.load(args.substrate, strict=True)
    geometry = VisualGeometry.load(args.geometry, strict=True)
    decoder = load_decoder(args.checkpoint, device=device)

    if not np.array_equal(geometry.cell_body_id, substrate.body_ids[substrate.is_clamped]):
        raise SystemExit("Visual geometry/substrate clamped order mismatch.")

    frontend = FrozenVisualFrontEnd(geometry, decision_ms=200.0, internal_steps=10, device=device)
    brain = FrozenMaleCNSCore(substrate, decision_ms=200.0, internal_steps=10, tau_ms=20.0, device=device)
    gain = brain.match_rest_gain(target=0.95, iterations=80)

    print("=== Preference R1 ARGMAX final audit ===")
    print("NO TRAINING. NO WEIGHT CHANGES.")

    selection = evaluate_closed_loop(
        decoder.model, mean=decoder.mean, std=decoder.std,
        frontend=frontend, brain=brain,
        episodes=16, max_steps=160,
        seed_prefix="malecns-v2-decoder-selection",
    )
    expected = {"meanScore":29.125,"medianScore":25.0,"meanLength":43.75,"reachedStepLimit":0}
    parity = (
        abs(float(selection["meanScore"])-expected["meanScore"]) < 1e-9 and
        abs(float(selection["medianScore"])-expected["medianScore"]) < 1e-9 and
        abs(float(selection["meanLength"])-expected["meanLength"]) < 1e-9 and
        int(selection["reachedStepLimit"]) == 0
    )
    print("selection parity:", parity, selection)
    if not parity:
        raise SystemExit("Selection parity failed; stop.")

    final = evaluate_closed_loop(
        decoder.model, mean=decoder.mean, std=decoder.std,
        frontend=frontend, brain=brain,
        episodes=50, max_steps=200,
        seed_prefix="malecns-v2-decoder-final",
    )
    planner = evaluate_expert(
        episodes=50, depth=4, max_steps=200,
        seed_prefix="malecns-v2-decoder-final",
    )
    relative = ratios(final, planner)

    report = {
        "version":1,
        "checkpoint":str(args.checkpoint.resolve()),
        "purpose":"Remove stochastic mode-selection confound and evaluate deterministic argmax on the exact same 50 final worlds.",
        "productionWeightsChanged":False,
        "frozenSystem":{"camera":True,"visualFrontend":True,"MaleCNS":True,"globalGain":gain},
        "selectionArgmaxParity":{"passed":parity,"observed":selection,"expected":expected},
        "finalArgmax":final,
        "plannerReference":planner,
        "relativeToPlanner":relative,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")

    print("\nFINAL ARGMAX:", json.dumps(final, indent=2))
    print("\nPLANNER:", json.dumps(planner, indent=2))
    print("\nRELATIVE:", json.dumps(relative, indent=2))
    print("\nreport:", args.output.resolve())

if __name__ == "__main__":
    main()
