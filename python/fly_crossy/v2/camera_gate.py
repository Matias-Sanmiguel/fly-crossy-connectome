from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from fly_crossy.env import (
    GameState,
    create_game,
    generate_rows,
    step_game,
)
from fly_crossy.expert_planner import plan_action

from .core import FrozenMaleCNSCore
from .crossy_camera import (
    CAMERA_HEIGHT,
    CAMERA_PITCH_DEG,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HFOV_DEG,
    LANE_COLORS,
    frame_sha256,
    project_world_point,
    render_crossy_neural_frame,
    write_bmp,
)
from .substrate import MaleCNSV2Substrate
from .vision_frontend import FrozenVisualFrontEnd
from .vision_geometry import VisualGeometry


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _find_seed_with_lane(kind: str, *, maximum: int = 500) -> tuple[str, int]:
    for index in range(maximum):
        seed = f"malecns-v2-{kind}-sample:{index:04d}"
        state = create_game(seed)
        candidates = [
            lane.row
            for lane in state.lanes
            if lane.kind == kind and 2 <= lane.row <= 7
        ]
        if candidates:
            return seed, min(candidates)
    raise RuntimeError(f"Could not find a {kind} sample seed.")


def _sample_state(kind: str) -> GameState:
    seed, target_row = _find_seed_with_lane(kind)
    state = create_game(seed)

    # Keep the requested mechanic 2-4 rows ahead so perspective remains obvious
    # while the state still uses real generated lanes and hazard timing.
    fly_row = max(0, target_row - 3)
    lanes = generate_rows(seed, fly_row - 8, 24)
    return replace(
        state,
        fly=replace(state.fly, row=fly_row, column=0.0),
        lanes=lanes,
        score=float(fly_row),
    )


def _perspective_compression(state: GameState) -> dict[str, float]:
    # Compare projected screen height of a one-row ground interval near/far.
    near_a = project_world_point(state, 0.0, 0.0, state.fly.row + 1.0)
    near_b = project_world_point(state, 0.0, 0.0, state.fly.row + 2.0)
    far_a = project_world_point(state, 0.0, 0.0, state.fly.row + 9.0)
    far_b = project_world_point(state, 0.0, 0.0, state.fly.row + 10.0)
    if any(value is None for value in (near_a, near_b, far_a, far_b)):
        raise RuntimeError("Perspective compression points unexpectedly fell behind camera.")
    near_height = abs(near_a[1] - near_b[1])
    far_height = abs(far_a[1] - far_b[1])
    return {
        "nearOneRowPixels": float(near_height),
        "farOneRowPixels": float(far_height),
        "farToNearRatio": float(far_height / max(near_height, 1e-9)),
    }


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Gate final perspective Crossy neural camera -> retina -> MaleCNS."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", default="malecns-v2-perspective-camera-gate")
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
        default=root / "reports" / "malecns-crossy-v2-perspective-camera-gate.json",
    )
    args = parser.parse_args()

    if args.steps < 5:
        raise SystemExit("--steps must be at least 5.")

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
    brain.match_rest_gain(target=0.95, iterations=80)

    print("=== MaleCNS Crossy V2 / Perspective Neural Camera Gate ===", flush=True)
    print(
        f"camera: {FRAME_WIDTH}x{FRAME_HEIGHT} RGB / HFOV={HFOV_DEG:.1f}deg / "
        f"height={CAMERA_HEIGHT:.2f} / pitch={CAMERA_PITCH_DEG:.1f}deg",
        flush=True,
    )
    print(f"device: {device}", flush=True)

    sample_dir = root / "reports" / "malecns-crossy-v2-perspective-samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Representative road / rail / river frames...", flush=True)
    representative_hashes = {}
    representative_colors = {}
    representative_lane_pixels = {}
    for kind in ("road", "rail", "river"):
        sample_state = _sample_state(kind)
        frame = render_crossy_neural_frame(sample_state)
        write_bmp(sample_dir / f"{kind}.bmp", frame)
        representative_hashes[kind] = frame_sha256(frame)
        representative_colors[kind] = int(len(np.unique(frame.reshape(-1, 3), axis=0)))
        lane_color = np.asarray(LANE_COLORS[kind], dtype=np.uint8)
        representative_lane_pixels[kind] = int(
            np.all(frame == lane_color[None, None, :], axis=2).sum()
        )
        print(
            f"  {kind}: uniqueColors={representative_colors[kind]} "
            f"lanePixels={representative_lane_pixels[kind]} "
            f"sha={representative_hashes[kind][:12]}...",
            flush=True,
        )

    state = create_game(args.seed)
    same_a = render_crossy_neural_frame(state)
    same_b = render_crossy_neural_frame(state)
    deterministic = bool(np.array_equal(same_a, same_b))

    shifted = replace(
        state,
        fly=replace(state.fly, column=state.fly.column + 1.0),
    )
    lateral_changes = not np.array_equal(
        same_a,
        render_crossy_neural_frame(shifted),
    )

    compression = _perspective_compression(state)
    perspective_present = (
        compression["farOneRowPixels"] > 0.0
        and compression["farToNearRatio"] < 0.55
    )

    print("[2/4] Perspective invariants...", flush=True)
    print(f"  deterministic same state: {deterministic}", flush=True)
    print(f"  lateral shift changes image: {lateral_changes}", flush=True)
    print(
        f"  near row={compression['nearOneRowPixels']:.3f}px "
        f"far row={compression['farOneRowPixels']:.3f}px "
        f"ratio={compression['farToNearRatio']:.3f}",
        flush=True,
    )

    print("[3/4] Planner trajectory -> perspective frames...", flush=True)
    state = create_game(args.seed)
    v_state = frontend.init_state(1)
    b_state = brain.init_state(1)

    hashes: list[str] = []
    actions: Counter[str] = Counter()
    render_ms: list[float] = []
    retina_brain_ms: list[float] = []
    feature_norms: list[float] = []

    for _ in range(args.steps):
        started = time.perf_counter()
        frame = render_crossy_neural_frame(state)
        render_ms.append((time.perf_counter() - started) * 1000.0)
        hashes.append(frame_sha256(frame))

        decision = plan_action(state, depth=4)
        actions[decision.action.value] += 1

        started = time.perf_counter()
        visual, v_state = frontend.process(frame[None], v_state)
        b_state = brain.step(b_state, visual)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        retina_brain_ms.append((time.perf_counter() - started) * 1000.0)

        feature_norms.append(float(torch.linalg.vector_norm(brain.decision_features(b_state))))

        next_state = step_game(state, decision.action).state
        if next_state.terminal is not None:
            break
        state = next_state

    unique_fraction = len(set(hashes)) / max(1, len(hashes))
    finite_features = bool(np.isfinite(feature_norms).all())
    nonzero_features = (
        sum(value > 1e-6 for value in feature_norms) / max(1, len(feature_norms))
    )

    mean_render_ms = float(np.mean(render_ms))
    mean_retina_brain_ms = float(np.mean(retina_brain_ms))
    mean_feature_norm = float(np.mean(feature_norms))

    print("[4/4] End-to-end metrics...", flush=True)
    print(f"  unique frame fraction: {unique_fraction:.3f}", flush=True)
    print(f"  render mean: {mean_render_ms:.3f} ms", flush=True)
    print(f"  retina+brain mean: {mean_retina_brain_ms:.3f} ms", flush=True)
    print(f"  mean feature norm: {mean_feature_norm:.3f}", flush=True)

    gates = {
        "deterministicSameStateSamePixels": deterministic,
        "lateralPositionVisuallyObservable": lateral_changes,
        "truePerspectiveRowCompression": perspective_present,
        "roadSampleRendered": (
            representative_colors["road"] >= 6
            and representative_lane_pixels["road"] >= 20
        ),
        "railSampleRendered": (
            representative_colors["rail"] >= 6
            and representative_lane_pixels["rail"] >= 20
        ),
        "riverSampleRendered": (
            representative_colors["river"] >= 6
            and representative_lane_pixels["river"] >= 20
        ),
        "trajectoryProducesChangingFrames": unique_fraction >= 0.80,
        "retinaBrainFeaturesFinite": finite_features,
        "retinaBrainFeaturesNonzero": nonzero_features >= 0.80,
        "cameraRendererUnder50ms": mean_render_ms < 50.0,
        "retinaBrainUnder100ms": mean_retina_brain_ms < 100.0,
    }

    report = {
        "version": 2,
        "sensorContract": {
            "kind": "fixed-perspective-crossy-neural-camera",
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "hfovDeg": HFOV_DEG,
            "cameraHeight": CAMERA_HEIGHT,
            "pitchDeg": CAMERA_PITCH_DEG,
            "cameraOrigin": "current fly position",
            "source": "authoritative GameState only",
            "presentationUiDependency": False,
            "observationV4Dependency": False,
            "plannerInformationInPixels": False,
        },
        "representativeSamples": {
            "directory": str(sample_dir.resolve()),
            "hashes": representative_hashes,
            "uniqueColors": representative_colors,
            "exactLaneColorPixels": representative_lane_pixels,
        },
        "perspective": compression,
        "trajectory": {
            "seed": args.seed,
            "stepsRendered": len(hashes),
            "actionCounts": dict(sorted(actions.items())),
            "uniqueFrameFraction": unique_fraction,
            "meanRenderMs": mean_render_ms,
            "meanRetinaBrainMs": mean_retina_brain_ms,
            "meanBrainFeatureNorm": mean_feature_norm,
        },
        "gates": gates,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\nreport:  {args.output.resolve()}", flush=True)
    print(f"samples: {sample_dir.resolve()}", flush=True)

    if all(gates.values()):
        print("\nPERSPECTIVE NEURAL CAMERA GATE: PASS", flush=True)
    else:
        print("\nPERSPECTIVE NEURAL CAMERA GATE: REVIEW", flush=True)
        print(json.dumps(gates, indent=2), flush=True)


if __name__ == "__main__":
    main()
