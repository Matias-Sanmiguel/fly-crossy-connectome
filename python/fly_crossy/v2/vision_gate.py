from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .core import FrozenMaleCNSCore
from .substrate import MaleCNSV2Substrate
from .vision_frontend import FrozenVisualFrontEnd
from .vision_geometry import CLAMPED_TYPES, VisualGeometry


def repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[3]


def _uniform(value: int, batch: int = 1) -> np.ndarray:
    frame = np.full((batch, 120, 160, 3), value, dtype=np.uint8)
    return frame


def _type_mask(geometry: VisualGeometry, name: str) -> np.ndarray:
    return geometry.cell_type_id == CLAMPED_TYPES.index(name)


def _mean_type(
    rates: torch.Tensor,
    geometry: VisualGeometry,
    name: str,
) -> float:
    mask = torch.as_tensor(
        _type_mask(geometry, name),
        dtype=torch.bool,
        device=rates.device,
    )
    return float(rates[:, mask].mean())


def _static_gate(frontend, geometry) -> dict[str, float]:
    rng = np.random.default_rng(1)
    gray = rng.integers(0, 256, size=(1, 120, 160, 1), dtype=np.uint8)
    frame = np.repeat(gray, 3, axis=3)
    state = frontend.init_state(1)
    out = None
    for _ in range(20):
        out, state = frontend.process(frame, state)
    assert out is not None
    last = out[-1, 0]
    return {
        "max": float(last.max()),
        "mean": float(last.mean()),
    }


def _flash_gate(frontend, geometry) -> dict[str, float]:
    state = frontend.init_state(1)
    for _ in range(8):
        _, state = frontend.process(_uniform(70), state)

    up, state = frontend.process(_uniform(210), state)
    up_rate = up[-1, 0]

    for _ in range(5):
        _, state = frontend.process(_uniform(210), state)

    down, state = frontend.process(_uniform(50), state)
    down_rate = down[-1, 0]

    return {
        "Mi1Bright": _mean_type(up_rate[None], geometry, "Mi1"),
        "Tm1Bright": _mean_type(up_rate[None], geometry, "Tm1"),
        "Mi1Dark": _mean_type(down_rate[None], geometry, "Mi1"),
        "Tm1Dark": _mean_type(down_rate[None], geometry, "Tm1"),
    }


def _moving_pair(direction: int, steps: int = 12) -> list[np.ndarray]:
    frames = []
    # Two high-contrast vertical bars move in opposite screen directions in
    # the two eye halves, producing a symmetric inward/outward optic-flow test.
    for t in range(steps):
        image = np.full((120, 160, 3), 35, dtype=np.uint8)
        frac = t / max(steps - 1, 1)

        if direction > 0:
            left_x = int(8 + frac * 62)
            right_x = int(151 - frac * 62)
        else:
            left_x = int(70 - frac * 62)
            right_x = int(89 + frac * 62)

        image[20:100, max(0, left_x - 3):min(160, left_x + 4)] = 235
        image[20:100, max(0, right_x - 3):min(160, right_x + 4)] = 235
        frames.append(image[None])
    return frames


def _motion_gate(frontend, geometry) -> dict[str, float]:
    result = {}
    for label, direction in (("inward", 1), ("outward", -1)):
        state = frontend.init_state(1)
        for _ in range(4):
            _, state = frontend.process(_uniform(35), state)

        responses = []
        for frame in _moving_pair(direction):
            rates, state = frontend.process(frame, state)
            responses.append(rates[-1, 0])

        stacked = torch.stack(responses, dim=0)
        for name in ("T4a", "T4b", "T5a", "T5b"):
            result[f"{label}_{name}"] = _mean_type(stacked, geometry, name)

    inward_pair = result["inward_T4a"] + result["inward_T4b"] + 1e-9
    outward_pair = result["outward_T4a"] + result["outward_T4b"] + 1e-9
    result["inwardT4ABSeparation"] = abs(
        result["inward_T4a"] - result["inward_T4b"]
    ) / inward_pair
    result["outwardT4ABSeparation"] = abs(
        result["outward_T4a"] - result["outward_T4b"]
    ) / outward_pair
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate the frozen visual frontend and visual->MaleCNS path."
    )
    root = repo_root()
    parser.add_argument(
        "--geometry",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "visual-geometry-160x120.npz",
    )
    parser.add_argument(
        "--substrate",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "substrate-w5-h4.npz",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark-frames", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "malecns-crossy-v2-vision-gate.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    geometry = VisualGeometry.load(args.geometry, strict=True)
    substrate = MaleCNSV2Substrate.load(args.substrate, strict=True)

    if not np.array_equal(
        geometry.cell_body_id,
        substrate.body_ids[substrate.is_clamped],
    ):
        raise SystemExit(
            "Visual geometry cell order does not exactly match frozen substrate clamped order."
        )

    frontend = FrozenVisualFrontEnd(
        geometry,
        decision_ms=200.0,
        internal_steps=10,
        device=device,
    )
    core = FrozenMaleCNSCore(
        substrate,
        decision_ms=200.0,
        internal_steps=10,
        tau_ms=20.0,
        device=device,
    )
    gain = core.match_rest_gain(target=0.95, iterations=80)

    print("=== MaleCNS Crossy V2 / Visual Frontend Gate ===", flush=True)
    print(f"gpu:      {torch.cuda.get_device_name(device) if device.type == 'cuda' else device}", flush=True)
    print(f"geometry: {geometry.n_columns:,} columns / {geometry.n_cells:,} cells", flush=True)
    print(f"brain:    {substrate.node_count:,} nodes / {substrate.edge_count:,} edges", flush=True)

    print("[1/4] Static adaptation...", flush=True)
    static = _static_gate(frontend, geometry)
    print(f"  max={static['max']:.6f} mean={static['mean']:.6f}", flush=True)

    print("[2/4] ON/OFF flash polarity...", flush=True)
    flash = _flash_gate(frontend, geometry)
    print("  " + " ".join(f"{k}={v:.5f}" for k, v in flash.items()), flush=True)

    print("[3/4] Motion direction selectivity...", flush=True)
    motion = _motion_gate(frontend, geometry)
    print(
        f"  inward T4a/T4b={motion['inward_T4a']:.5f}/{motion['inward_T4b']:.5f} "
        f"sep={motion['inwardT4ABSeparation']:.3f}",
        flush=True,
    )
    print(
        f"  outward T4a/T4b={motion['outward_T4a']:.5f}/{motion['outward_T4b']:.5f} "
        f"sep={motion['outwardT4ABSeparation']:.3f}",
        flush=True,
    )

    print("[4/4] End-to-end retinal frame -> MaleCNS benchmark...", flush=True)
    rng = np.random.default_rng(7)
    state_v = frontend.init_state(1)
    state_b = core.init_state(1)

    # Warm up temporal filters.
    for _ in range(5):
        frame = rng.integers(0, 256, size=(1, 120, 160, 3), dtype=np.uint8)
        visual, state_v = frontend.process(frame, state_v)
        state_b = core.step(state_b, visual)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    started = time.perf_counter()
    max_visual = 0.0
    mean_visual = 0.0
    for _ in range(args.benchmark_frames):
        frame = rng.integers(0, 256, size=(1, 120, 160, 3), dtype=np.uint8)
        visual, state_v = frontend.process(frame, state_v)
        max_visual = max(max_visual, float(visual.max()))
        mean_visual += float(visual.mean())
        state_b = core.step(state_b, visual)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    rates = core.rates(state_b).index_select(0, core.dynamic_idx)
    finite = bool(torch.isfinite(rates).all() and torch.isfinite(state_b.h).all())
    saturated = float((rates >= core.r_max).float().mean())
    brain_mean = float(rates.mean())
    ms = elapsed * 1000.0 / args.benchmark_frames
    mean_visual /= args.benchmark_frames

    peak_mib = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else None
    )

    gates = {
        "exactCellAlignment": True,
        "zeroTrainableVisualParameters": frontend.trainable_parameters == 0,
        "staticSceneAdapts": static["max"] < 1e-3,
        "onPathRespondsToBrightFlash": flash["Mi1Bright"] > 0.02,
        "offPathRespondsToDarkFlash": flash["Tm1Dark"] > 0.02,
        "brightFlashPrefersOnOverOff": flash["Mi1Bright"] > 3.0 * max(flash["Tm1Bright"], 1e-6),
        "darkFlashPrefersOffOverOn": flash["Tm1Dark"] > 3.0 * max(flash["Mi1Dark"], 1e-6),
        "motionChannelsAreDirectional": max(
            motion["inwardT4ABSeparation"],
            motion["outwardT4ABSeparation"],
        ) > 0.15,
        "finiteEndToEndDynamics": finite,
        "endToEndUnderHalfDecisionBudget": ms < 100.0,
        "brainNotCollapsedToSaturation": saturated < 0.05,
    }

    report = {
        "version": 1,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "geometry": frontend.summary(),
        "coreGain": gain,
        "static": static,
        "flash": flash,
        "motion": motion,
        "endToEnd": {
            "frames": args.benchmark_frames,
            "msPerDecision": ms,
            "realtimeFactorVs200ms": 200.0 / ms,
            "meanVisualRate": mean_visual,
            "maxVisualRate": max_visual,
            "meanDynamicBrainRate": brain_mean,
            "fractionBrainSaturated": saturated,
            "finite": finite,
            "peakAllocatedMiB": peak_mib,
        },
        "gates": gates,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"end-to-end ms/decision: {ms:.3f}", flush=True)
    print(f"visual mean/max:         {mean_visual:.5f} / {max_visual:.5f}", flush=True)
    print(f"brain mean/saturated:    {brain_mean:.5f} / {100*saturated:.3f}%", flush=True)
    if peak_mib is not None:
        print(f"peak VRAM:               {peak_mib:.1f} MiB", flush=True)
    print(f"report:                  {args.output.resolve()}", flush=True)

    if all(gates.values()):
        print("\nVISUAL FRONTEND GATE: PASS", flush=True)
    else:
        print("\nVISUAL FRONTEND GATE: REVIEW", flush=True)
        print(json.dumps(gates, indent=2), flush=True)


if __name__ == "__main__":
    main()
