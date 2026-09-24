from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .core import FrozenMaleCNSCore
from .substrate import MaleCNSV2Substrate


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "peakAllocatedMiB": None,
            "peakReservedMiB": None,
        }
    return {
        "peakAllocatedMiB": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peakReservedMiB": torch.cuda.max_memory_reserved(device) / (1024**2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the frozen 138,968-cell MaleCNS Crossy V2 core."
    )
    parser.add_argument(
        "--substrate",
        type=Path,
        default=Path("../artifacts/malecns-crossy-v2/substrate-w5-h4.npz"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decisions", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--gain-iterations", type=int, default=80)
    parser.add_argument("--gain-target", type=float, default=0.95)
    parser.add_argument("--clamped-level", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../reports/malecns-crossy-v2-core-benchmark.json"),
    )
    args = parser.parse_args()

    if args.decisions <= 0 or args.warmup < 0:
        raise SystemExit("decisions must be positive and warmup nonnegative.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")

    print("=== MaleCNS Crossy V2 / Frozen Core Gate ===", flush=True)
    print(f"substrate: {args.substrate.resolve()}", flush=True)
    print(f"device:    {device}", flush=True)
    if device.type == "cuda":
        print(f"gpu:       {torch.cuda.get_device_name(device)}", flush=True)

    substrate = MaleCNSV2Substrate.load(args.substrate, strict=True)
    print(
        f"graph:     {substrate.node_count:,} nodes / "
        f"{substrate.edge_count:,} edges / "
        f"{substrate.decision_readout_count:,} DN+VPN readout",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    core = FrozenMaleCNSCore(
        substrate,
        decision_ms=200.0,
        internal_steps=10,
        tau_ms=20.0,
        r_max=5.0,
        device=device,
    )
    _sync(device)
    build_seconds = time.perf_counter() - t0

    print(
        f"core:      K={core.internal_steps} dt={core.decision_ms/core.internal_steps:.1f}ms "
        f"tau={core.tau_ms:.1f}ms",
        flush=True,
    )
    print(
        f"centring:  inhibitory factor={core.inhibitory_factor:.6f}",
        flush=True,
    )
    print(
        f"trainable: {sum(p.numel() for p in core.parameters()):,} parameters",
        flush=True,
    )

    print("[1/3] Matching frozen recurrent gain at rest...", flush=True)
    t_gain = time.perf_counter()
    gain = core.match_rest_gain(
        target=args.gain_target,
        iterations=args.gain_iterations,
        seed=0,
    )
    _sync(device)
    gain_seconds = time.perf_counter() - t_gain
    print(
        f"  baseGain={gain['baseGainAtW0One']:.6f} "
        f"w0={gain['w0']:.8f} target={gain['target']:.3f} "
        f"({gain_seconds:.2f}s)",
        flush=True,
    )

    clamp = torch.full(
        (1, core.n_clamped),
        float(args.clamped_level),
        dtype=torch.float32,
        device=device,
    )
    state = core.init_state(1)

    print("[2/3] Warmup...", flush=True)
    with torch.inference_mode():
        for _ in range(args.warmup):
            state = core.step(state, clamp)
    _sync(device)

    print("[3/3] Timed recurrent decisions...", flush=True)
    t_run = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.decisions):
            state = core.step(state, clamp)
    _sync(device)
    elapsed = time.perf_counter() - t_run

    rates = core.rates(state)
    dynamic_rates = rates.index_select(0, core.dynamic_idx)
    finite = bool(torch.isfinite(state.h).all() and torch.isfinite(rates).all())
    mean_rate = float(dynamic_rates.mean())
    saturated = float((dynamic_rates >= core.r_max).float().mean())
    near_silent = float((dynamic_rates < 0.05).float().mean())
    feature_shape = list(core.decision_features(state).shape)

    ms_per_decision = elapsed * 1000.0 / args.decisions
    ms_per_substep = ms_per_decision / core.internal_steps
    realtime_factor = core.decision_ms / ms_per_decision
    # Reserve at least half of the 200 ms game-decision budget for the future
    # front end, decoder, rendering and browser/server overhead.
    realtime_headroom_pass = ms_per_decision <= 100.0

    report = {
        "version": 1,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "substrate": str(args.substrate.resolve()),
        "core": core.summary(),
        "buildSeconds": build_seconds,
        "gainCalibrationSeconds": gain_seconds,
        "gain": gain,
        "benchmark": {
            "decisions": args.decisions,
            "warmup": args.warmup,
            "clampedLevel": args.clamped_level,
            "elapsedSeconds": elapsed,
            "msPerDecision": ms_per_decision,
            "msPerSubstep": ms_per_substep,
            "realtimeFactorVs200ms": realtime_factor,
            "finite": finite,
            "meanDynamicRate": mean_rate,
            "fractionSaturated": saturated,
            "fractionBelow005": near_silent,
            "decisionFeatureShape": feature_shape,
        },
        "memory": _memory(device),
        "gates": {
            "zeroTrainableCoreParameters": sum(p.numel() for p in core.parameters()) == 0,
            "finiteDynamics": finite,
            "realtimeWithHalfBudgetReserved": realtime_headroom_pass,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"ms/decision: {ms_per_decision:.3f}", flush=True)
    print(f"ms/substep:  {ms_per_substep:.3f}", flush=True)
    print(f"real-time:   {realtime_factor:.2f}x vs 200 ms budget", flush=True)
    print(f"mean rate:   {mean_rate:.4f}", flush=True)
    print(f"saturated:   {100*saturated:.3f}%", flush=True)
    print(f"silent<.05:  {100*near_silent:.3f}%", flush=True)
    print(f"features:    {feature_shape}", flush=True)
    if device.type == "cuda":
        mem = report["memory"]
        print(
            f"peak VRAM:   {mem['peakAllocatedMiB']:.1f} MiB allocated / "
            f"{mem['peakReservedMiB']:.1f} MiB reserved",
            flush=True,
        )
    print(f"report:      {args.output.resolve()}", flush=True)

    if all(report["gates"].values()):
        print("\nFROZEN CORE GATE: PASS", flush=True)
    else:
        print("\nFROZEN CORE GATE: REVIEW", flush=True)
        print(json.dumps(report["gates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
