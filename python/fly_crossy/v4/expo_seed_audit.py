from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from fly_crossy.env import (
    DECISION_SECONDS,
    GameState,
    create_game,
    generate_rows,
    step_game,
)
from fly_crossy.expert_planner import plan_action
from fly_crossy.schema import Action
from fly_crossy.v2.crossy_camera import (
    render_crossy_neural_frame,
    write_bmp,
)


REQUIRED_KINDS = ("road", "rail", "river")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def lane_profile(seed: str, *, rows: int) -> dict[str, Any]:
    lanes = generate_rows(seed, 0, rows)
    kind_counts = Counter(lane.kind for lane in lanes)
    hazard_counts = Counter(hazard.kind for lane in lanes for hazard in lane.hazards)
    first_row: dict[str, int | None] = {}
    for kind in ("grass", *REQUIRED_KINDS):
        first_row[kind] = next((lane.row for lane in lanes if lane.kind == kind), None)
    return {
        "rowsInspected": rows,
        "laneCounts": dict(sorted(kind_counts.items())),
        "hazardCounts": dict(sorted(hazard_counts.items())),
        "firstRowByKind": first_row,
        "hasRoadRailRiver": all(kind_counts[kind] > 0 for kind in REQUIRED_KINDS),
    }


def grayscale_rgb(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame, dtype=np.float32)
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    gray = np.clip(np.rint(gray), 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _lane_kind(state: GameState, row: int) -> str | None:
    lane = next((lane for lane in state.lanes if lane.row == row), None)
    if lane is None:
        lane = generate_rows(state.seed, row, 1)[0]
    return lane.kind


def rollout_expert(
    seed: str,
    *,
    depth: int,
    max_steps: int,
    time_offset: float = 0.0,
    keep_trace: bool = False,
) -> dict[str, Any]:
    state = create_game(seed)
    if time_offset:
        state = replace(
            state,
            time=float(time_offset),
            step=int(round(time_offset / DECISION_SECONDS)),
        )

    actions: Counter[str] = Counter()
    lane_visits: Counter[str] = Counter()
    trace: list[dict[str, Any]] = []
    total_nodes = 0
    total_cache_hits = 0

    decisions = 0
    while state.terminal is None and decisions < max_steps:
        current_lane = _lane_kind(state, state.fly.row)
        next_lane = _lane_kind(state, state.fly.row + 1)
        decision = plan_action(state, depth=depth)
        result = step_game(state, decision.action)

        actions[decision.action.value] += 1
        lane_visits[current_lane or "unknown"] += 1
        total_nodes += decision.nodes_expanded
        total_cache_hits += decision.cache_hits

        if keep_trace:
            trace.append(
                {
                    "decision": decisions,
                    "gameStep": state.step,
                    "time": state.time,
                    "row": state.fly.row,
                    "column": state.fly.column,
                    "score": state.score,
                    "currentLane": current_lane,
                    "nextLane": next_lane,
                    "action": decision.action.value,
                    "principalVariation": [a.value for a in decision.principal_variation],
                    "plannerValue": list(decision.value),
                    "events": jsonable(result.events),
                    "terminalAfterAction": result.state.terminal,
                }
            )

        state = result.state
        decisions += 1

    non_forward = decisions - actions.get("forward", 0)
    return {
        "seed": seed,
        "depth": depth,
        "timeOffset": time_offset,
        "maxSteps": max_steps,
        "steps": decisions,
        "score": float(state.score),
        "finalRow": int(state.fly.row),
        "finalColumn": float(state.fly.column),
        "reachedStepLimit": state.terminal is None and decisions >= max_steps,
        "terminalReason": state.terminal,
        "actions": dict(sorted(actions.items())),
        "distinctActions": len(actions),
        "nonForwardActions": int(non_forward),
        "laneVisits": dict(sorted(lane_visits.items())),
        "meanSearchNodesPerDecision": total_nodes / max(1, decisions),
        "meanCacheHitsPerDecision": total_cache_hits / max(1, decisions),
        "trace": trace if keep_trace else None,
    }


def candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    rollout = candidate["rollout"]
    actions = rollout["actions"]
    return (
        bool(rollout["reachedStepLimit"]),
        int("wait" in actions),
        int(rollout["distinctActions"]),
        int(rollout["nonForwardActions"]),
        float(rollout["score"]),
    )


def choose_snapshot_steps(trace: list[dict[str, Any]], limit: int) -> list[int]:
    if not trace:
        return []

    selected: list[int] = []
    previous_lane: str | None = None
    for item in trace:
        lane = item["currentLane"]
        interesting = (
            item["action"] != "forward"
            or lane != previous_lane
            or item["terminalAfterAction"] is not None
            or any(event.get("type") == "train-warning" for event in item["events"])
        )
        if interesting:
            selected.append(int(item["decision"]))
        previous_lane = lane

    if len(trace) > 1:
        for step in np.linspace(0, len(trace) - 1, min(6, len(trace)), dtype=int):
            selected.append(int(step))

    deduped = sorted(set(selected))
    if len(deduped) <= limit:
        return deduped
    indexes = np.linspace(0, len(deduped) - 1, limit, dtype=int)
    return [deduped[int(index)] for index in indexes]


def write_visual_audit(
    seed: str,
    trace: list[dict[str, Any]],
    *,
    output_dir: Path,
    snapshot_limit: int,
) -> list[dict[str, Any]]:
    color_dir = output_dir / "frames" / "rgb"
    gray_dir = output_dir / "frames" / "v3-grayscale"
    color_dir.mkdir(parents=True, exist_ok=True)
    gray_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(choose_snapshot_steps(trace, snapshot_limit))
    state = create_game(seed)
    samples: list[dict[str, Any]] = []

    for decision_index, item in enumerate(trace):
        if decision_index in wanted:
            frame = render_crossy_neural_frame(state)
            lane = item["currentLane"] or "unknown"
            action = item["action"]
            stem = f"step-{decision_index:03d}-{lane}-{action}"
            rgb_path = color_dir / f"{stem}.bmp"
            gray_path = gray_dir / f"{stem}.bmp"
            write_bmp(rgb_path, frame)
            write_bmp(gray_path, grayscale_rgb(frame))
            samples.append(
                {
                    "decision": decision_index,
                    "lane": lane,
                    "action": action,
                    "rgb": str(rgb_path.resolve()),
                    "v3Grayscale": str(gray_path.resolve()),
                }
            )

        state = step_game(state, Action(item["action"])).state
        if state.terminal is not None:
            break

    return samples


def search_expo_seed(
    *,
    prefix: str,
    candidates: int,
    inspect_rows: int,
    depth: int,
    max_steps: int,
    target_passes: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    evaluated: list[dict[str, Any]] = []
    passes = 0

    for index in range(candidates):
        seed = f"{prefix}:{index:04d}"
        profile = lane_profile(seed, rows=inspect_rows)
        if not profile["hasRoadRailRiver"]:
            continue

        rollout = rollout_expert(seed, depth=depth, max_steps=max_steps)
        candidate = {"seed": seed, "profile": profile, "rollout": rollout}
        evaluated.append(candidate)

        status = "PASS" if rollout["reachedStepLimit"] else "fail"
        print(
            f"{status:4s} {seed} steps={rollout['steps']:3d} "
            f"score={rollout['score']:5.1f} actions={rollout['actions']}",
            flush=True,
        )

        if rollout["reachedStepLimit"]:
            passes += 1
            if passes >= target_passes:
                break

    if not evaluated:
        return None, []

    evaluated.sort(key=candidate_key, reverse=True)
    selected = next(
        (candidate for candidate in evaluated if candidate["rollout"]["reachedStepLimit"]),
        None,
    )
    return selected, evaluated


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find one visually interesting Crossy seed that the authoritative "
            "planner can actually master before V4 expo-specialist training."
        )
    )
    parser.add_argument("--prefix", default="crossy-v4-expo")
    parser.add_argument("--candidates", type=int, default=250)
    parser.add_argument("--inspect-rows", type=int, default=42)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--target-passes", type=int, default=3)
    parser.add_argument("--snapshot-limit", type=int, default=18)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "reports" / "crossy-v4-expo-seed-audit.json",
    )
    args = parser.parse_args()

    print("=== CROSSY V4 / EXPO SEED GATE ===", flush=True)
    print(
        "Requirement: road + rail + river early, then planner must survive "
        f"{args.max_steps} decisions.",
        flush=True,
    )
    print("This command does NO neural training and does not use reward shaping.", flush=True)

    selected, evaluated = search_expo_seed(
        prefix=args.prefix,
        candidates=args.candidates,
        inspect_rows=args.inspect_rows,
        depth=args.depth,
        max_steps=args.max_steps,
        target_passes=args.target_passes,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if selected is None:
        report = {
            "version": "crossy-v4-expo-seed-gate-1",
            "status": "NO_PASS",
            "search": {
                "prefix": args.prefix,
                "candidates": args.candidates,
                "inspectRows": args.inspect_rows,
                "depth": args.depth,
                "maxSteps": args.max_steps,
            },
            "evaluated": evaluated,
            "recommendation": (
                "Do not train MaleCNS. Increase --candidates first; if the planner "
                "repeatedly fails, audit or deepen the teacher before V4."
            ),
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("\nNO EXPO SEED PASSED THE TEACHER GATE.", flush=True)
        print(f"report: {args.output.resolve()}", flush=True)
        raise SystemExit(2)

    seed = selected["seed"]
    detailed = rollout_expert(seed, depth=args.depth, max_steps=args.max_steps, keep_trace=True)

    phase_offsets = [0.0, 0.4, 0.8, 1.2]
    phase_checks = [
        rollout_expert(seed, depth=args.depth, max_steps=args.max_steps, time_offset=offset)
        for offset in phase_offsets
    ]

    visual_dir = args.output.parent / "crossy-v4-expo-seed-audit"
    visual_samples = write_visual_audit(
        seed,
        detailed["trace"] or [],
        output_dir=visual_dir,
        snapshot_limit=args.snapshot_limit,
    )

    report = {
        "version": "crossy-v4-expo-seed-gate-1",
        "status": "PASS",
        "selectedSeed": seed,
        "teacher": {
            "depth": args.depth,
            "maxSteps": args.max_steps,
            "fixedStart": detailed,
            "phaseShiftChecks": phase_checks,
        },
        "profile": selected["profile"],
        "visualAudit": {
            "purpose": (
                "Matched RGB frames and the grayscale representation used by V3, "
                "from the same authoritative states."
            ),
            "samples": visual_samples,
        },
        "search": {
            "prefix": args.prefix,
            "candidatesRequested": args.candidates,
            "interestingCandidatesEvaluated": len(evaluated),
            "targetPasses": args.target_passes,
        },
        "topCandidates": evaluated[:10],
        "nextGate": (
            "Only after inspecting the teacher trajectory and RGB-vs-grayscale "
            "audit should V4 sensory/training changes be implemented."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== SELECTED EXPO SEED ===", flush=True)
    print(f"seed: {seed}", flush=True)
    print(
        f"teacher: {detailed['steps']}/{args.max_steps} steps "
        f"score={detailed['score']:.1f} terminal={detailed['terminalReason']}",
        flush=True,
    )
    print(f"actions: {detailed['actions']}", flush=True)
    print(f"lane profile: {selected['profile']['laneCounts']}", flush=True)
    print("\nphase-shift sanity checks:", flush=True)
    for check in phase_checks:
        print(
            f"  t0={check['timeOffset']:.1f}s steps={check['steps']} "
            f"score={check['score']:.1f} terminal={check['terminalReason']}",
            flush=True,
        )
    print(f"\nreport: {args.output.resolve()}", flush=True)
    print(f"visual samples: {visual_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
