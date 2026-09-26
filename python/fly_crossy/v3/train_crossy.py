from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.feather as feather
import torch
import torch.nn.functional as F

from fly_crossy.env import create_game, step_game
from fly_crossy.expert_planner import plan_action
from fly_crossy.schema import ACTION_ORDER
from fly_crossy.v2.crossy_camera import render_crossy_neural_frame
from .policy import FullMaleCNSCrossyPolicy, IMAGE_H, IMAGE_W

ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_ORDER)}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_flyhard_root() -> Path:
    value = os.environ.get("FLYHARD_ROOT")
    return Path(value) if value else repo_root().parent / "flyhard"


def resize_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected RGB frame [H,W,3].")
    h, w, _ = frame.shape
    ys = np.rint(np.linspace(0, h - 1, IMAGE_H)).astype(np.int64)
    xs = np.rint(np.linspace(0, w - 1, IMAGE_W)).astype(np.int64)
    sampled = frame[ys[:, None], xs[None, :]].astype(np.float32) / 255.0
    gray = (
        0.2126 * sampled[..., 0]
        + 0.7152 * sampled[..., 1]
        + 0.0722 * sampled[..., 2]
    )
    return gray.astype(np.float32)


def safe_alternatives(state, expert_index: int) -> np.ndarray:
    result = []
    for index, action in enumerate(ACTION_ORDER):
        if index == expert_index:
            continue
        if step_game(state, action).state.terminal is None:
            result.append(index)
    return np.asarray(result, dtype=np.int64)


def demonstrations(seeds, *, max_steps, noise, rng):
    images, labels, starts = [], [], []
    for episode_index, seed in enumerate(seeds):
        state = create_game(seed)
        starts.append(len(images))
        for _ in range(max_steps):
            expert = plan_action(state, depth=4)
            expert_index = ACTION_TO_INDEX[expert.action]
            images.append(resize_gray(render_crossy_neural_frame(state)))
            labels.append(expert_index)

            executed = expert_index
            if rng.random() < noise:
                alternatives = safe_alternatives(state, expert_index)
                if len(alternatives):
                    executed = int(rng.choice(alternatives))
            state = step_game(state, ACTION_ORDER[executed]).state
            if state.terminal is not None:
                break
        if (episode_index + 1) % 10 == 0 or episode_index + 1 == len(seeds):
            print(
                f"  demos {episode_index + 1}/{len(seeds)} pairs={len(images):,}",
                flush=True,
            )
    starts.append(len(images))
    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(starts, dtype=np.int64),
    )


def collect_on_policy(policy, device, seeds, *, max_steps):
    policy.eval()
    per_images = [[] for _ in seeds]
    per_labels = [[] for _ in seeds]
    states = [create_game(seed) for seed in seeds]
    neural = policy.zero_state(len(seeds), device=device)
    active = list(range(len(seeds)))

    for _ in range(max_steps):
        if not active:
            break
        batch_images = np.stack(
            [resize_gray(render_crossy_neural_frame(states[i])) for i in active]
        )
        tensor_images = torch.as_tensor(
            batch_images, dtype=torch.float32, device=device
        )
        with torch.no_grad():
            sub = policy.step_state(neural[:, active], tensor_images)
            neural[:, active] = sub
            commands = policy.readout(sub).argmax(dim=1).cpu().numpy()

        still = []
        for local, episode_index in enumerate(active):
            state = states[episode_index]
            expert = plan_action(state, depth=4)
            per_images[episode_index].append(batch_images[local])
            per_labels[episode_index].append(ACTION_TO_INDEX[expert.action])
            next_state = step_game(
                state, ACTION_ORDER[int(commands[local])]
            ).state
            states[episode_index] = next_state
            if next_state.terminal is None:
                still.append(episode_index)
        active = still

    policy.train()
    images, labels, starts = [], [], []
    for episode_images, episode_labels in zip(
        per_images, per_labels, strict=True
    ):
        starts.append(len(images))
        images.extend(episode_images)
        labels.extend(episode_labels)
    starts.append(len(images))
    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(starts, dtype=np.int64),
    )


def valid_window_starts(starts, window):
    parts = []
    for episode in range(len(starts) - 1):
        begin, end = int(starts[episode]), int(starts[episode + 1])
        if end - begin >= window:
            parts.append(np.arange(begin, end - window + 1, dtype=np.int64))
    if not parts:
        raise RuntimeError("No episode is long enough for the requested window.")
    return np.concatenate(parts)


@torch.no_grad()
def evaluate(policy, device, seeds, *, max_steps):
    policy.eval()
    scores, lengths = [], []
    terminals = Counter()
    actions = Counter()
    reached = 0

    for seed in seeds:
        state = create_game(seed)
        neural = policy.zero_state(1, device=device)
        steps = 0
        while state.terminal is None and steps < max_steps:
            image = torch.as_tensor(
                resize_gray(render_crossy_neural_frame(state))[None],
                dtype=torch.float32,
                device=device,
            )
            neural = policy.step_state(neural, image)
            action_index = int(policy.readout(neural)[0].argmax().item())
            action = ACTION_ORDER[action_index]
            actions[action.value] += 1
            state = step_game(state, action).state
            steps += 1
        scores.append(float(state.score))
        lengths.append(steps)
        if state.terminal is None:
            reached += 1
        else:
            terminals[state.terminal] += 1

    ordered = sorted(scores)
    return {
        "episodes": len(seeds),
        "maxSteps": max_steps,
        "meanScore": float(np.mean(scores)),
        "medianScore": float(ordered[len(ordered) // 2]),
        "bestScore": float(max(scores)),
        "meanLength": float(np.mean(lengths)),
        "reachedStepLimit": reached,
        "terminalReasons": dict(sorted(terminals.items())),
        "actions": dict(sorted(actions.items())),
    }


def train_steps(
    *,
    policy,
    optimizer,
    x,
    y,
    starts,
    steps,
    window,
    batch,
    rng,
    device,
    tag,
    history,
):
    valid = valid_window_starts(starts, window)
    for local_step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        s0 = rng.choice(valid, batch, replace=True)
        idx = torch.as_tensor(
            s0[None, :] + np.arange(window)[:, None],
            dtype=torch.long,
            device=device,
        )
        state = policy.zero_state(batch, device=device)
        logits, _ = policy.run_window(x[idx], state)
        loss = F.cross_entropy(
            logits.reshape(-1, len(ACTION_ORDER)), y[idx].reshape(-1)
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        history.append(
            {"step": len(history) + 1, "stage": tag, "loss": float(loss.detach())}
        )
        if local_step == 0 or (local_step + 1) % 25 == 0 or local_step + 1 == steps:
            print(
                f"  {tag} {local_step + 1}/{steps} loss={history[-1]['loss']:.4f}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Crossy V3: full MaleCNS recipe aligned with fly-self-driving."
    )
    parser.add_argument("--flyhard-root", type=Path, default=default_flyhard_root())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--demo-episodes", type=int, default=60)
    parser.add_argument("--demo-steps", type=int, default=160)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--clone-steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-steps", type=int, default=500)
    parser.add_argument("--dagger-episodes", type=int, default=30)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root() / "runs" / "crossy-v3-full-malecns",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    graph_root = args.flyhard_root / "data" / "graph-traced-v1"
    graph_path = graph_root / "graph.npz"
    nodes_path = graph_root / "nodes.feather"
    if not graph_path.is_file() or not nodes_path.is_file():
        raise SystemExit(
            "Full traced MaleCNS graph missing. Expected flyhard/data/graph-traced-v1."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    graph = dict(np.load(graph_path))
    nodes = feather.read_table(nodes_path)
    classes = np.asarray(nodes["superclass"].fill_null("").to_pylist())
    sensory_ids = np.flatnonzero(classes == "ol_sensory")
    motor_ids = np.flatnonzero(classes == "vnc_motor")

    print("=== CROSSY V3 / FLY-SELF-DRIVING RECIPE ===", flush=True)
    print(f"neurons: {len(graph['crow']) - 1:,}", flush=True)
    print(f"edges: {len(graph['col']):,}", flush=True)
    print(f"ol_sensory: {len(sensory_ids):,}", flush=True)
    print(f"vnc_motor: {len(motor_ids):,}", flush=True)
    print("trainable: one gain per measured edge + one leak per neuron", flush=True)
    print("state reset: episode start only", flush=True)

    policy = FullMaleCNSCrossyPolicy(
        graph, sensory_ids, motor_ids, seed=args.seed
    ).to(device)

    train_seeds = [
        f"crossy-v3-train:{i:04d}" for i in range(args.demo_episodes)
    ]
    images, labels, starts = demonstrations(
        train_seeds,
        max_steps=args.demo_steps,
        noise=args.noise,
        rng=rng,
    )
    x = torch.as_tensor(images, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    calibration = x[:: max(1, len(x) // 64)][:64]
    policy.calibrate_decoder(calibration)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    history = []
    started = time.perf_counter()

    print("\n[BC] full-connectome TBPTT...", flush=True)
    train_steps(
        policy=policy,
        optimizer=optimizer,
        x=x,
        y=y,
        starts=starts,
        steps=args.clone_steps,
        window=args.window,
        batch=args.batch,
        rng=rng,
        device=device,
        tag="clone",
        history=history,
    )

    eval_sets = [
        [f"crossy-v3-final-a:{i:04d}" for i in range(20)],
        [f"crossy-v3-final-b:{i:04d}" for i in range(20)],
        [f"crossy-v3-final-c:{i:04d}" for i in range(20)],
    ]
    stage_metrics = {
        "clone": [evaluate(policy, device, seeds, max_steps=args.eval_steps) for seeds in eval_sets]
    }

    for round_index in range(args.dagger_rounds):
        print(f"\n[DAgger {round_index + 1}/{args.dagger_rounds}] collect...", flush=True)
        chosen = rng.choice(
            np.asarray(train_seeds),
            size=min(args.dagger_episodes, len(train_seeds)),
            replace=False,
        ).tolist()
        xi, yi, si = collect_on_policy(
            policy, device, chosen, max_steps=args.demo_steps
        )
        offset = len(images)
        images = np.concatenate([images, xi], axis=0)
        labels = np.concatenate([labels, yi], axis=0)
        starts = np.concatenate([starts[:-1], si + offset])
        x = torch.as_tensor(images, dtype=torch.float32, device=device)
        y = torch.as_tensor(labels, dtype=torch.long, device=device)
        print(f"  appended {len(xi):,}; total={len(images):,}", flush=True)

        train_steps(
            policy=policy,
            optimizer=optimizer,
            x=x,
            y=y,
            starts=starts,
            steps=args.dagger_steps,
            window=args.window,
            batch=args.batch,
            rng=rng,
            device=device,
            tag=f"dagger{round_index + 1}",
            history=history,
        )
        stage_metrics[f"dagger{round_index + 1}"] = [
            evaluate(policy, device, seeds, max_steps=args.eval_steps)
            for seeds in eval_sets
        ]

    checkpoint = args.out / "checkpoint.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "config": vars(args),
            "sensoryCount": int(len(sensory_ids)),
            "motorCount": int(len(motor_ids)),
        },
        checkpoint,
    )

    report = {
        "recipe": "fly-self-driving-style-full-connectome-bc-dagger",
        "graph": {
            "neurons": int(len(graph["crow"]) - 1),
            "edges": int(len(graph["col"])),
            "sensory": int(len(sensory_ids)),
            "motor": int(len(motor_ids)),
        },
        "training": {
            "demoEpisodes": args.demo_episodes,
            "noise": args.noise,
            "window": args.window,
            "batch": args.batch,
            "cloneSteps": args.clone_steps,
            "learningRate": args.lr,
            "daggerRounds": args.dagger_rounds,
            "daggerStepsPerRound": args.dagger_steps,
            "trainableParameters": int(sum(p.numel() for p in policy.parameters())),
            "stateReset": "episode start only",
            "graphUpdatesPerDecision": 4,
        },
        "evaluation": stage_metrics,
        "elapsedSeconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print("\nV3 TRAINING COMPLETE", flush=True)
    print(f"checkpoint: {checkpoint.resolve()}", flush=True)
    print(f"report: {report_path.resolve()}", flush=True)
    for stage, sets in stage_metrics.items():
        summary = [
            f"{m['reachedStepLimit']}/{m['episodes']} mean={m['meanScore']:.1f}"
            for m in sets
        ]
        print(f"{stage}: {' | '.join(summary)}", flush=True)


if __name__ == "__main__":
    main()
