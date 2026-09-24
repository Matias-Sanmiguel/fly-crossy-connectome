from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

import torch

from .checkpoint import validate_checkpoint
from .env import create_game, observe, step_game
from .models import PopulationFixedGraphPolicy
from .schema import ACTION_ORDER, flatten_observation


@torch.no_grad()
def evaluate_neural_policy(
    *,
    checkpoint_path: Path,
    episodes: int,
    max_steps: int,
    seed_prefix: str,
    device_name: str,
) -> dict[str, object]:
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive.")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else device_name
        if device_name != "auto"
        else "cpu"
    )

    saved = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    validated = validate_checkpoint(saved)
    if validated.controller != "connectome":
        raise ValueError("Expected a connectome checkpoint.")
    if validated.connectome_interface != "population":
        raise ValueError("Expected a plain population inference policy.")
    if validated.graph is None:
        raise ValueError("Checkpoint graph is missing.")

    model = PopulationFixedGraphPolicy(
        validated.graph,
        validated.observation_size,
        validated.actions,
    ).to(device)
    model.load_state_dict(validated.state_dict)
    model.eval()

    scores: list[float] = []
    lengths: list[int] = []
    terminals: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    reached_step_limit = 0

    for episode in range(episodes):
        state = create_game(f"{seed_prefix}:{episode:04d}")
        hidden = torch.zeros(
            1,
            validated.graph.node_count,
            device=device,
        )
        steps = 0

        while state.terminal is None and steps < max_steps:
            observation = torch.tensor(
                flatten_observation(observe(state)),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            logits, _, hidden = model(observation, hidden)
            action = ACTION_ORDER[int(torch.argmax(logits, dim=1).item())]
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
        "controller": "MaleCNS-neural-only",
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


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained MaleCNS policy without the expert planner."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed-prefix", required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_neural_policy(
                checkpoint_path=args.checkpoint,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed_prefix=args.seed_prefix,
                device_name=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
