from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical

from .connectome import load_default_reduced_graph
from .env import FlyCrossyEnv, WORLD_VERSION, hash_seed
from .export import export_policy
from .models import DensePolicy, FixedGraphPolicy
from .schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE


HIDDEN_SIZE = 64
ROLLOUT_STEPS = 128
PPO_EPOCHS = 4
MINIBATCH_SIZE = 256
DISCOUNT = 0.99
GAE_LAMBDA = 0.95
CLIP_COEFFICIENT = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
MAX_GRADIENT_NORM = 0.5
DETERMINISTIC_CUBLAS_WORKSPACE = ":4096:8"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    controller: str
    seed: str
    steps: int
    envs: int
    learning_rate: float
    output: Path
    device: str = "auto"

    def validate(self) -> None:
        if self.controller not in ("conventional", "connectome"):
            raise ValueError("Controller must be conventional or connectome.")
        if not self.seed:
            raise ValueError("Training seed must not be empty.")
        if self.steps <= 0 or self.envs <= 0:
            raise ValueError("Training steps and environment count must be positive.")
        if self.steps % self.envs != 0:
            raise ValueError("Training steps must be divisible by the environment count.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite.")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if requested not in ("cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda.")
    return torch.device(requested)


def _configure_deterministic_runtime(requested_device: str) -> None:
    if requested_device in ("auto", "cuda"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE
    torch.use_deterministic_algorithms(True)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _ppo_update(
    model: DensePolicy | FixedGraphPolicy,
    optimizer: torch.optim.Optimizer,
    observations: Tensor,
    actions: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    returns: Tensor,
    hidden_states: Tensor | None = None,
) -> dict[str, float]:
    normalized_advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    batch_size = observations.shape[0]
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approximate_kls: list[float] = []

    for _ in range(PPO_EPOCHS):
        for indices in torch.randperm(batch_size, device=observations.device).split(
            MINIBATCH_SIZE
        ):
            if isinstance(model, FixedGraphPolicy):
                if hidden_states is None:
                    raise ValueError("Fixed graph PPO updates require recurrent hidden states.")
                logits, values, _ = model(observations[indices], hidden_states[indices])
            else:
                logits, values = model(observations[indices])
            distribution = Categorical(logits=logits)
            new_log_probabilities = distribution.log_prob(actions[indices])
            log_ratio = new_log_probabilities - old_log_probabilities[indices]
            ratio = log_ratio.exp()
            unclipped = -normalized_advantages[indices] * ratio
            clipped = -normalized_advantages[indices] * torch.clamp(
                ratio, 1 - CLIP_COEFFICIENT, 1 + CLIP_COEFFICIENT
            )
            policy_loss = torch.maximum(unclipped, clipped).mean()
            value_loss = 0.5 * (values - returns[indices]).square().mean()
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + VALUE_COEFFICIENT * value_loss
                - ENTROPY_COEFFICIENT * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRADIENT_NORM)
            optimizer.step()

            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            approximate_kls.append(float(((ratio - 1) - log_ratio).mean().detach().cpu()))

    return {
        "policyLoss": _mean(policy_losses),
        "valueLoss": _mean(value_losses),
        "entropy": _mean(entropies),
        "approximateKl": _mean(approximate_kls),
    }


def train(config: TrainingConfig) -> dict[str, Any]:
    """Train a conventional or reduced-connectome PPO actor-critic artifact bundle."""
    config.validate()
    _configure_deterministic_runtime(config.device)
    device = _resolve_device(config.device)
    numeric_seed = hash_seed(config.seed)
    np.random.seed(numeric_seed)
    torch.manual_seed(numeric_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(numeric_seed)

    environments = [FlyCrossyEnv() for _ in range(config.envs)]
    episode_counts = [0] * config.envs
    training_world_seeds = [
        f"{config.seed}:{index}:0" for index in range(config.envs)
    ]
    observations = np.stack(
        [
            environment.reset(training_world_seeds[index])[0]
            for index, environment in enumerate(environments)
        ]
    )
    episode_returns = [0.0] * config.envs
    episode_lengths = [0] * config.envs
    completed_episodes: list[dict[str, Any]] = []

    if config.controller == "connectome":
        model: DensePolicy | FixedGraphPolicy = FixedGraphPolicy(
            load_default_reduced_graph(),
            OBSERVATION_INPUT_SIZE,
            len(ACTION_ORDER),
        ).to(device)
        hidden_state: Tensor | None = torch.zeros(
            config.envs, model.graph.node_count, dtype=torch.float32, device=device
        )
    else:
        model = DensePolicy(
            OBSERVATION_INPUT_SIZE, HIDDEN_SIZE, len(ACTION_ORDER)
        ).to(device)
        hidden_state = None
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_steps = 0
    update_index = 0
    training_curve: list[dict[str, Any]] = []

    while total_steps < config.steps:
        rollout_length = min(ROLLOUT_STEPS, (config.steps - total_steps) // config.envs)
        rollout_observations: list[Tensor] = []
        rollout_actions: list[Tensor] = []
        rollout_log_probabilities: list[Tensor] = []
        rollout_rewards: list[Tensor] = []
        rollout_dones: list[Tensor] = []
        rollout_values: list[Tensor] = []
        rollout_hidden_states: list[Tensor] = []
        episode_start = len(completed_episodes)

        for _ in range(rollout_length):
            observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
            with torch.no_grad():
                if isinstance(model, FixedGraphPolicy):
                    assert hidden_state is not None
                    rollout_hidden_states.append(hidden_state)
                    logits, values, next_hidden_state = model(
                        observation_tensor, hidden_state
                    )
                else:
                    logits, values = model(observation_tensor)
                    next_hidden_state = None
                distribution = Categorical(logits=logits)
                action_indices = distribution.sample()
                log_probabilities = distribution.log_prob(action_indices)

            next_observations: list[np.ndarray[Any, np.dtype[np.float32]]] = []
            rewards: list[float] = []
            dones: list[bool] = []
            for environment_index, (environment, action_index) in enumerate(
                zip(environments, action_indices.detach().cpu().tolist(), strict=True)
            ):
                next_observation, reward, terminated, truncated, info = environment.step(
                    ACTION_ORDER[action_index]
                )
                done = terminated or truncated
                episode_returns[environment_index] += reward
                episode_lengths[environment_index] += 1
                rewards.append(reward)
                dones.append(done)
                if done:
                    completed_episodes.append(
                        {
                            "episode": len(completed_episodes) + 1,
                            "environment": environment_index,
                            "return": float(episode_returns[environment_index]),
                            "length": episode_lengths[environment_index],
                            "score": float(info["score"]),
                            "terminalReason": info["terminalReason"],
                        }
                    )
                    episode_counts[environment_index] += 1
                    next_seed = (
                        f"{config.seed}:{environment_index}:"
                        f"{episode_counts[environment_index]}"
                    )
                    training_world_seeds.append(next_seed)
                    next_observation, _ = environment.reset(next_seed)
                    episode_returns[environment_index] = 0.0
                    episode_lengths[environment_index] = 0
                next_observations.append(next_observation)

            rollout_observations.append(observation_tensor)
            rollout_actions.append(action_indices)
            rollout_log_probabilities.append(log_probabilities)
            rollout_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=device))
            rollout_dones.append(torch.tensor(dones, dtype=torch.float32, device=device))
            rollout_values.append(values)
            observations = np.stack(next_observations)
            if next_hidden_state is not None:
                hidden_state = next_hidden_state.clone()
                hidden_state[
                    torch.as_tensor(dones, dtype=torch.bool, device=device)
                ] = 0
            total_steps += config.envs

        with torch.no_grad():
            bootstrap_observations = torch.as_tensor(
                observations, dtype=torch.float32, device=device
            )
            if isinstance(model, FixedGraphPolicy):
                assert hidden_state is not None
                _, bootstrap_value, _ = model(bootstrap_observations, hidden_state)
            else:
                _, bootstrap_value = model(bootstrap_observations)
        rewards_tensor = torch.stack(rollout_rewards)
        dones_tensor = torch.stack(rollout_dones)
        values_tensor = torch.stack(rollout_values)
        advantages = torch.zeros_like(rewards_tensor)
        last_advantage = torch.zeros(config.envs, dtype=torch.float32, device=device)
        for timestep in reversed(range(rollout_length)):
            next_value = bootstrap_value if timestep == rollout_length - 1 else values_tensor[timestep + 1]
            nonterminal = 1.0 - dones_tensor[timestep]
            delta = (
                rewards_tensor[timestep]
                + DISCOUNT * next_value * nonterminal
                - values_tensor[timestep]
            )
            last_advantage = (
                delta + DISCOUNT * GAE_LAMBDA * nonterminal * last_advantage
            )
            advantages[timestep] = last_advantage
        returns = advantages + values_tensor

        update_metrics = _ppo_update(
            model,
            optimizer,
            torch.stack(rollout_observations).flatten(0, 1),
            torch.stack(rollout_actions).flatten(),
            torch.stack(rollout_log_probabilities).flatten(),
            advantages.flatten(),
            returns.flatten(),
            (
                torch.stack(rollout_hidden_states).flatten(0, 1)
                if rollout_hidden_states
                else None
            ),
        )
        update_index += 1
        recent_returns = [
            float(episode["return"]) for episode in completed_episodes[episode_start:]
        ]
        training_curve.append(
            {
                "update": update_index,
                "totalSteps": total_steps,
                **update_metrics,
                "episodesCompleted": len(completed_episodes),
                "meanEpisodeReturn": _mean(recent_returns),
            }
        )

    config.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output / "checkpoint.pt"
    model_metadata: dict[str, Any]
    if isinstance(model, FixedGraphPolicy):
        model_metadata = {
            "observation_size": OBSERVATION_INPUT_SIZE,
            "actions": len(ACTION_ORDER),
            "graph": model.graph.to_checkpoint(),
        }
    else:
        model_metadata = {
            "observation_size": OBSERVATION_INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "actions": len(ACTION_ORDER),
        }
    torch.save(
        {
            "format_version": 1,
            "environment_version": WORLD_VERSION,
            "controller": config.controller,
            "model": model_metadata,
            "model_state_dict": model.state_dict(),
            "training": {
                "seed": config.seed,
                "steps": config.steps,
                "envs": config.envs,
                "learning_rate": config.learning_rate,
                "world_seeds": training_world_seeds,
            },
        },
        checkpoint_path,
    )
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    policy_path = config.output / "policy.json"
    export_policy(checkpoint_path, policy_path)

    metadata = {
        "version": 1,
        "environmentVersion": WORLD_VERSION,
        "configuration": {
            **asdict(config),
            "output": str(config.output),
            "resolvedDevice": str(device),
            "trainingWorldSeeds": training_world_seeds,
            **(
                {
                    "graphNodes": model.graph.node_count,
                    "graphEdges": model.graph.edge_count,
                    "graphArtifactHash": model.graph.artifact_sha256,
                }
                if isinstance(model, FixedGraphPolicy)
                else {"hiddenSize": HIDDEN_SIZE}
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cudaAvailable": torch.cuda.is_available(),
        },
        "checkpoint": {"path": checkpoint_path.name, "sha256": checkpoint_hash},
        "policy": {"path": policy_path.name},
    }
    episode_return_values = [float(episode["return"]) for episode in completed_episodes]
    metrics = {
        "version": 1,
        "totalSteps": total_steps,
        "trainingCurve": training_curve,
        "episodes": completed_episodes,
        "summary": {
            "episodesCompleted": len(completed_episodes),
            "meanEpisodeReturn": _mean(episode_return_values),
            "bestScore": max(
                (float(episode["score"]) for episode in completed_episodes), default=0.0
            ),
        },
    }
    _write_json(config.output / "metadata.json", metadata)
    _write_json(config.output / "metrics.json", metrics)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpointHash": checkpoint_hash,
        "metadata": str(config.output / "metadata.json"),
        "metrics": str(config.output / "metrics.json"),
        "policy": str(policy_path),
    }


def _parse_arguments() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train the Fly Crossy PPO baseline.")
    parser.add_argument(
        "--controller", choices=("conventional", "connectome"), default="conventional"
    )
    parser.add_argument("--seed", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    return TrainingConfig(
        controller=arguments.controller,
        seed=arguments.seed,
        steps=arguments.steps,
        envs=arguments.envs,
        learning_rate=arguments.learning_rate,
        output=arguments.output,
        device=arguments.device,
    )


def _main() -> None:
    print(json.dumps(train(_parse_arguments()), indent=2))


if __name__ == "__main__":
    _main()
