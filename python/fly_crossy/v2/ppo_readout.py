from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
import torch.nn.functional as F

from fly_crossy.env import create_game, step_game
from fly_crossy.schema import ACTION_ORDER

from .core import FrozenMaleCNSCore
from .crossy_camera import render_crossy_neural_frame
from .decoder import (
    EXPECTED_INPUT_SIZE,
    MaleCNSActionDecoder,
    load_decoder,
    save_decoder,
)
from .substrate import MaleCNSV2Substrate
from .vision_frontend import FrozenVisualFrontEnd
from .vision_geometry import VisualGeometry


PPO_VERSION = "malecns-crossy-v2-ppo-readout-1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class ValueReadout(nn.Module):
    """Training-only critic. Never exported to the final controller."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(EXPECTED_INPUT_SIZE, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x).squeeze(-1)


def normalize_features(
    features: Tensor,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    return torch.clamp((features - mean) / std, -10.0, 10.0)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    if not (
        values.shape == rewards.shape
        and next_values.shape == rewards.shape
        and dones.shape == rewards.shape
    ):
        raise ValueError("GAE arrays must share shape.")

    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0

    for t in range(n - 1, -1, -1):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        # Because the rollout can contain multiple complete episodes, a done
        # transition must stop the recursive advantage chain.
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns.astype(np.float32)


@torch.no_grad()
def encode_state(
    state,
    *,
    frontend: FrozenVisualFrontEnd,
    brain: FrozenMaleCNSCore,
    vision_state,
    brain_state,
) -> tuple[Tensor, object, object]:
    frame = render_crossy_neural_frame(state)
    visual_rates, vision_state = frontend.process(frame[None], vision_state)
    brain_state = brain.step(brain_state, visual_rates)
    features = brain.decision_features(brain_state)[0].detach()
    if features.shape != (EXPECTED_INPUT_SIZE,):
        raise RuntimeError(
            f"MaleCNS decision feature width changed: {features.shape}."
        )
    return features, vision_state, brain_state


@torch.no_grad()
def evaluate_policy(
    policy: MaleCNSActionDecoder,
    *,
    mean: Tensor,
    std: Tensor,
    frontend: FrozenVisualFrontEnd,
    brain: FrozenMaleCNSCore,
    episodes: int,
    max_steps: int,
    seed_prefix: str,
) -> dict[str, object]:
    policy.eval()
    scores: list[float] = []
    lengths: list[int] = []
    terminals: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    reached_limit = 0

    for episode in range(episodes):
        state = create_game(f"{seed_prefix}:{episode:04d}")
        vision_state = frontend.init_state(1)
        brain_state = brain.init_state(1)
        steps = 0

        while state.terminal is None and steps < max_steps:
            features, vision_state, brain_state = encode_state(
                state,
                frontend=frontend,
                brain=brain,
                vision_state=vision_state,
                brain_state=brain_state,
            )
            x = normalize_features(features, mean, std)
            logits = policy(x[None])[0]
            action_index = int(torch.argmax(logits).item())
            action = ACTION_ORDER[action_index]
            actions[action.value] += 1
            state = step_game(state, action).state
            steps += 1

        scores.append(float(state.score))
        lengths.append(steps)
        if state.terminal is None:
            reached_limit += 1
        else:
            terminals[state.terminal] += 1

    ordered = sorted(scores)
    return {
        "episodes": episodes,
        "maxSteps": max_steps,
        "meanScore": float(np.mean(scores)),
        "medianScore": float(ordered[len(ordered) // 2]),
        "bestScore": float(max(scores)),
        "meanLength": float(np.mean(lengths)),
        "reachedStepLimit": reached_limit,
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
            "Closed-loop PPO on the MaleCNS V2 readout only. Camera, retina and "
            "138,968-cell MaleCNS remain frozen. The planner is not used for training."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.20)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--episode-max-steps", type=int, default=200)
    parser.add_argument("--dev-every", type=int, default=10)
    parser.add_argument("--dev-episodes", type=int, default=16)
    parser.add_argument("--dev-max-steps", type=int, default=160)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "decoder-r0.pt",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "artifacts" / "malecns-crossy-v2" / "decoder-ppo-r1.pt",
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
        default=root / "reports" / "malecns-crossy-v2-ppo-r1-training.json",
    )
    args = parser.parse_args()

    if args.rollout_steps <= 0 or args.updates <= 0:
        raise SystemExit("updates and rollout-steps must be positive.")
    if args.batch_size <= 0 or args.batch_size > args.rollout_steps:
        raise SystemExit("batch-size must be in [1, rollout-steps].")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    substrate = MaleCNSV2Substrate.load(args.substrate, strict=True)
    geometry = VisualGeometry.load(args.geometry, strict=True)
    initial = load_decoder(args.initial_checkpoint, device=device)

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
    gain = brain.match_rest_gain(target=0.95, iterations=80)

    if sum(parameter.numel() for parameter in brain.parameters()) != 0:
        raise SystemExit("MaleCNS unexpectedly has trainable parameters.")
    if frontend.trainable_parameters != 0:
        raise SystemExit("Visual frontend unexpectedly has trainable parameters.")

    # The actor is exactly the existing R0 decoder architecture and starts from
    # its imitation weights. Only this readout changes in PPO.
    policy = initial.model
    policy.train()

    critic = ValueReadout().to(device)

    mean = initial.mean.detach()
    std = initial.std.detach()

    actor_optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_learning_rate,
        weight_decay=1e-5,
    )

    rng = np.random.default_rng(args.seed)

    train_episode_counter = 0
    train_episode_returns: list[float] = []
    train_episode_scores: list[float] = []
    train_episode_lengths: list[int] = []
    train_terminals: Counter[str] = Counter()

    def reset_training_episode():
        nonlocal train_episode_counter
        state = create_game(f"malecns-v2-ppo-train:{train_episode_counter:06d}")
        train_episode_counter += 1
        return state, frontend.init_state(1), brain.init_state(1)

    state, vision_state, brain_state = reset_training_episode()
    current_features, vision_state, brain_state = encode_state(
        state,
        frontend=frontend,
        brain=brain,
        vision_state=vision_state,
        brain_state=brain_state,
    )
    current_episode_return = 0.0
    current_episode_length = 0

    print("=== MaleCNS Crossy V2 / PPO Readout R1 ===", flush=True)
    print("camera: frozen", flush=True)
    print("retina: frozen", flush=True)
    print("MaleCNS: frozen", flush=True)
    print("planner in training loop: NO", flush=True)
    print("actor: R0 21022 -> 256 -> 5, PPO trainable", flush=True)
    print("critic: training-only 21022 -> 128 -> 1", flush=True)
    print(
        f"budget: {args.updates} x {args.rollout_steps} = "
        f"{args.updates * args.rollout_steps:,} environment steps",
        flush=True,
    )

    baseline_dev = evaluate_policy(
        policy,
        mean=mean,
        std=std,
        frontend=frontend,
        brain=brain,
        episodes=args.dev_episodes,
        max_steps=args.dev_max_steps,
        seed_prefix="malecns-v2-ppo-dev",
    )
    print(
        f"baseline dev: mean={baseline_dev['meanScore']:.2f} "
        f"median={baseline_dev['medianScore']:.2f} "
        f"limit={baseline_dev['reachedStepLimit']}/{args.dev_episodes}",
        flush=True,
    )

    best_dev = baseline_dev
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in policy.state_dict().items()
    }
    best_update = 0
    updates_report: list[dict[str, object]] = []

    started = time.perf_counter()
    total_env_steps = 0

    for update in range(1, args.updates + 1):
        policy.eval()
        critic.eval()

        feature_rows = np.empty(
            (args.rollout_steps, EXPECTED_INPUT_SIZE),
            dtype=np.float16,
        )
        actions = np.empty(args.rollout_steps, dtype=np.int64)
        old_log_probs = np.empty(args.rollout_steps, dtype=np.float32)
        rewards = np.empty(args.rollout_steps, dtype=np.float32)
        values = np.empty(args.rollout_steps, dtype=np.float32)
        next_values = np.empty(args.rollout_steps, dtype=np.float32)
        dones = np.empty(args.rollout_steps, dtype=np.bool_)

        rollout_reward = 0.0
        rollout_deaths: Counter[str] = Counter()
        rollout_episodes = 0

        for t in range(args.rollout_steps):
            with torch.no_grad():
                x = normalize_features(current_features, mean, std)
                logits = policy(x[None])[0]
                distribution = Categorical(logits=logits)
                action_tensor = distribution.sample()
                action_index = int(action_tensor.item())
                log_prob = float(distribution.log_prob(action_tensor).item())
                value = float(critic(x[None]).item())

            feature_rows[t] = (
                current_features.detach()
                .to("cpu", dtype=torch.float16)
                .numpy()
            )
            actions[t] = action_index
            old_log_probs[t] = log_prob
            values[t] = value

            result = step_game(state, ACTION_ORDER[action_index])
            next_state = result.state
            reward = float(result.reward)
            terminal = next_state.terminal is not None
            time_limit = (
                not terminal
                and current_episode_length + 1 >= args.episode_max_steps
            )
            done = terminal or time_limit

            rewards[t] = reward
            dones[t] = done
            rollout_reward += reward
            current_episode_return += reward
            current_episode_length += 1
            total_env_steps += 1

            if done:
                # A 200-step time limit is treated as a successful finite
                # training episode boundary. It carries no death penalty.
                next_values[t] = 0.0
                train_episode_returns.append(current_episode_return)
                train_episode_scores.append(float(next_state.score))
                train_episode_lengths.append(current_episode_length)
                if terminal:
                    train_terminals[next_state.terminal] += 1
                    rollout_deaths[next_state.terminal] += 1
                else:
                    train_terminals["step_limit"] += 1
                rollout_episodes += 1

                state, vision_state, brain_state = reset_training_episode()
                current_features, vision_state, brain_state = encode_state(
                    state,
                    frontend=frontend,
                    brain=brain,
                    vision_state=vision_state,
                    brain_state=brain_state,
                )
                current_episode_return = 0.0
                current_episode_length = 0
            else:
                state = next_state
                current_features, vision_state, brain_state = encode_state(
                    state,
                    frontend=frontend,
                    brain=brain,
                    vision_state=vision_state,
                    brain_state=brain_state,
                )
                with torch.no_grad():
                    nx = normalize_features(current_features, mean, std)
                    next_values[t] = float(critic(nx[None]).item())

        advantages_np, returns_np = compute_gae(
            rewards,
            values,
            next_values,
            dones,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        advantages_np = (
            advantages_np - advantages_np.mean()
        ) / max(float(advantages_np.std()), 1e-6)

        policy.train()
        critic.train()

        order = np.arange(args.rollout_steps, dtype=np.int64)
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropy_values: list[float] = []
        kl_values: list[float] = []
        clip_fractions: list[float] = []

        stop_for_kl = False

        for epoch in range(args.ppo_epochs):
            rng.shuffle(order)

            for start in range(0, len(order), args.batch_size):
                rows = order[start:start + args.batch_size]

                features = torch.as_tensor(
                    feature_rows[rows],
                    dtype=torch.float32,
                    device=device,
                )
                x = normalize_features(features, mean, std)
                action_batch = torch.as_tensor(
                    actions[rows],
                    dtype=torch.long,
                    device=device,
                )
                old_log_batch = torch.as_tensor(
                    old_log_probs[rows],
                    dtype=torch.float32,
                    device=device,
                )
                advantage_batch = torch.as_tensor(
                    advantages_np[rows],
                    dtype=torch.float32,
                    device=device,
                )
                return_batch = torch.as_tensor(
                    returns_np[rows],
                    dtype=torch.float32,
                    device=device,
                )

                logits = policy(x)
                distribution = Categorical(logits=logits)
                new_log = distribution.log_prob(action_batch)
                entropy = distribution.entropy().mean()
                ratio = torch.exp(new_log - old_log_batch)

                unclipped = ratio * advantage_batch
                clipped = torch.clamp(
                    ratio,
                    1.0 - args.clip,
                    1.0 + args.clip,
                ) * advantage_batch
                policy_loss = -torch.min(unclipped, clipped).mean()
                actor_loss = policy_loss - args.entropy_coef * entropy

                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    args.max_grad_norm,
                )
                actor_optimizer.step()

                predicted_value = critic(x)
                value_loss = F.mse_loss(predicted_value, return_batch)

                critic_optimizer.zero_grad(set_to_none=True)
                (args.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(),
                    args.max_grad_norm,
                )
                critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = float((old_log_batch - new_log).mean().item())
                    clip_fraction = float(
                        ((ratio - 1.0).abs() > args.clip)
                        .to(torch.float32)
                        .mean()
                        .item()
                    )

                actor_losses.append(float(policy_loss.detach().item()))
                critic_losses.append(float(value_loss.detach().item()))
                entropy_values.append(float(entropy.detach().item()))
                kl_values.append(approx_kl)
                clip_fractions.append(clip_fraction)

                if approx_kl > args.target_kl:
                    stop_for_kl = True
                    break

            if stop_for_kl:
                break

        update_report: dict[str, object] = {
            "update": update,
            "environmentSteps": total_env_steps,
            "rolloutReward": rollout_reward,
            "rolloutEpisodesCompleted": rollout_episodes,
            "rolloutDeaths": dict(sorted(rollout_deaths.items())),
            "actorLoss": float(np.mean(actor_losses)),
            "criticLoss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropy_values)),
            "approxKl": float(np.mean(kl_values)),
            "clipFraction": float(np.mean(clip_fractions)),
            "earlyStoppedForKl": stop_for_kl,
        }

        should_dev = (
            update == 1
            or update % args.dev_every == 0
            or update == args.updates
        )
        if should_dev:
            dev = evaluate_policy(
                policy,
                mean=mean,
                std=std,
                frontend=frontend,
                brain=brain,
                episodes=args.dev_episodes,
                max_steps=args.dev_max_steps,
                seed_prefix="malecns-v2-ppo-dev",
            )
            update_report["dev"] = dev

            if selection_key(dev) > selection_key(best_dev):
                best_dev = dev
                best_update = update
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in policy.state_dict().items()
                }

            print(
                f"update {update:02d}/{args.updates} "
                f"steps={total_env_steps:,} "
                f"rolloutR={rollout_reward:.1f} "
                f"entropy={update_report['entropy']:.3f} "
                f"kl={update_report['approxKl']:.4f} | "
                f"dev mean={dev['meanScore']:.2f} "
                f"median={dev['medianScore']:.2f} "
                f"limit={dev['reachedStepLimit']}/{args.dev_episodes} "
                f"bestUpdate={best_update}",
                flush=True,
            )
        else:
            print(
                f"update {update:02d}/{args.updates} "
                f"steps={total_env_steps:,} "
                f"rolloutR={rollout_reward:.1f} "
                f"episodes={rollout_episodes} "
                f"entropy={update_report['entropy']:.3f} "
                f"kl={update_report['approxKl']:.4f}",
                flush=True,
            )

        updates_report.append(update_report)

    policy.load_state_dict(best_state)
    policy.to(device)
    policy.eval()

    metadata = {
        "ppoVersion": PPO_VERSION,
        "initialCheckpoint": str(args.initial_checkpoint.resolve()),
        "bestUpdate": best_update,
        "bestDev": best_dev,
        "trainingSeedPrefix": "malecns-v2-ppo-train",
        "devSeedPrefix": "malecns-v2-ppo-dev",
        "environmentSteps": total_env_steps,
        "plannerUsedForTraining": False,
        "cameraFrozen": True,
        "visualFrontendFrozen": True,
        "MaleCNSFrozen": True,
        "criticExported": False,
        "actorArchitecture": "21022 -> 256 GELU LayerNorm -> 5",
        "training": {
            "algorithm": "clipped PPO",
            "gamma": args.gamma,
            "gaeLambda": args.gae_lambda,
            "clip": args.clip,
            "entropyCoef": args.entropy_coef,
            "learningRate": args.learning_rate,
            "criticLearningRate": args.critic_learning_rate,
            "rolloutSteps": args.rollout_steps,
            "updates": args.updates,
            "ppoEpochs": args.ppo_epochs,
            "batchSize": args.batch_size,
            "episodeMaxSteps": args.episode_max_steps,
        },
    }

    save_decoder(
        args.checkpoint,
        model=policy,
        mean=mean.detach().cpu().numpy(),
        std=std.detach().cpu().numpy(),
        metadata=metadata,
    )

    report = {
        "version": 1,
        "ppoVersion": PPO_VERSION,
        "checkpoint": str(args.checkpoint.resolve()),
        "initialCheckpoint": str(args.initial_checkpoint.resolve()),
        "frozenSystem": {
            "camera": True,
            "visualFrontend": True,
            "MaleCNS": True,
            "globalGain": gain,
        },
        "plannerUsedForTraining": False,
        "budget": {
            "updates": args.updates,
            "rolloutSteps": args.rollout_steps,
            "environmentSteps": total_env_steps,
        },
        "baselineDev": baseline_dev,
        "bestUpdate": best_update,
        "bestDev": best_dev,
        "trainingEpisodesCompleted": len(train_episode_scores),
        "trainingEpisodeSummary": {
            "meanReturn": (
                float(np.mean(train_episode_returns))
                if train_episode_returns else None
            ),
            "meanScore": (
                float(np.mean(train_episode_scores))
                if train_episode_scores else None
            ),
            "meanLength": (
                float(np.mean(train_episode_lengths))
                if train_episode_lengths else None
            ),
            "terminalReasons": dict(sorted(train_terminals.items())),
        },
        "updates": updates_report,
        "elapsedSeconds": time.perf_counter() - started,
        "nextStep": (
            "Run the existing decoder_eval with --checkpoint decoder-ppo-r1.pt. "
            "Do not perform another PPO run before the 50-seed closed-loop result is reviewed."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nPPO READOUT R1 TRAINING: COMPLETE", flush=True)
    print(f"best update: {best_update}", flush=True)
    print(
        f"best dev mean={best_dev['meanScore']:.2f} "
        f"median={best_dev['medianScore']:.2f} "
        f"limit={best_dev['reachedStepLimit']}/{args.dev_episodes}",
        flush=True,
    )
    print(f"checkpoint: {args.checkpoint.resolve()}", flush=True)
    print(f"report:     {args.output.resolve()}", flush=True)
    print(
        "\nNext gate: existing decoder_eval on 50 unseen final seeds. "
        "No second PPO run before that result.",
        flush=True,
    )


if __name__ == "__main__":
    main()
