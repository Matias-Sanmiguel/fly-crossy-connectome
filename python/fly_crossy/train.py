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

from .connectome import ReducedGraphArtifact, load_reduced_graph_variant
from .env import (
    BLOCKED_COST,
    FlyCrossyEnv,
    PROGRESS_REWARD,
    STAGNATION_COST,
    STEP_COST,
    WAIT_COST,
    TERMINAL_PENALTY,
    WORLD_VERSION,
    hash_seed,
)
from .export import export_policy
from .models import (
    DensePolicy,
    FeedbackNestedPopulationFixedGraphPolicy,
    FixedGraphPolicy,
    GatedNestedPopulationFixedGraphPolicy,
    NestedPopulationFixedGraphPolicy,
    PopulationFixedGraphPolicy,
    TrafficAwarePopulationPolicy,
)
from .schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE
from .traffic_teacher import action_teacher_targets


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
RECURRENT_TRAINING_VERSION = "sequence-bptt-v1"
PREDICTIVE_AUXILIARY_VERSION = "traffic-next-observation-v1"
WIDE_SENSORY_COUNT = 128
PREDICTIVE_TARGET_SIZE = 132
PREDICTIVE_AUXILIARY_COEFFICIENT = 0.05
CONTROLLER_V2_SENSORY_COUNT = 128
CONTROLLER_V2_READOUT_COUNT = 128
CONTROLLER_V2_RISK_COEFFICIENT = 1.0
CONTROLLER_V2_ROUTE_COEFFICIENT = 0.5


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    controller: str
    seed: str
    steps: int
    envs: int
    learning_rate: float
    output: Path
    device: str = "auto"
    connectome_graph: str = "80"
    connectome_interface: str = "legacy"

    def validate(self) -> None:
        if self.controller not in ("conventional", "connectome"):
            raise ValueError("Controller must be conventional or connectome.")
        if self.connectome_graph not in ("80", "1k"):
            raise ValueError("Connectome graph must be 80 or 1k.")
        if self.connectome_interface not in (
            "legacy",
            "population",
            "population-wide-predictive",
            "controller-v2",
            "nested",
            "nested-gated",
            "nested-feedback",
        ):
            raise ValueError(
                "Connectome interface must be legacy, population, "
                "population-wide-predictive, controller-v2, nested, "
                "nested-gated, or nested-feedback."
            )
        if (
            self.connectome_interface in (
                "population-wide-predictive",
                "controller-v2",
            )
            and self.connectome_graph != "1k"
        ):
            raise ValueError(
                "population-wide-predictive and controller-v2 require the 1k graph."
            )
        if self.controller != "connectome" and (
            self.connectome_graph != "80"
            or self.connectome_interface != "legacy"
        ):
            raise ValueError(
                "Non-connectome controllers must keep default connectome selectors."
            )
        if not self.seed:
            raise ValueError("Training seed must not be empty.")
        if self.steps <= 0 or self.envs <= 0:
            raise ValueError("Training steps and environment count must be positive.")
        if self.steps % self.envs != 0:
            raise ValueError("Training steps must be divisible by the environment count.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite.")



def _build_wide_sensory_graph(
    graph: ReducedGraphArtifact,
    target_count: int = WIDE_SENSORY_COUNT,
) -> ReducedGraphArtifact:
    """Derive a wider input interface without using game outcomes.

    Keep every historical sensory cell, exclude declared readout cells, and add
    bidirectionally well-connected graph hubs until target_count is reached.
    Node/edge topology and recurrent weights stay unchanged.
    """
    original = [int(value) for value in graph.sensory_body_ids]
    if len(original) >= target_count:
        return graph
    if graph.node_count < target_count:
        raise ValueError("Graph is too small for the requested sensory population.")

    body_ids = [int(value) for value in graph.body_ids]
    readout_ids = set(int(value) for value in graph.readout_body_ids)
    original_set = set(original)

    incoming = np.zeros(graph.node_count, dtype=np.float64)
    outgoing = np.zeros(graph.node_count, dtype=np.float64)
    for source, target, weight in zip(
        graph.edge_index[0],
        graph.edge_index[1],
        graph.edge_weight,
        strict=True,
    ):
        magnitude = abs(float(weight))
        outgoing[int(source)] += magnitude
        incoming[int(target)] += magnitude

    candidates = [
        index
        for index, body_id in enumerate(body_ids)
        if body_id not in original_set and body_id not in readout_ids
    ]
    candidates.sort(
        key=lambda index: (
            -min(incoming[index], outgoing[index]),
            -(incoming[index] + outgoing[index]),
            -outgoing[index],
            body_ids[index],
        )
    )
    needed = target_count - len(original)
    if len(candidates) < needed:
        raise ValueError("Not enough non-readout graph cells for wide sensory input.")

    added = [body_ids[index] for index in candidates[:needed]]
    sensory_ids = np.asarray([*original, *added], dtype=np.int64)

    return ReducedGraphArtifact(
        dataset_version=graph.dataset_version,
        source_url=graph.source_url,
        license=graph.license,
        source_sha256=graph.source_sha256,
        selection_rule=(
            graph.selection_rule
            + " Wide predictive interface: retain the historical sensory cells "
            + f"and add {needed} non-readout cells using graph topology only, "
            + "ranked by bidirectional weighted degree, then total weighted "
            + "degree, outgoing weighted degree, and ascending body ID. "
            + "No Fly Crossy rewards, scores, policies, training seeds, "
            + "evaluation seeds, or outcomes are used for this interface selection."
        ),
        minimum_edge_threshold=graph.minimum_edge_threshold,
        body_ids=graph.body_ids.copy(),
        edge_index=graph.edge_index.copy(),
        edge_weight=graph.edge_weight.copy(),
        sensory_body_ids=sensory_ids,
        readout_body_ids=graph.readout_body_ids.copy(),
    )



def _build_controller_v2_graph(
    graph: ReducedGraphArtifact,
) -> ReducedGraphArtifact:
    """Build the 128-input / 128-decision interface from topology only."""
    graph = _build_wide_sensory_graph(
        graph,
        target_count=CONTROLLER_V2_SENSORY_COUNT,
    )
    original_readout = [int(value) for value in graph.readout_body_ids]
    if len(original_readout) >= CONTROLLER_V2_READOUT_COUNT:
        return graph

    body_ids = [int(value) for value in graph.body_ids]
    sensory_ids = set(int(value) for value in graph.sensory_body_ids)
    readout_set = set(original_readout)
    incoming = np.zeros(graph.node_count, dtype=np.float64)
    outgoing = np.zeros(graph.node_count, dtype=np.float64)

    for source, target, weight in zip(
        graph.edge_index[0],
        graph.edge_index[1],
        graph.edge_weight,
        strict=True,
    ):
        magnitude = abs(float(weight))
        outgoing[int(source)] += magnitude
        incoming[int(target)] += magnitude

    candidates = [
        index
        for index, body_id in enumerate(body_ids)
        if body_id not in sensory_ids and body_id not in readout_set
    ]
    candidates.sort(
        key=lambda index: (
            -incoming[index],
            -min(incoming[index], outgoing[index]),
            -(incoming[index] + outgoing[index]),
            body_ids[index],
        )
    )
    needed = CONTROLLER_V2_READOUT_COUNT - len(original_readout)
    if len(candidates) < needed:
        raise ValueError("Not enough graph cells for Controller V2 readout.")

    readout_ids = np.asarray(
        [*original_readout, *[body_ids[index] for index in candidates[:needed]]],
        dtype=np.int64,
    )
    return ReducedGraphArtifact(
        dataset_version=graph.dataset_version,
        source_url=graph.source_url,
        license=graph.license,
        source_sha256=graph.source_sha256,
        selection_rule=(
            graph.selection_rule
            + " Controller V2 expands the decision population to 128 non-sensory "
            + "cells using graph topology only: incoming weighted degree, "
            + "bidirectional weighted degree, total weighted degree, and body ID."
        ),
        minimum_edge_threshold=graph.minimum_edge_threshold,
        body_ids=graph.body_ids.copy(),
        edge_index=graph.edge_index.copy(),
        edge_weight=graph.edge_weight.copy(),
        sensory_body_ids=graph.sensory_body_ids.copy(),
        readout_body_ids=readout_ids,
    )

def _predictive_traffic_target(observations: Tensor) -> Tensor:
    """Extract next-step local-scene targets from ObservationV3.

    Observation rows 6..8 are the three rows immediately ahead. For their 33
    cells predict normalized cell code, two motion values, and sub-cell hazard
    offset: 33 + 66 + 33 = 132 values.
    """
    if observations.shape[-1] != OBSERVATION_INPUT_SIZE:
        raise ValueError("Predictive target received an incompatible observation.")
    if OBSERVATION_INPUT_SIZE != 517:
        raise ValueError("Predictive target requires ObservationV4 (517 values).")

    prefix = observations.shape[:-1]
    cells = observations[..., :121].reshape(*prefix, 11, 11)
    motion = observations[..., 121:363].reshape(*prefix, 11, 11, 2)
    offsets = observations[..., 363:484].reshape(*prefix, 11, 11)

    selected_cells = cells[..., 6:9, :].flatten(start_dim=-2)
    selected_motion = motion[..., 6:9, :, :].flatten(start_dim=-3)
    selected_offsets = offsets[..., 6:9, :].flatten(start_dim=-2)
    target = torch.cat(
        (selected_cells, selected_motion, selected_offsets),
        dim=-1,
    )
    if target.shape[-1] != PREDICTIVE_TARGET_SIZE:
        raise RuntimeError("Predictive traffic target has an unexpected width.")
    return target

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
    model: DensePolicy | FixedGraphPolicy | PopulationFixedGraphPolicy | NestedPopulationFixedGraphPolicy | GatedNestedPopulationFixedGraphPolicy | FeedbackNestedPopulationFixedGraphPolicy,
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
            if isinstance(model, (FixedGraphPolicy, PopulationFixedGraphPolicy, NestedPopulationFixedGraphPolicy, GatedNestedPopulationFixedGraphPolicy, FeedbackNestedPopulationFixedGraphPolicy)):
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



def _ppo_update_recurrent(
    model: FixedGraphPolicy
    | PopulationFixedGraphPolicy
    | NestedPopulationFixedGraphPolicy
    | GatedNestedPopulationFixedGraphPolicy
    | FeedbackNestedPopulationFixedGraphPolicy,
    optimizer: torch.optim.Optimizer,
    observations: Tensor,
    actions: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    returns: Tensor,
    dones: Tensor,
    initial_hidden_state: Tensor,
    next_observations: Tensor | None = None,
    traffic_predictor: torch.nn.Linear | None = None,
) -> dict[str, float]:
    """Sequence-aware PPO update with truncated BPTT through each rollout."""
    if observations.ndim != 3:
        raise ValueError(
            "Recurrent PPO observations must have shape [time, env, input]."
        )
    if actions.shape != observations.shape[:2]:
        raise ValueError("Recurrent PPO actions must have shape [time, env].")
    if old_log_probabilities.shape != actions.shape:
        raise ValueError("Recurrent PPO log probabilities must match actions.")
    if advantages.shape != actions.shape or returns.shape != actions.shape:
        raise ValueError("Recurrent PPO advantages and returns must match actions.")
    if dones.shape != actions.shape:
        raise ValueError("Recurrent PPO dones must match actions.")
    if initial_hidden_state.ndim != 2:
        raise ValueError(
            "Recurrent PPO initial hidden state must have shape [env, node]."
        )
    if initial_hidden_state.shape[0] != observations.shape[1]:
        raise ValueError(
            "Recurrent PPO hidden-state environment count is incompatible."
        )
    if traffic_predictor is not None:
        if not isinstance(model, PopulationFixedGraphPolicy):
            raise ValueError(
                "Traffic prediction auxiliary is supported only by the population policy."
            )
        if next_observations is None or next_observations.shape != observations.shape:
            raise ValueError(
                "Traffic prediction requires next observations matching the rollout."
            )

    normalized_advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    time_steps = observations.shape[0]
    environment_count = observations.shape[1]
    environments_per_minibatch = max(
        1,
        MINIBATCH_SIZE // max(1, time_steps),
    )

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approximate_kls: list[float] = []
    traffic_prediction_losses: list[float] = []

    for _ in range(PPO_EPOCHS):
        for environment_indices in torch.randperm(
            environment_count,
            device=observations.device,
        ).split(environments_per_minibatch):
            hidden = initial_hidden_state[environment_indices].clone()
            logits_by_step: list[Tensor] = []
            values_by_step: list[Tensor] = []
            traffic_predictions_by_step: list[Tensor] = []

            for timestep in range(time_steps):
                logits, values, next_hidden = model(
                    observations[timestep, environment_indices],
                    hidden,
                )
                logits_by_step.append(logits)
                values_by_step.append(values)

                if traffic_predictor is not None:
                    assert isinstance(model, PopulationFixedGraphPolicy)
                    readout_activity = next_hidden.index_select(
                        1, model.readout_indices
                    )
                    selected_action = actions[timestep, environment_indices]
                    action_one_hot = torch.nn.functional.one_hot(
                        selected_action,
                        num_classes=len(ACTION_ORDER),
                    ).to(dtype=readout_activity.dtype)
                    traffic_predictions_by_step.append(
                        traffic_predictor(
                            torch.cat(
                                (readout_activity, action_one_hot),
                                dim=1,
                            )
                        )
                    )

                alive = (
                    1.0 - dones[timestep, environment_indices]
                ).unsqueeze(1)
                hidden = next_hidden * alive

            logits = torch.stack(logits_by_step)
            values = torch.stack(values_by_step)
            selected_actions = actions[:, environment_indices]

            if traffic_predictor is not None:
                assert next_observations is not None
                predictions = torch.stack(traffic_predictions_by_step)
                targets = _predictive_traffic_target(
                    next_observations[:, environment_indices]
                )
                alive_mask = (
                    1.0 - dones[:, environment_indices]
                ).unsqueeze(-1)
                prediction_error = torch.nn.functional.smooth_l1_loss(
                    predictions,
                    targets,
                    reduction="none",
                )
                prediction_denominator = torch.clamp(
                    alive_mask.sum() * PREDICTIVE_TARGET_SIZE,
                    min=1.0,
                )
                traffic_prediction_loss = (
                    prediction_error * alive_mask
                ).sum() / prediction_denominator
            else:
                traffic_prediction_loss = torch.zeros(
                    (),
                    dtype=values.dtype,
                    device=values.device,
                )

            distribution = Categorical(logits=logits)
            new_log_probabilities = distribution.log_prob(selected_actions)

            old_selected_log_probabilities = old_log_probabilities[
                :, environment_indices
            ]
            selected_advantages = normalized_advantages[:, environment_indices]
            selected_returns = returns[:, environment_indices]

            log_ratio = (
                new_log_probabilities - old_selected_log_probabilities
            )
            ratio = log_ratio.exp()
            unclipped = -selected_advantages * ratio
            clipped = -selected_advantages * torch.clamp(
                ratio,
                1 - CLIP_COEFFICIENT,
                1 + CLIP_COEFFICIENT,
            )
            policy_loss = torch.maximum(unclipped, clipped).mean()
            value_loss = 0.5 * (
                values - selected_returns
            ).square().mean()
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + VALUE_COEFFICIENT * value_loss
                - ENTROPY_COEFFICIENT * entropy
                + PREDICTIVE_AUXILIARY_COEFFICIENT * traffic_prediction_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_parameters = list(model.parameters())
            if traffic_predictor is not None:
                gradient_parameters.extend(traffic_predictor.parameters())
            torch.nn.utils.clip_grad_norm_(
                gradient_parameters,
                MAX_GRADIENT_NORM,
            )
            optimizer.step()

            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            traffic_prediction_losses.append(
                float(traffic_prediction_loss.detach().cpu())
            )
            approximate_kls.append(
                float(
                    ((ratio - 1) - log_ratio)
                    .mean()
                    .detach()
                    .cpu()
                )
            )

    return {
        "policyLoss": _mean(policy_losses),
        "valueLoss": _mean(value_losses),
        "entropy": _mean(entropies),
        "approximateKl": _mean(approximate_kls),
        "trafficPredictionLoss": _mean(traffic_prediction_losses),
    }


def _ppo_update_controller_v2(
    model: TrafficAwarePopulationPolicy,
    optimizer: torch.optim.Optimizer,
    observations: Tensor,
    actions: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    returns: Tensor,
    dones: Tensor,
    initial_hidden_state: Tensor,
    teacher_risk_targets: Tensor,
    teacher_route_targets: Tensor,
) -> dict[str, float]:
    expected = (*actions.shape, len(ACTION_ORDER))
    if observations.ndim != 3:
        raise ValueError("Controller V2 observations must be [time, env, input].")
    if teacher_risk_targets.shape != expected:
        raise ValueError("Controller V2 risk targets have an incompatible shape.")
    if teacher_route_targets.shape != expected:
        raise ValueError("Controller V2 route targets have an incompatible shape.")

    normalized_advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    time_steps = observations.shape[0]
    environment_count = observations.shape[1]
    environments_per_minibatch = max(1, MINIBATCH_SIZE // max(1, time_steps))

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approximate_kls: list[float] = []
    risk_losses: list[float] = []
    route_losses: list[float] = []
    risk_accuracies: list[float] = []

    for _ in range(PPO_EPOCHS):
        for environment_indices in torch.randperm(
            environment_count,
            device=observations.device,
        ).split(environments_per_minibatch):
            hidden = initial_hidden_state[environment_indices].clone()
            logits_by_step: list[Tensor] = []
            values_by_step: list[Tensor] = []
            risk_by_step: list[Tensor] = []
            route_by_step: list[Tensor] = []

            for timestep in range(time_steps):
                logits, values, next_hidden = model(
                    observations[timestep, environment_indices],
                    hidden,
                )
                risk_logits, route_logits = model.auxiliary_from_activity(next_hidden)
                logits_by_step.append(logits)
                values_by_step.append(values)
                risk_by_step.append(risk_logits)
                route_by_step.append(route_logits)
                alive = (1.0 - dones[timestep, environment_indices]).unsqueeze(1)
                hidden = next_hidden * alive

            logits = torch.stack(logits_by_step)
            values = torch.stack(values_by_step)
            risk_logits = torch.stack(risk_by_step)
            route_logits = torch.stack(route_by_step)
            selected_actions = actions[:, environment_indices]
            distribution = Categorical(logits=logits)
            new_log_probabilities = distribution.log_prob(selected_actions)
            old_selected = old_log_probabilities[:, environment_indices]
            selected_advantages = normalized_advantages[:, environment_indices]
            selected_returns = returns[:, environment_indices]

            log_ratio = new_log_probabilities - old_selected
            ratio = log_ratio.exp()
            unclipped = -selected_advantages * ratio
            clipped = -selected_advantages * torch.clamp(
                ratio,
                1 - CLIP_COEFFICIENT,
                1 + CLIP_COEFFICIENT,
            )
            policy_loss = torch.maximum(unclipped, clipped).mean()
            value_loss = 0.5 * (values - selected_returns).square().mean()
            entropy = distribution.entropy().mean()

            risk_target = teacher_risk_targets[:, environment_indices]
            route_target = teacher_route_targets[:, environment_indices]
            risk_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                risk_logits,
                risk_target,
            )
            route_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                route_logits,
                route_target,
            )
            loss = (
                policy_loss
                + VALUE_COEFFICIENT * value_loss
                - ENTROPY_COEFFICIENT * entropy
                + CONTROLLER_V2_RISK_COEFFICIENT * risk_loss
                + CONTROLLER_V2_ROUTE_COEFFICIENT * route_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRADIENT_NORM)
            optimizer.step()

            predicted_risk = torch.sigmoid(risk_logits) >= 0.5
            risk_accuracy = (
                predicted_risk == (risk_target >= 0.5)
            ).float().mean()
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            risk_losses.append(float(risk_loss.detach().cpu()))
            route_losses.append(float(route_loss.detach().cpu()))
            risk_accuracies.append(float(risk_accuracy.detach().cpu()))
            approximate_kls.append(
                float((((ratio - 1) - log_ratio).mean()).detach().cpu())
            )

    return {
        "policyLoss": _mean(policy_losses),
        "valueLoss": _mean(value_losses),
        "entropy": _mean(entropies),
        "approximateKl": _mean(approximate_kls),
        "teacherRiskLoss": _mean(risk_losses),
        "teacherRouteLoss": _mean(route_losses),
        "teacherRiskAccuracy": _mean(risk_accuracies),
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
    traffic_predictor: torch.nn.Linear | None = None

    if config.controller == "connectome":
        graph = load_reduced_graph_variant(config.connectome_graph)
        if config.connectome_interface == "population-wide-predictive":
            graph = _build_wide_sensory_graph(graph)
        elif config.connectome_interface == "controller-v2":
            graph = _build_controller_v2_graph(graph)
        if config.connectome_interface == "legacy":
            model = FixedGraphPolicy(
                graph,
                OBSERVATION_INPUT_SIZE,
                len(ACTION_ORDER),
            ).to(device)
        elif config.connectome_interface == "controller-v2":
            model = TrafficAwarePopulationPolicy(
                graph,
                OBSERVATION_INPUT_SIZE,
                len(ACTION_ORDER),
            ).to(device)
        elif config.connectome_interface in (
            "population",
            "population-wide-predictive",
        ):
            model = PopulationFixedGraphPolicy(
                graph,
                OBSERVATION_INPUT_SIZE,
                len(ACTION_ORDER),
            ).to(device)
        elif config.connectome_interface == "nested":
            model = NestedPopulationFixedGraphPolicy(
                graph,
                load_reduced_graph_variant("80"),
                OBSERVATION_INPUT_SIZE,
                len(ACTION_ORDER),
            ).to(device)
        elif config.connectome_interface == "nested-gated":
            model = GatedNestedPopulationFixedGraphPolicy(
                graph,
                load_reduced_graph_variant("80"),
                OBSERVATION_INPUT_SIZE,
                len(ACTION_ORDER),
            ).to(device)
        else:
            model = FeedbackNestedPopulationFixedGraphPolicy(
                graph,
                load_reduced_graph_variant("80"),
                OBSERVATION_INPUT_SIZE,
                len(ACTION_ORDER),
            ).to(device)

        hidden_state: Tensor | None = torch.zeros(
            config.envs, model.graph.node_count, dtype=torch.float32, device=device
        )
        if config.connectome_interface == "population-wide-predictive":
            traffic_predictor = torch.nn.Linear(
                len(model.readout_indices) + len(ACTION_ORDER),
                PREDICTIVE_TARGET_SIZE,
            ).to(device)
    else:
        model = DensePolicy(
            OBSERVATION_INPUT_SIZE, HIDDEN_SIZE, len(ACTION_ORDER)
        ).to(device)
        hidden_state = None
    optimizer_parameters = list(model.parameters())
    if traffic_predictor is not None:
        optimizer_parameters.extend(traffic_predictor.parameters())
    optimizer = torch.optim.Adam(
        optimizer_parameters,
        lr=config.learning_rate,
    )
    total_steps = 0
    update_index = 0
    training_curve: list[dict[str, Any]] = []

    while total_steps < config.steps:
        rollout_length = min(ROLLOUT_STEPS, (config.steps - total_steps) // config.envs)
        rollout_initial_hidden_state = (
            hidden_state.clone()
            if hidden_state is not None
            else None
        )
        rollout_observations: list[Tensor] = []
        rollout_next_observations: list[Tensor] = []
        rollout_actions: list[Tensor] = []
        rollout_log_probabilities: list[Tensor] = []
        rollout_rewards: list[Tensor] = []
        rollout_dones: list[Tensor] = []
        rollout_values: list[Tensor] = []
        rollout_teacher_risk: list[Tensor] = []
        rollout_teacher_route: list[Tensor] = []
        episode_start = len(completed_episodes)

        for _ in range(rollout_length):
            if isinstance(model, TrafficAwarePopulationPolicy):
                teacher_rows = [
                    action_teacher_targets(environment.state)
                    for environment in environments
                ]
                rollout_teacher_risk.append(
                    torch.tensor(
                        [[float(target.short_horizon_risk) for target in row] for row in teacher_rows],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                rollout_teacher_route.append(
                    torch.tensor(
                        [[float(target.opens_safe_forward) for target in row] for row in teacher_rows],
                        dtype=torch.float32,
                        device=device,
                    )
                )

            observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
            with torch.no_grad():
                if isinstance(model, (FixedGraphPolicy, PopulationFixedGraphPolicy, NestedPopulationFixedGraphPolicy, GatedNestedPopulationFixedGraphPolicy, FeedbackNestedPopulationFixedGraphPolicy)):
                    assert hidden_state is not None
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
            next_observation_array = np.stack(next_observations)
            rollout_next_observations.append(
                torch.as_tensor(
                    next_observation_array,
                    dtype=torch.float32,
                    device=device,
                )
            )
            observations = next_observation_array
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
            if isinstance(model, (FixedGraphPolicy, PopulationFixedGraphPolicy, NestedPopulationFixedGraphPolicy, GatedNestedPopulationFixedGraphPolicy, FeedbackNestedPopulationFixedGraphPolicy)):
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

        if isinstance(model, TrafficAwarePopulationPolicy):
            if rollout_initial_hidden_state is None:
                raise RuntimeError("Controller V2 rollout has no initial hidden state.")
            update_metrics = _ppo_update_controller_v2(
                model,
                optimizer,
                torch.stack(rollout_observations),
                torch.stack(rollout_actions),
                torch.stack(rollout_log_probabilities),
                advantages,
                returns,
                dones_tensor,
                rollout_initial_hidden_state,
                torch.stack(rollout_teacher_risk),
                torch.stack(rollout_teacher_route),
            )
        elif isinstance(
            model,
            (
                FixedGraphPolicy,
                PopulationFixedGraphPolicy,
                NestedPopulationFixedGraphPolicy,
                GatedNestedPopulationFixedGraphPolicy,
                FeedbackNestedPopulationFixedGraphPolicy,
            ),
        ):
            if rollout_initial_hidden_state is None:
                raise RuntimeError(
                    "Recurrent rollout has no initial hidden state."
                )
            update_metrics = _ppo_update_recurrent(
                model,
                optimizer,
                torch.stack(rollout_observations),
                torch.stack(rollout_actions),
                torch.stack(rollout_log_probabilities),
                advantages,
                returns,
                dones_tensor,
                rollout_initial_hidden_state,
                next_observations=torch.stack(rollout_next_observations),
                traffic_predictor=traffic_predictor,
            )
        else:
            update_metrics = _ppo_update(
                model,
                optimizer,
                torch.stack(rollout_observations).flatten(0, 1),
                torch.stack(rollout_actions).flatten(),
                torch.stack(rollout_log_probabilities).flatten(),
                advantages.flatten(),
                returns.flatten(),
                None,
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

    if isinstance(model, (FixedGraphPolicy, PopulationFixedGraphPolicy, NestedPopulationFixedGraphPolicy, GatedNestedPopulationFixedGraphPolicy, FeedbackNestedPopulationFixedGraphPolicy)):
        model_metadata = {
            "observation_size": OBSERVATION_INPUT_SIZE,
            "actions": len(ACTION_ORDER),
            "graph": model.graph.to_checkpoint(),
            "interface": config.connectome_interface,
            **(
                {"core_graph": model.core_graph.to_checkpoint()}
                if isinstance(
                    model,
                    (
                        NestedPopulationFixedGraphPolicy,
                        GatedNestedPopulationFixedGraphPolicy,
                        FeedbackNestedPopulationFixedGraphPolicy,
                    ),
                )
                else {}
            ),
        }
    else:
        model_metadata = {
            "observation_size": OBSERVATION_INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "actions": len(ACTION_ORDER),
        }

    reward_metadata = {
        "version": 5,
        "progress": PROGRESS_REWARD,
        "terminal": TERMINAL_PENALTY,
        "step": STEP_COST,
        "stagnation": STAGNATION_COST,
        "wait": WAIT_COST,
        "blocked": BLOCKED_COST,
        "supportedCarryWaitExempt": True,
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
                "connectome_graph": (
                    config.connectome_graph
                    if config.controller == "connectome"
                    else None
                ),
                "connectome_interface": (
                    config.connectome_interface
                    if config.controller == "connectome"
                    else None
                ),
                "reward": reward_metadata,
                "recurrent_training": (
                    RECURRENT_TRAINING_VERSION
                    if config.controller == "connectome"
                    else None
                ),
                "predictive_auxiliary": (
                    {
                        "version": PREDICTIVE_AUXILIARY_VERSION,
                        "coefficient": PREDICTIVE_AUXILIARY_COEFFICIENT,
                        "targetRowsAhead": [1, 2, 3],
                        "targetSize": PREDICTIVE_TARGET_SIZE,
                        "sensoryCells": WIDE_SENSORY_COUNT,
                        "inferenceHeadExported": False,
                    }
                    if traffic_predictor is not None
                    else None
                ),
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
            "reward": reward_metadata,
            "recurrentTraining": (
                RECURRENT_TRAINING_VERSION
                if config.controller == "connectome"
                else None
            ),
            "predictiveAuxiliary": (
                {
                    "version": PREDICTIVE_AUXILIARY_VERSION,
                    "coefficient": PREDICTIVE_AUXILIARY_COEFFICIENT,
                    "targetRowsAhead": [1, 2, 3],
                    "targetSize": PREDICTIVE_TARGET_SIZE,
                    "sensoryCells": WIDE_SENSORY_COUNT,
                    "inferenceHeadExported": False,
                }
                if traffic_predictor is not None
                else None
            ),
            **(
                {
                    "graphNodes": model.graph.node_count,
                    "graphEdges": model.graph.edge_count,
                    "graphArtifactHash": model.graph.artifact_sha256,
                }
                if isinstance(model, (FixedGraphPolicy, PopulationFixedGraphPolicy, NestedPopulationFixedGraphPolicy, GatedNestedPopulationFixedGraphPolicy, FeedbackNestedPopulationFixedGraphPolicy))
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
    parser.add_argument(
        "--connectome-graph",
        choices=("80", "1k"),
        default="80",
        help="MaleCNS graph capacity variant used by the connectome controller.",
    )
    parser.add_argument(
        "--connectome-interface",
        choices=(
            "legacy",
            "population",
            "population-wide-predictive",
            "controller-v2",
            "nested",
            "nested-gated",
            "nested-feedback",
        ),
        default="legacy",
        help="Artificial interface mode for the fixed MaleCNS graph.",
    )
    arguments = parser.parse_args()
    return TrainingConfig(
        controller=arguments.controller,
        seed=arguments.seed,
        steps=arguments.steps,
        envs=arguments.envs,
        learning_rate=arguments.learning_rate,
        output=arguments.output,
        device=arguments.device,
        connectome_graph=arguments.connectome_graph,
        connectome_interface=arguments.connectome_interface,
    )


def _main() -> None:
    print(json.dumps(train(_parse_arguments()), indent=2))


if __name__ == "__main__":
    _main()
