from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from fly_crossy.env import GameState
from fly_crossy.v2.preference_distill import preference_loss

from .contracts import ACTION_NAMES, OBSERVATION_SIZE
from .dataset import BehaviorPolicy, TransitionDataset


class PrivilegedTeacher(nn.Module):
    def __init__(
        self,
        observation_size: int = OBSERVATION_SIZE,
        hidden_size: int = 128,
        actions: int = len(ACTION_NAMES),
        *,
        action_order: tuple[str, ...] = ACTION_NAMES,
    ) -> None:
        super().__init__()
        if observation_size != OBSERVATION_SIZE:
            raise ValueError(f"teacher observation size must be {OBSERVATION_SIZE}")
        if hidden_size <= 0:
            raise ValueError("teacher hidden size must be positive")
        if actions != len(ACTION_NAMES) or action_order != ACTION_NAMES:
            raise ValueError("teacher actions must match canonical ACTION_ORDER")
        self.hidden_1 = nn.Linear(observation_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, actions)

    def forward(self, observations: Tensor) -> Tensor:
        if observations.ndim != 2 or observations.shape[1] != OBSERVATION_SIZE:
            raise ValueError(
                f"teacher input must have shape [B,{OBSERVATION_SIZE}]"
            )
        hidden = torch.tanh(self.hidden_1(observations))
        hidden = torch.tanh(self.hidden_2(hidden))
        return self.output(hidden)


@dataclass(frozen=True, slots=True)
class TeacherLoss:
    total: Tensor
    preference: Tensor
    acceptable: Tensor
    safety: Tensor


def _tensor(array: np.ndarray, rows: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(array[rows], device=device)


def teacher_loss(
    logits: Tensor,
    dataset: TransitionDataset,
    rows: np.ndarray,
) -> TeacherLoss:
    dataset.validate()
    if logits.shape != (len(rows), len(ACTION_NAMES)):
        raise ValueError("teacher logits have an incompatible shape")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("teacher logits contain non-finite values")
    device = logits.device
    total, components = preference_loss(
        logits,
        _tensor(dataset.acceptable, rows, device).bool(),
        _tensor(dataset.immediate_safe, rows, device).bool(),
        _tensor(dataset.pair_sign, rows, device).to(logits.dtype),
        _tensor(dataset.pair_weight, rows, device).to(logits.dtype),
        torch.ones(len(rows), dtype=logits.dtype, device=device),
    )
    if not torch.isfinite(total):
        raise FloatingPointError("teacher loss is non-finite")
    return TeacherLoss(
        total=total,
        preference=components["pair"],
        acceptable=components["set"],
        safety=components["fatal"],
    )


def fit_teacher_epoch(
    model: PrivilegedTeacher,
    dataset: TransitionDataset,
    optimizer: Optimizer,
    *,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    dataset.validate()
    if batch_size <= 0:
        raise ValueError("teacher batch size must be positive")
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset.frames))
    totals = {name: 0.0 for name in ("total", "preference", "acceptable", "safety")}
    samples = 0
    model.train()
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        observations = _tensor(dataset.observations, rows, device).float()
        optimizer.zero_grad(set_to_none=True)
        loss = teacher_loss(model(observations), dataset, rows)
        loss.total.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("teacher gradients contain non-finite values")
        optimizer.step()
        count = len(rows)
        samples += count
        for name in totals:
            totals[name] += float(getattr(loss, name).detach()) * count
    return {name: value / samples for name, value in totals.items()}


class _TeacherBehavior:
    def __init__(self, model: PrivilegedTeacher, device: torch.device) -> None:
        self.model = model
        self.device = device

    def reset(self) -> None:
        return None

    @torch.no_grad()
    def act(
        self,
        state: GameState,
        frame: np.ndarray,
        observation: np.ndarray,
    ) -> int:
        del state, frame
        batch = torch.as_tensor(
            observation[None], dtype=torch.float32, device=self.device
        )
        self.model.eval()
        return int(self.model(batch)[0].argmax().item())


def teacher_behavior(
    model: PrivilegedTeacher,
    device: torch.device,
) -> BehaviorPolicy:
    return _TeacherBehavior(model, device)


@torch.no_grad()
def teacher_agreement(
    model: PrivilegedTeacher,
    dataset: TransitionDataset,
    device: torch.device,
) -> float:
    dataset.validate()
    model.eval()
    observations = torch.as_tensor(
        dataset.observations, dtype=torch.float32, device=device
    )
    actions = model(observations).argmax(dim=1).cpu().numpy()
    rows = np.arange(len(actions))
    return float(dataset.acceptable[rows, actions].mean())
