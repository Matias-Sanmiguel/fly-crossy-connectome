from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import Optimizer

from fly_crossy.env import GameState
from fly_crossy.v2.preference_distill import PAIR_INDEX

from .contracts import ACTION_NAMES, FRAME_SHAPE
from .dataset import BehaviorPolicy, TransitionDataset
from .student import StudentOutput
from .teacher import PrivilegedTeacher


PLANNER_SUPERVISION_WEIGHT = 0.75
MAX_PLANNER_CLASS_WEIGHT = 4.0


@dataclass(frozen=True, slots=True)
class StudentLoss:
    total: Tensor
    distillation: Tensor
    preference: Tensor
    safety: Tensor
    planner: Tensor
    trust: Tensor


@dataclass(frozen=True, slots=True)
class StudentBatch:
    frames: Tensor
    acceptable: Tensor
    immediate_safe: Tensor
    pair_sign: Tensor
    pair_weight: Tensor
    best_action: Tensor
    best_action_weight: Tensor
    teacher_logits: Tensor
    mask: Tensor


@torch.no_grad()
def attach_teacher_logits(
    dataset: TransitionDataset,
    teacher: PrivilegedTeacher,
    device: torch.device,
) -> np.ndarray:
    dataset.validate()
    teacher.eval()
    observations = torch.as_tensor(
        dataset.observations, dtype=torch.float32, device=device
    )
    logits = teacher(observations)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("teacher logits contain non-finite values")
    return logits.detach().cpu().numpy().astype(np.float32, copy=True)


def _validate_batch(output: StudentOutput, batch: StudentBatch) -> Tensor:
    rows = output.logits.shape[0]
    if output.logits.shape != (rows, len(ACTION_NAMES)):
        raise ValueError("student logits have an incompatible shape")
    expected = {
        "frames": (rows, *FRAME_SHAPE),
        "acceptable": (rows, len(ACTION_NAMES)),
        "immediate_safe": (rows, len(ACTION_NAMES)),
        "pair_sign": (rows, len(PAIR_INDEX)),
        "pair_weight": (rows, len(PAIR_INDEX)),
        "best_action": (rows,),
        "best_action_weight": (rows,),
        "teacher_logits": (rows, len(ACTION_NAMES)),
        "mask": (rows,),
    }
    for name, shape in expected.items():
        if tuple(getattr(batch, name).shape) != shape:
            raise ValueError(f"student batch {name} has an incompatible shape")
    tensors = (
        output.logits,
        batch.teacher_logits,
        batch.pair_weight,
        batch.best_action_weight,
    )
    if any(not torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError("student loss input contains non-finite values")
    if bool(((batch.best_action < 0) | (batch.best_action >= len(ACTION_NAMES))).any()):
        raise ValueError("student batch best_action contains an invalid action")
    if bool((batch.best_action_weight <= 0).any()):
        raise ValueError("student batch best_action_weight must be positive")
    mask = batch.mask.bool()
    if not bool(mask.any()):
        raise ValueError("student batch mask must select at least one row")
    return mask


def student_loss(
    output: StudentOutput,
    batch: StudentBatch,
    *,
    trust_penalty: Tensor,
    temperature: float = 2.0,
) -> StudentLoss:
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    mask = _validate_batch(output, batch)
    if not torch.isfinite(trust_penalty):
        raise FloatingPointError("student trust penalty is non-finite")
    logits = output.logits[mask]
    teacher = batch.teacher_logits[mask].to(logits.dtype)
    distillation = (
        F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            F.softmax(teacher / temperature, dim=1),
            reduction="none",
        ).sum(dim=1).mean()
        * temperature**2
    )

    acceptable = batch.acceptable[mask].bool()
    masked_logits = logits.masked_fill(~acceptable, -1e9)
    set_loss = (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(masked_logits, dim=1)
    ).mean()
    pair_losses = []
    pair_weights = []
    signs = batch.pair_sign[mask].to(logits.dtype)
    weights = batch.pair_weight[mask].to(logits.dtype)
    for pair_index, (left, right) in enumerate(PAIR_INDEX):
        weight = weights[:, pair_index]
        margin = logits[:, left] - logits[:, right]
        pair_losses.append(F.softplus(-signs[:, pair_index] * margin) * weight)
        pair_weights.append(weight)
    pair_loss = torch.stack(pair_losses, dim=1).sum(dim=1) / (
        torch.stack(pair_weights, dim=1).sum(dim=1).clamp_min(1e-6)
    )
    preference = set_loss + 0.5 * pair_loss.mean()
    safety = (
        torch.softmax(logits, dim=1)
        * (~batch.immediate_safe[mask].bool()).to(logits.dtype)
    ).sum(dim=1).mean()
    best_action = batch.best_action[mask].long()
    best_action_weight = batch.best_action_weight[mask].to(logits.dtype)
    planner = (
        F.cross_entropy(logits, best_action, reduction="none") * best_action_weight
    ).sum() / best_action_weight.sum().clamp_min(1e-6)
    total = (
        distillation
        + preference
        + 2.0 * safety
        + PLANNER_SUPERVISION_WEIGHT * planner
        + 0.01 * trust_penalty
    )
    if not all(
        bool(torch.isfinite(value))
        for value in (
            total,
            distillation,
            preference,
            safety,
            planner,
            trust_penalty,
        )
    ):
        raise FloatingPointError("student loss is non-finite")
    return StudentLoss(
        total, distillation, preference, safety, planner, trust_penalty
    )


def _planner_row_weights(dataset: TransitionDataset) -> np.ndarray:
    counts = np.bincount(dataset.best_action, minlength=len(ACTION_NAMES)).astype(
        np.float64
    )
    present = counts > 0
    class_weights = np.zeros(len(ACTION_NAMES), dtype=np.float64)
    class_weights[present] = len(dataset.best_action) / (
        present.sum() * counts[present]
    )
    class_weights[present] = np.clip(
        class_weights[present], 0.25, MAX_PLANNER_CLASS_WEIGHT
    )
    row_weights = class_weights[dataset.best_action]
    row_weights /= row_weights.mean()
    return row_weights.astype(np.float32)


def _batch_for_rows(
    dataset: TransitionDataset,
    teacher_logits: np.ndarray,
    planner_row_weights: np.ndarray,
    rows: np.ndarray,
    device: torch.device,
) -> StudentBatch:
    return StudentBatch(
        frames=torch.as_tensor(dataset.frames[rows], dtype=torch.float32, device=device)
        / 255.0,
        acceptable=torch.as_tensor(dataset.acceptable[rows], device=device),
        immediate_safe=torch.as_tensor(dataset.immediate_safe[rows], device=device),
        pair_sign=torch.as_tensor(dataset.pair_sign[rows], device=device),
        pair_weight=torch.as_tensor(dataset.pair_weight[rows], device=device),
        best_action=torch.as_tensor(dataset.best_action[rows], device=device),
        best_action_weight=torch.as_tensor(
            planner_row_weights[rows], dtype=torch.float32, device=device
        ),
        teacher_logits=torch.as_tensor(teacher_logits[rows], device=device),
        mask=torch.ones(len(rows), dtype=torch.bool, device=device),
    )


def fit_student_epoch(
    model: nn.Module,
    dataset: TransitionDataset,
    teacher_logits: np.ndarray,
    optimizer: Optimizer,
    *,
    batch_size: int,
    window: int,
    seed: int,
) -> dict[str, float]:
    dataset.validate()
    if teacher_logits.shape != (len(dataset.frames), len(ACTION_NAMES)):
        raise ValueError("teacher logits have an incompatible shape")
    if not np.isfinite(teacher_logits).all():
        raise FloatingPointError("teacher logits contain non-finite values")
    if batch_size <= 0 or window <= 0:
        raise ValueError("student batch size and window must be positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("student has no trainable parameters")
    device = parameters[0].device
    reference = [parameter.detach().clone() for parameter in parameters]
    planner_row_weights = _planner_row_weights(dataset)
    rng = np.random.default_rng(seed)
    episode_names = np.unique(dataset.episode_seed)
    rng.shuffle(episode_names)
    totals = {
        name: 0.0
        for name in ("total", "distillation", "preference", "safety", "planner", "trust")
    }
    samples = 0
    model.train()
    effective_window = min(window, batch_size)

    for episode_name in episode_names:
        episode_rows = np.flatnonzero(dataset.episode_seed == episode_name)
        episode_rows = episode_rows[np.argsort(dataset.step_index[episode_rows])]
        recurrent_state = model.zero_state(1, device=device)
        for start in range(0, len(episode_rows), effective_window):
            rows = episode_rows[start : start + effective_window]
            outputs: list[StudentOutput] = []
            for row in rows:
                frame = torch.as_tensor(
                    dataset.frames[row : row + 1], dtype=torch.float32, device=device
                ) / 255.0
                output = model(frame, recurrent_state)
                recurrent_state = output.next_recurrent_state
                outputs.append(output)
            joined = StudentOutput(
                logits=torch.cat([item.logits for item in outputs], dim=0),
                neuron_activity=torch.cat(
                    [item.neuron_activity for item in outputs], dim=0
                ),
                next_recurrent_state=recurrent_state,
            )
            batch = _batch_for_rows(
                dataset, teacher_logits, planner_row_weights, rows, device
            )
            trust = torch.stack(
                [
                    (parameter - original).square().mean()
                    for parameter, original in zip(parameters, reference, strict=True)
                ]
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss = student_loss(joined, batch, trust_penalty=trust)
            loss.total.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in parameters
            ):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("student gradients contain non-finite values")
            optimizer.step()
            recurrent_state = recurrent_state.detach()
            count = len(rows)
            samples += count
            for name in totals:
                totals[name] += float(getattr(loss, name).detach()) * count
    return {name: value / samples for name, value in totals.items()}


class _StudentBehavior:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.recurrent_state: Tensor | None = None

    def reset(self) -> None:
        self.recurrent_state = self.model.zero_state(1, device=self.device)

    @torch.no_grad()
    def act(
        self,
        state: GameState,
        frame: np.ndarray,
        observation: np.ndarray,
    ) -> int:
        del state, observation
        if self.recurrent_state is None:
            self.reset()
        pixels = torch.as_tensor(frame[None], dtype=torch.float32, device=self.device)
        if pixels.max() > 1.0:
            pixels = pixels / 255.0
        self.model.eval()
        output = self.model(pixels, self.recurrent_state)
        self.recurrent_state = output.next_recurrent_state
        return int(output.logits[0].argmax().item())


def student_behavior(model: nn.Module, device: torch.device) -> BehaviorPolicy:
    return _StudentBehavior(model, device)
