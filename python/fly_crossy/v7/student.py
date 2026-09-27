from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from fly_crossy.connectome import ReducedGraphArtifact, load_reduced_graph_variant
from fly_crossy.models import GatedNestedPopulationFixedGraphPolicy
from fly_crossy.v4.policy import CHANNELS, IMAGE_H, IMAGE_W, FullMaleCNSRGBPolicy
from fly_crossy.v6.structured_retina_core import (
    RETINA_CHANNELS,
    _initial_retina_kernel,
    _structured_slots,
)


@dataclass(frozen=True, slots=True)
class StudentOutput:
    logits: Tensor
    neuron_activity: Tensor
    next_recurrent_state: Tensor


class StructuredLocalRetina(nn.Module):
    """Shared local RGB filters sampled at deterministic retinal positions."""

    def __init__(self, sensory_count: int, channels: int = RETINA_CHANNELS) -> None:
        super().__init__()
        if sensory_count <= 0:
            raise ValueError("sensory_count must be positive")
        if channels != RETINA_CHANNELS:
            raise ValueError(f"Structured retina requires {RETINA_CHANNELS} channels")
        channel, yy, xx = _structured_slots(sensory_count)
        self.register_buffer("retina_channel", torch.as_tensor(channel, dtype=torch.long))
        self.register_buffer("retina_y", torch.as_tensor(yy, dtype=torch.long))
        self.register_buffer("retina_x", torch.as_tensor(xx, dtype=torch.long))
        self.retina_kernel = nn.Parameter(_initial_retina_kernel())
        self.retina_bias = nn.Parameter(torch.zeros(channels, dtype=torch.float32))
        self.sensory_gain = nn.Parameter(torch.zeros(sensory_count, dtype=torch.float32))
        self.sensory_bias = nn.Parameter(torch.zeros(sensory_count, dtype=torch.float32))

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 4 or tuple(frames.shape[1:]) != (IMAGE_H, IMAGE_W, CHANNELS):
            raise ValueError(f"Expected RGB frames [B,{IMAGE_H},{IMAGE_W},{CHANNELS}].")
        pixels = ((frames - 0.5) * 2.0).clamp(-2.0, 2.0).permute(0, 3, 1, 2)
        maps = torch.tanh(
            F.conv2d(pixels, self.retina_kernel, self.retina_bias, padding=1)
        )
        sampled = maps[:, self.retina_channel, self.retina_y, self.retina_x]
        gain = 1.0 + 0.25 * torch.tanh(self.sensory_gain)[None, :]
        bias = 0.10 * torch.tanh(self.sensory_bias)[None, :]
        return sampled * gain + bias


def _motor_head(readout_count: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(readout_count, 64), nn.Tanh(), nn.Linear(64, 5))


class ReducedVisualConnectomeStudent(nn.Module):
    def __init__(
        self,
        graph: ReducedGraphArtifact,
        core_graph: ReducedGraphArtifact,
    ) -> None:
        super().__init__()
        sensory_count = len(graph.sensory_body_ids)
        self.retina = StructuredLocalRetina(sensory_count)
        self.dynamics = GatedNestedPopulationFixedGraphPolicy(
            graph, core_graph, observation_size=sensory_count
        )
        # V7 injects the local retinal drive directly. Remove the legacy global
        # observation projection and freeze legacy heads that are not consulted.
        self.dynamics.sensory = nn.Identity()
        for parameter in self.dynamics.actor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.dynamics.critic.parameters():
            parameter.requires_grad_(False)
        self.motor_head = _motor_head(len(self.dynamics.readout_indices))

    def zero_state(self, batch: int, *, device=None) -> Tensor:
        if batch <= 0:
            raise ValueError("batch must be positive")
        return torch.zeros(
            batch,
            self.dynamics.graph.node_count,
            dtype=torch.float32,
            device=device or self.retina.retina_kernel.device,
        )

    def forward(self, frames: Tensor, recurrent_state: Tensor) -> StudentOutput:
        sensory_drive = self.retina(frames)
        _, _, activity = self.dynamics.forward_sensory_drive(
            sensory_drive, recurrent_state
        )
        readout = activity.index_select(1, self.dynamics.readout_indices)
        return StudentOutput(
            logits=self.motor_head(readout),
            neuron_activity=activity,
            next_recurrent_state=activity,
        )


class FullVisualConnectomeStudent(nn.Module):
    def __init__(self, base: FullMaleCNSRGBPolicy) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.retina = StructuredLocalRetina(len(base.sensory_ids))
        self.motor_head = _motor_head(len(base.motor_ids))

    def zero_state(self, batch: int, *, device=None) -> Tensor:
        if batch <= 0:
            raise ValueError("batch must be positive")
        return torch.zeros(
            batch,
            self.base.core.n,
            dtype=torch.float32,
            device=device or self.retina.retina_kernel.device,
        )

    def forward(self, frames: Tensor, recurrent_state: Tensor) -> StudentOutput:
        sensory_drive = self.retina(frames)
        expected = (len(frames), self.base.core.n)
        if recurrent_state.shape != expected:
            raise ValueError("Full connectome recurrent state has an incompatible shape.")
        drive = torch.zeros(
            self.base.core.n,
            len(frames),
            dtype=sensory_drive.dtype,
            device=sensory_drive.device,
        )
        drive[self.base.sensory_ids] = sensory_drive.T
        next_state = self.base.core(recurrent_state.T, steps=4, drive=drive).T
        motor = next_state.index_select(1, self.base.motor_ids)
        return StudentOutput(
            logits=self.motor_head(motor),
            neuron_activity=next_state,
            next_recurrent_state=next_state,
        )


def _load_full_policy(flyhard_root: Path) -> FullMaleCNSRGBPolicy:
    try:
        import pyarrow.feather as feather
    except ImportError as error:  # pragma: no cover - production dependency guard
        raise RuntimeError("pyarrow is required to load the full MaleCNS profile") from error

    graph_root = flyhard_root / "data" / "graph-traced-v1"
    graph_path = graph_root / "graph.npz"
    nodes_path = graph_root / "nodes.feather"
    if not graph_path.is_file() or not nodes_path.is_file():
        raise FileNotFoundError(
            "Full traced MaleCNS graph missing. Expected "
            "flyhard/data/graph-traced-v1/{graph.npz,nodes.feather}."
        )
    graph = dict(np.load(graph_path))
    nodes = feather.read_table(nodes_path)
    classes = np.asarray(nodes["superclass"].fill_null("").to_pylist())
    sensory_ids = np.flatnonzero(classes == "ol_sensory")
    motor_ids = np.flatnonzero(classes == "vnc_motor")
    return FullMaleCNSRGBPolicy(graph, sensory_ids, motor_ids)


def build_visual_student(
    profile: Literal["80", "1k", "full"],
    *,
    device: torch.device,
    flyhard_root: Path | None = None,
) -> nn.Module:
    if profile in ("80", "1k"):
        graph = load_reduced_graph_variant(profile)
        core = load_reduced_graph_variant("80")
        return ReducedVisualConnectomeStudent(graph, core).to(device)
    if profile == "full":
        if flyhard_root is None:
            raise ValueError("flyhard_root is required for the full profile")
        return FullVisualConnectomeStudent(_load_full_policy(flyhard_root)).to(device)
    raise ValueError(f"unknown visual student profile: {profile}")
