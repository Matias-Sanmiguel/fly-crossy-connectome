"""Deterministic runtime compute-backend selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from fly_crossy.protocol import BackendPreference


class BackendUnavailable(RuntimeError):
    """Raised when a caller explicitly requires an unavailable accelerator."""


@dataclass(frozen=True, slots=True)
class BackendStatus:
    requested: BackendPreference
    resolved: Literal["cpu", "gpu"]
    device: str
    fallback_reason: str | None


def resolve_backend(preference: BackendPreference, torch_module: object = torch) -> BackendStatus:
    """Resolve one public CPU/GPU status without probing any other device state."""
    available = bool(torch_module.cuda.is_available())  # type: ignore[attr-defined]
    if preference == "gpu-strict" and not available:
        raise BackendUnavailable("CUDA was requested strictly but is unavailable")

    use_gpu = available and preference in {"auto", "gpu", "gpu-strict"}
    return BackendStatus(
        requested=preference,
        resolved="gpu" if use_gpu else "cpu",
        device="cuda:0" if use_gpu else "cpu",
        fallback_reason=None if use_gpu or preference == "cpu" else "cuda-unavailable",
    )
