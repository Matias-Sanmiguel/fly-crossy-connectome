from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .substrate import MaleCNSV2Substrate


@dataclass(frozen=True, slots=True)
class FrozenMaleCNSState:
    """Recurrent state for one or more parallel Crossy episodes."""

    h: Tensor
    x_clamped: Tensor
    previous_readout: Tensor


class FrozenMaleCNSCore(nn.Module):
    """Frozen sparse MaleCNS rate RNN for Crossy V2.

    The anatomical graph and every synaptic relative magnitude remain fixed.
    A single global scalar ``w0`` calibrates recurrent gain; it is a buffer,
    not a trainable parameter.

    Dynamics per internal substep:
        h <- h + dt/tau * ( -h + W r )
        r  = clamp(softplus(h), r_max)

    Visual cells are clamped externally every substep and have no incoming
    recurrent edges in the frozen substrate.
    """

    def __init__(
        self,
        substrate: MaleCNSV2Substrate,
        *,
        decision_ms: float = 200.0,
        internal_steps: int = 10,
        tau_ms: float = 20.0,
        r_max: float = 5.0,
        w0: float = 1.0,
        inhibitory_centring: bool = True,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if internal_steps <= 0:
            raise ValueError("internal_steps must be positive.")
        if decision_ms <= 0 or tau_ms <= 0:
            raise ValueError("decision_ms and tau_ms must be positive.")
        if r_max <= 0:
            raise ValueError("r_max must be positive.")
        if w0 <= 0 or not math.isfinite(w0):
            raise ValueError("w0 must be positive and finite.")

        substrate.validate(strict=False)
        self.substrate = substrate
        self.decision_ms = float(decision_ms)
        self.internal_steps = int(internal_steps)
        self.tau_ms = float(tau_ms)
        self.r_max = float(r_max)
        self.dtype = dtype

        dt_ms = self.decision_ms / self.internal_steps
        # Explicit Euler becomes unstable if dt/tau is allowed to exceed 1 in
        # this simple rate model, so tau below dt behaves as tau=dt.
        self.integration_coefficient = min(1.0, dt_ms / self.tau_ms)

        dev = torch.device(device)

        clamped = np.flatnonzero(substrate.is_clamped)
        dynamic = np.flatnonzero(substrate.is_dynamic)
        readout = np.flatnonzero(substrate.decision_readout_mask)

        self.register_buffer(
            "clamped_idx",
            torch.as_tensor(clamped, dtype=torch.long, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "dynamic_idx",
            torch.as_tensor(dynamic, dtype=torch.long, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "readout_idx",
            torch.as_tensor(readout, dtype=torch.long, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "w0",
            torch.tensor(float(w0), dtype=dtype, device=dev),
            persistent=True,
        )

        pre = substrate.indices_pre.astype(np.int64, copy=False)
        sign = substrate.sign.astype(np.float64, copy=False)
        contacts = substrate.weight.astype(np.float64, copy=False)

        magnitude = np.log1p(contacts)
        edge_sign = sign[pre]
        inhibitory = edge_sign < 0

        if inhibitory_centring and inhibitory.any() and (~inhibitory).any():
            excitatory_sum = float(magnitude[~inhibitory].sum())
            inhibitory_sum = float(magnitude[inhibitory].sum())
            inhibitory_factor = excitatory_sum / inhibitory_sum
        else:
            inhibitory_factor = 1.0

        centred_magnitude = magnitude.copy()
        centred_magnitude[inhibitory] *= inhibitory_factor
        signed_values = centred_magnitude * edge_sign

        self.inhibitory_factor = float(inhibitory_factor)

        crow = torch.tensor(
            substrate.indptr_post,
            dtype=torch.int64,
            device=dev,
        )
        col = torch.tensor(
            substrate.indices_pre,
            dtype=torch.int64,
            device=dev,
        )
        values = torch.as_tensor(
            signed_values,
            dtype=dtype,
            device=dev,
        )

        adjacency = torch.sparse_csr_tensor(
            crow,
            col,
            values,
            size=(substrate.node_count, substrate.node_count),
            dtype=dtype,
            device=dev,
        )
        self.register_buffer("base_adjacency", adjacency, persistent=False)

        coefficient = torch.zeros(
            substrate.node_count, 1, dtype=dtype, device=dev
        )
        coefficient[self.dynamic_idx] = self.integration_coefficient
        self.register_buffer(
            "integration_coef",
            coefficient,
            persistent=False,
        )

    @property
    def node_count(self) -> int:
        return self.substrate.node_count

    @property
    def edge_count(self) -> int:
        return self.substrate.edge_count

    @property
    def n_clamped(self) -> int:
        return int(self.clamped_idx.numel())

    @property
    def n_dynamic(self) -> int:
        return int(self.dynamic_idx.numel())

    @property
    def n_readout(self) -> int:
        return int(self.readout_idx.numel())

    @property
    def device(self) -> torch.device:
        return self.w0.device

    def set_w0(self, value: float) -> None:
        if value <= 0 or not math.isfinite(value):
            raise ValueError("w0 must be positive and finite.")
        with torch.no_grad():
            self.w0.fill_(float(value))

    def rate(self, h: Tensor) -> Tensor:
        return torch.clamp(F.softplus(h), max=self.r_max)

    def _base_mm(self, rates: Tensor) -> Tensor:
        return torch.sparse.mm(self.base_adjacency, rates)

    def apply_w(self, rates: Tensor) -> Tensor:
        return self.w0 * self._base_mm(rates)

    def init_state(self, batch: int = 1) -> FrozenMaleCNSState:
        if batch <= 0:
            raise ValueError("batch must be positive.")
        h = torch.zeros(
            self.node_count,
            batch,
            dtype=self.dtype,
            device=self.device,
        )
        x_clamped = torch.zeros(
            self.n_clamped,
            batch,
            dtype=self.dtype,
            device=self.device,
        )
        readout = self.rate(h).index_select(0, self.readout_idx)
        return FrozenMaleCNSState(
            h=h,
            x_clamped=x_clamped,
            previous_readout=readout,
        )

    def _normalize_clamped(self, value: Tensor, batch: int) -> Tensor:
        xc = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if xc.ndim == 2:
            xc = xc.unsqueeze(0).expand(self.internal_steps, -1, -1)
        expected = (self.internal_steps, batch, self.n_clamped)
        if tuple(xc.shape) != expected:
            raise ValueError(
                f"clamped_rates shape {tuple(xc.shape)} != {expected}"
            )
        if not torch.isfinite(xc).all():
            raise ValueError("clamped_rates contains non-finite values.")
        return xc

    @torch.no_grad()
    def step(
        self,
        state: FrozenMaleCNSState,
        clamped_rates: Tensor,
    ) -> FrozenMaleCNSState:
        if state.h.shape[0] != self.node_count:
            raise ValueError("State node dimension does not match this core.")
        batch = state.h.shape[1]
        xc = self._normalize_clamped(clamped_rates, batch)

        h = state.h
        previous_readout = self.rate(h).index_select(0, self.readout_idx)

        for k in range(self.internal_steps):
            rates = self.rate(h).index_copy(
                0,
                self.clamped_idx,
                xc[k].T,
            )
            drive = -h + self.apply_w(rates)
            h = h + self.integration_coef * drive

        return FrozenMaleCNSState(
            h=h,
            x_clamped=xc[-1].T,
            previous_readout=previous_readout,
        )

    def rates(self, state: FrozenMaleCNSState) -> Tensor:
        return self.rate(state.h).index_copy(
            0,
            self.clamped_idx,
            state.x_clamped,
        )

    def decision_features(self, state: FrozenMaleCNSState) -> Tensor:
        """Return [B, 2 * 10,511]: readout rates plus first differences."""
        current = self.rate(state.h).index_select(0, self.readout_idx)
        return torch.cat(
            (current, current - state.previous_readout),
            dim=0,
        ).T

    @torch.no_grad()
    def estimate_rest_gain(
        self,
        *,
        iterations: int = 80,
        seed: int = 0,
    ) -> float:
        """Estimate spectral radius of the linearised recurrent core at rest.

        At h=0, d softplus / dh = 0.5. Visual cells are externally clamped,
        so perturbations on those rows are forced to zero.
        """
        if iterations < 10:
            raise ValueError("iterations must be at least 10.")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        v = torch.randn(
            self.node_count,
            1,
            generator=generator,
            dtype=self.dtype,
            device=self.device,
        )
        v.index_fill_(0, self.clamped_idx, 0.0)
        v = v / v.norm().clamp_min(1e-12)

        start_average = iterations // 2
        log_growth = 0.0
        averaged = 0

        # Estimate the w0=1 operator so calibration remains a single scalar.
        for i in range(iterations):
            y = 0.5 * self._base_mm(v)
            y.index_fill_(0, self.clamped_idx, 0.0)
            norm = float(y.norm())
            if norm == 0.0:
                return 0.0
            if not math.isfinite(norm):
                return float("inf")
            if i >= start_average:
                log_growth += math.log(norm)
                averaged += 1
            v = y / norm

        return math.exp(log_growth / max(averaged, 1))

    @torch.no_grad()
    def match_rest_gain(
        self,
        target: float = 0.95,
        *,
        iterations: int = 80,
        seed: int = 0,
    ) -> dict[str, float]:
        if target <= 0 or target >= 1.5:
            raise ValueError("target rest gain must be in (0, 1.5).")

        base_gain = self.estimate_rest_gain(
            iterations=iterations,
            seed=seed,
        )
        if not (base_gain > 0 and math.isfinite(base_gain)):
            raise RuntimeError(
                f"Cannot calibrate recurrent gain from base gain {base_gain}."
            )

        calibrated_w0 = target / base_gain
        self.set_w0(calibrated_w0)
        return {
            "target": float(target),
            "baseGainAtW0One": float(base_gain),
            "w0": float(calibrated_w0),
            "estimatedGain": float(base_gain * calibrated_w0),
        }

    def summary(self) -> dict[str, object]:
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "clampedVisual": self.n_clamped,
            "dynamic": self.n_dynamic,
            "decisionReadout": self.n_readout,
            "decisionMs": self.decision_ms,
            "internalSteps": self.internal_steps,
            "dtMs": self.decision_ms / self.internal_steps,
            "tauMs": self.tau_ms,
            "integrationCoefficient": self.integration_coefficient,
            "rMax": self.r_max,
            "w0": float(self.w0),
            "inhibitoryCentringFactor": self.inhibitory_factor,
            "trainableParameters": sum(p.numel() for p in self.parameters()),
        }
