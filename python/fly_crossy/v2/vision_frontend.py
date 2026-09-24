from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .vision_geometry import CLAMPED_TYPES, VisualGeometry


SOURCES = (
    "t4a", "t4b", "t4c", "t4d",
    "t5a", "t5b", "t5c", "t5d",
    "on", "off",
    "rg_pos", "rg_neg",
    "by_pos", "by_neg",
)

# Fixed visual-interface constants. These are not trained on Crossy.
TYPE_PARAMS: dict[str, tuple[str, float]] = {
    "T4a": ("t4a", 30.0),
    "T4b": ("t4b", 30.0),
    "T4c": ("t4c", 30.0),
    "T4d": ("t4d", 30.0),
    "T5a": ("t5a", 30.0),
    "T5b": ("t5b", 30.0),
    "T5c": ("t5c", 30.0),
    "T5d": ("t5d", 30.0),
    "Mi1": ("on", 30.0),
    "Mi4": ("on", 150.0),
    "Mi9": ("off", 150.0),
    "Tm1": ("off", 30.0),
    "Tm2": ("off", 20.0),
    "Tm4": ("off", 30.0),
    "Tm9": ("off", 150.0),
    "Tm20": ("rg_neg", 60.0),
    "Tm5a": ("by_pos", 60.0),
    "Tm5b": ("by_neg", 60.0),
    "Tm5c": ("by_pos", 150.0),
    "TmY5a": ("rg_pos", 60.0),
}

EMD_GAIN = 4.0


@dataclass(slots=True)
class VisualState:
    previous_columns: Tensor
    photoreceptor: Tensor
    running_mean: Tensor
    delayed_on: Tensor
    delayed_off: Tensor
    bank: Tensor
    live: Tensor


class FrozenVisualFrontEnd:
    """Fixed retinal/optic-lobe input transform for MaleCNS Crossy V2.

    Input: egocentric 160x120 uint8 RGB frame.
    Output: [K, B, 30_906] clamped rates ordered exactly like the frozen substrate.

    The transform has zero trainable parameters. It implements:
    - measured optic-lobe column geometry;
    - Gaussian angular acceptance per column;
    - luminance adaptation and ON/OFF contrast;
    - Hassenstein-Reichardt opponent motion channels;
    - fixed per-type temporal filtering;
    - fixed RGB opponent proxy channels for the five chromatic input types.
    """

    def __init__(
        self,
        geometry: VisualGeometry,
        *,
        decision_ms: float = 200.0,
        internal_steps: int = 10,
        device: str | torch.device = "cpu",
        eps: float = 0.02,
        tau_photo_ms: float = 15.0,
        tau_mean_ms: float = 200.0,
        tau_delay_ms: float = 50.0,
    ) -> None:
        if internal_steps <= 0:
            raise ValueError("internal_steps must be positive.")
        if decision_ms <= 0:
            raise ValueError("decision_ms must be positive.")

        geometry.validate(strict=True)
        self.geometry = geometry
        self.device = torch.device(device)
        self.decision_ms = float(decision_ms)
        self.internal_steps = int(internal_steps)
        self.dt_ms = self.decision_ms / self.internal_steps
        self.eps = float(eps)

        c = geometry.n_columns
        active = geometry.coverage > 0
        neighbor_rows = geometry.nbr[active]
        valid_neighbor = neighbor_rows[neighbor_rows >= 0]
        active[valid_neighbor] = True

        self.active_columns = np.flatnonzero(active)
        active_count = len(self.active_columns)

        # +2 virtual columns, one per eye, used for acceptance mass outside the camera frustum.
        self.reduced_columns = active_count + 2

        col_map = np.full(c, -1, dtype=np.int64)
        col_map[self.active_columns] = np.arange(active_count)
        col_map[~active] = active_count + geometry.col_side[~active].astype(np.int64)
        self.col_map = col_map

        reduced_side = np.concatenate(
            [geometry.col_side[self.active_columns], np.array([0, 1], dtype=np.int8)]
        )
        reduced_driven = np.concatenate(
            [geometry.driven[self.active_columns], np.array([False, False])]
        )

        neighbor = np.full((self.reduced_columns, 6), -1, dtype=np.int64)
        source_nbr = geometry.nbr[self.active_columns]
        neighbor[:active_count] = np.where(
            source_nbr >= 0,
            col_map[np.maximum(source_nbr, 0)],
            -1,
        )
        has_neighbor = neighbor >= 0
        safe_neighbor = np.where(
            has_neighbor,
            neighbor,
            np.arange(self.reduced_columns)[:, None],
        )

        up_weight = has_neighbor[:, 2:4].astype(np.float32)
        down_weight = has_neighbor[:, 4:6].astype(np.float32)
        up_weight /= np.maximum(up_weight.sum(axis=1, keepdims=True), 1.0)
        down_weight /= np.maximum(down_weight.sum(axis=1, keepdims=True), 1.0)

        self.side = torch.as_tensor(
            reduced_side,
            dtype=torch.long,
            device=self.device,
        )
        self.neighbor = torch.as_tensor(
            safe_neighbor,
            dtype=torch.long,
            device=self.device,
        )
        self.up_weight = torch.as_tensor(
            up_weight,
            dtype=torch.float32,
            device=self.device,
        )
        self.down_weight = torch.as_tensor(
            down_weight,
            dtype=torch.float32,
            device=self.device,
        )

        eye_mean = np.zeros((2, self.reduced_columns), dtype=np.float32)
        for eye in (0, 1):
            mask = reduced_driven & (reduced_side == eye)
            eye_mean[eye, mask] = 1.0 / max(int(mask.sum()), 1)
        self.eye_mean = torch.as_tensor(
            eye_mean,
            dtype=torch.float32,
            device=self.device,
        )

        self.full_sampler, self.full_coverage = self._build_runtime_sampler(
            geometry.pix_indptr,
            geometry.pix_indices,
            geometry.pix_weights,
            geometry.coverage,
            geometry.frame_hw[0] * geometry.frame_hw[1],
        )
        self.rgb_sampler, self.rgb_coverage = self._build_runtime_sampler(
            geometry.rgb_indptr,
            geometry.rgb_indices,
            geometry.rgb_weights,
            geometry.rgb_coverage,
            geometry.rgb_hw[0] * geometry.rgb_hw[1],
        )

        exp_decay = lambda tau: float(np.exp(-self.dt_ms / tau))
        self.photo_decay = exp_decay(tau_photo_ms)
        self.mean_decay = exp_decay(tau_mean_ms)
        self.delay_decay = exp_decay(tau_delay_ms)

        self.type_decay = torch.as_tensor(
            [exp_decay(TYPE_PARAMS[t][1]) for t in CLAMPED_TYPES],
            dtype=torch.float32,
            device=self.device,
        )[None, :, None]

        self.source_index = torch.as_tensor(
            [SOURCES.index(TYPE_PARAMS[t][0]) for t in CLAMPED_TYPES],
            dtype=torch.long,
            device=self.device,
        )

        type_gain = np.asarray(
            [
                EMD_GAIN if TYPE_PARAMS[t][0].startswith(("t4", "t5")) else 1.0
                for t in CLAMPED_TYPES
            ],
            dtype=np.float32,
        )
        gain = type_gain[:, None] * geometry.side_gain[reduced_side][None, :]
        self.gain = torch.as_tensor(
            gain,
            dtype=torch.float32,
            device=self.device,
        )[None]

        # Gather each real visual cell from [type, reduced_column].
        cell_reduced_col = np.where(
            geometry.cell_col >= 0,
            col_map[np.maximum(geometry.cell_col, 0)],
            self.reduced_columns,  # final bank column is hard zero for the five unassigned cells
        )
        self.unit_flat = torch.as_tensor(
            geometry.cell_type_id.astype(np.int64) * (self.reduced_columns + 1)
            + cell_reduced_col,
            dtype=torch.long,
            device=self.device,
        )
        self.n_clamped = geometry.n_cells

    def _build_runtime_sampler(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        coverage: np.ndarray,
        pixels: int,
    ) -> tuple[Tensor, Tensor]:
        rows = [
            np.arange(indptr[i], indptr[i + 1], dtype=np.int64)
            for i in self.active_columns
        ]
        selected = (
            np.concatenate(rows)
            if rows
            else np.zeros(0, dtype=np.int64)
        )
        lengths = np.asarray([len(r) for r in rows], dtype=np.int64)
        crow = np.r_[0, np.cumsum(lengths)].astype(np.int64)

        sampler = torch.sparse_csr_tensor(
            torch.as_tensor(crow, dtype=torch.int64, device=self.device),
            torch.as_tensor(indices[selected], dtype=torch.int64, device=self.device),
            torch.as_tensor(weights[selected], dtype=torch.float32, device=self.device),
            size=(len(self.active_columns), pixels),
            device=self.device,
        )

        reduced_coverage = np.concatenate(
            [coverage[self.active_columns], np.array([0.0, 0.0], dtype=np.float32)]
        )
        return sampler, torch.as_tensor(
            reduced_coverage,
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def trainable_parameters(self) -> int:
        return 0

    def init_state(self, batch: int = 1) -> VisualState:
        if batch <= 0:
            raise ValueError("batch must be positive.")
        z = lambda *shape: torch.zeros(
            *shape,
            dtype=torch.float32,
            device=self.device,
        )
        return VisualState(
            previous_columns=z(batch, self.reduced_columns, 4),
            photoreceptor=z(batch, self.reduced_columns, 4),
            running_mean=z(batch, self.reduced_columns),
            delayed_on=z(batch, self.reduced_columns),
            delayed_off=z(batch, self.reduced_columns),
            bank=z(batch, len(CLAMPED_TYPES), self.reduced_columns + 1),
            live=torch.zeros(batch, dtype=torch.bool, device=self.device),
        )

    def reset(self, state: VisualState, mask: np.ndarray | Tensor) -> None:
        state.live[torch.as_tensor(mask, dtype=torch.bool, device=self.device)] = False

    def _fill_missing_acceptance(self, coverage: Tensor, sampled: Tensor) -> Tensor:
        batch, _, channels = sampled.shape
        columns = torch.cat(
            [
                sampled,
                torch.zeros(
                    batch,
                    2,
                    channels,
                    dtype=sampled.dtype,
                    device=self.device,
                ),
            ],
            dim=1,
        )
        normalized = columns / coverage.clamp_min(1e-6)[None, :, None]
        mean_by_eye = torch.einsum(
            "sr,brk->bsk",
            self.eye_mean,
            normalized,
        )
        return columns + (
            (1.0 - coverage)[None, :, None]
            * mean_by_eye.index_select(1, self.side)
        )

    @torch.no_grad()
    def sample_frame(self, rgb: Tensor | np.ndarray) -> Tensor:
        rgb_t = torch.as_tensor(rgb, device=self.device)
        if rgb_t.ndim != 4 or rgb_t.shape[-1] != 3:
            raise ValueError("Expected RGB frames with shape [B,H,W,3].")

        h, w = self.geometry.frame_hw
        if tuple(rgb_t.shape[1:3]) != (h, w):
            raise ValueError(
                f"Expected {h}x{w} RGB frames, got {tuple(rgb_t.shape[1:3])}."
            )

        rgb_float = rgb_t.to(torch.float32)
        # BT.709 luminance for the full-resolution motion/form path.
        luminance = (
            0.2126 * rgb_float[..., 0]
            + 0.7152 * rgb_float[..., 1]
            + 0.0722 * rgb_float[..., 2]
        ) / 255.0

        batch = rgb_t.shape[0]
        lum_flat = luminance.reshape(batch, -1).T
        lum_sampled = torch.sparse.mm(self.full_sampler, lum_flat).T[:, :, None]
        lum_columns = self._fill_missing_acceptance(
            self.full_coverage,
            lum_sampled,
        )

        small_h, small_w = self.geometry.rgb_hw
        rgb_chw = rgb_float.permute(0, 3, 1, 2)
        rgb_small = torch.nn.functional.adaptive_avg_pool2d(
            rgb_chw,
            (small_h, small_w),
        )
        rgb_flat = rgb_small.reshape(batch * 3, -1).T / 255.0
        rgb_sampled = torch.sparse.mm(self.rgb_sampler, rgb_flat)
        rgb_sampled = (
            rgb_sampled.T
            .reshape(batch, 3, -1)
            .permute(0, 2, 1)
        )
        rgb_columns = self._fill_missing_acceptance(
            self.rgb_coverage,
            rgb_sampled,
        )

        return torch.cat([lum_columns, rgb_columns], dim=2)

    def _initialize_new_rows(self, state: VisualState, current: Tensor) -> None:
        dead = ~state.live
        if not bool(dead.any()):
            return
        state.previous_columns[dead] = current[dead]
        state.photoreceptor[dead] = current[dead]
        state.running_mean[dead] = current[dead, :, 0]
        state.delayed_on[dead] = 0.0
        state.delayed_off[dead] = 0.0
        state.bank[dead] = 0.0
        state.live[dead] = True

    def _emd(self, direct: Tensor, delayed: Tensor) -> Tensor:
        get = lambda x, slot: x.index_select(1, self.neighbor[:, slot])

        direct_front = get(direct, 0)
        direct_back = get(direct, 1)
        delayed_front = get(delayed, 0)
        delayed_back = get(delayed, 1)

        direct_up = (
            self.up_weight[:, 0] * get(direct, 2)
            + self.up_weight[:, 1] * get(direct, 3)
        )
        direct_down = (
            self.down_weight[:, 0] * get(direct, 4)
            + self.down_weight[:, 1] * get(direct, 5)
        )
        delayed_up = (
            self.up_weight[:, 0] * get(delayed, 2)
            + self.up_weight[:, 1] * get(delayed, 3)
        )
        delayed_down = (
            self.down_weight[:, 0] * get(delayed, 4)
            + self.down_weight[:, 1] * get(delayed, 5)
        )

        a = delayed_front * direct - direct_front * delayed
        b = delayed_back * direct - direct_back * delayed
        c = delayed_down * direct - direct_down * delayed
        d = delayed_up * direct - direct_up * delayed

        return torch.relu(torch.stack([a, b, c, d], dim=1))

    def _substep(self, x: Tensor, state: VisualState) -> None:
        state.photoreceptor.mul_(self.photo_decay).add_(
            (1.0 - self.photo_decay) * x
        )
        lum = state.photoreceptor[:, :, 0]

        contrast = torch.tanh(
            (lum - state.running_mean)
            / (state.running_mean + self.eps)
        )

        state.running_mean.mul_(self.mean_decay).add_(
            (1.0 - self.mean_decay) * lum
        )
        on = torch.relu(contrast)
        off = torch.relu(-contrast)

        state.delayed_on.mul_(self.delay_decay).add_(
            (1.0 - self.delay_decay) * on
        )
        state.delayed_off.mul_(self.delay_decay).add_(
            (1.0 - self.delay_decay) * off
        )

        red = state.photoreceptor[:, :, 1]
        green = state.photoreceptor[:, :, 2]
        blue = state.photoreceptor[:, :, 3]

        rg = (red - green) / (red + green + 2.0 * self.eps)
        yellow = 0.5 * (red + green)
        by = (blue - yellow) / (blue + yellow + 2.0 * self.eps)

        sources = torch.cat(
            [
                self._emd(on, state.delayed_on),
                self._emd(off, state.delayed_off),
                torch.stack(
                    [
                        on,
                        off,
                        torch.relu(rg),
                        torch.relu(-rg),
                        torch.relu(by),
                        torch.relu(-by),
                    ],
                    dim=1,
                ),
            ],
            dim=1,
        )

        drive = sources.index_select(1, self.source_index) * self.gain
        live_bank = state.bank[:, :, : self.reduced_columns]
        live_bank.mul_(self.type_decay).add_(
            (1.0 - self.type_decay) * drive
        )

    @torch.no_grad()
    def process(
        self,
        rgb: Tensor | np.ndarray,
        state: VisualState,
    ) -> tuple[Tensor, VisualState]:
        current = self.sample_frame(rgb)
        self._initialize_new_rows(state, current)
        batch = current.shape[0]

        output = torch.empty(
            self.internal_steps,
            batch,
            self.n_clamped,
            dtype=torch.float32,
            device=self.device,
        )

        for k in range(1, self.internal_steps + 1):
            alpha = k / self.internal_steps
            interpolated = state.previous_columns + alpha * (
                current - state.previous_columns
            )
            self._substep(interpolated, state)
            torch.index_select(
                state.bank.reshape(batch, -1),
                1,
                self.unit_flat,
                out=output[k - 1],
            )

        state.previous_columns.copy_(current)
        return output, state

    def summary(self) -> dict[str, object]:
        return {
            "frame": [
                self.geometry.meta["frame_w"],
                self.geometry.meta["frame_h"],
            ],
            "internalSteps": self.internal_steps,
            "decisionMs": self.decision_ms,
            "dtMs": self.dt_ms,
            "visualCells": self.n_clamped,
            "geometryColumns": self.geometry.n_columns,
            "reducedColumns": self.reduced_columns,
            "trainableParameters": 0,
            "unassignedCells": int((self.geometry.cell_col < 0).sum()),
            "chromaticInterface": (
                "fixed RGB opponent proxies for Tm20/Tm5a/Tm5b/Tm5c/TmY5a; "
                "not claimed as a literal Drosophila spectral model"
            ),
        }
