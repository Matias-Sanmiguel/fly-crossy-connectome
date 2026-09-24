from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


CLAMPED_TYPES = (
    "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
    "Mi1", "Mi4", "Mi9",
    "Tm1", "Tm2", "Tm4", "Tm9", "Tm20",
    "Tm5a", "Tm5b", "Tm5c", "TmY5a",
)

HEX_TYPES = (
    "L1", "L2", "L3", "L5", "C2", "C3", "T1",
    "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9", "Tm20",
)

SIDES = ("L", "R")
NBR_NAMES = ("front", "back", "up_front", "up_back", "down_front", "down_back")

EXPECTED_COLUMNS = 1_771
EXPECTED_LEFT_COLUMNS = 879
EXPECTED_RIGHT_COLUMNS = 892
EXPECTED_CELLS = 30_906

SPACING_DEG = 5.7
FWHM_DEG = 5.7
HFOV_DEG = 108.0
OVERLAP_DEG = 10.0
FRAME_W = 160
FRAME_H = 120
RGB_DOWNSAMPLE = 2
TRUNC_SIGMA = 3.0

E1 = np.array([1.0, 0.0], dtype=np.float64)
E2 = np.array([0.5, np.sqrt(3.0) / 2.0], dtype=np.float64)
LATTICE_STEPS = np.array(
    [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]],
    dtype=np.int64,
)


def hex_to_raw(hexes: np.ndarray) -> np.ndarray:
    h = np.asarray(hexes, dtype=np.float64)
    return h[:, :1] * E1 + h[:, 1:2] * E2


def pixel_directions(hfov_deg: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    focal = (width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
    px = (np.arange(width) + 0.5 - width / 2.0) / focal
    py = -(np.arange(height) + 0.5 - height / 2.0) / focal
    x, y = np.meshgrid(px, py, indexing="xy")
    x = x.ravel()
    y = y.ravel()
    radius2 = 1.0 + x * x + y * y
    directions = np.stack([x, y, np.ones_like(x)], axis=1)
    directions /= np.sqrt(radius2)[:, None]
    solid_angle = 1.0 / (focal * focal * radius2 ** 1.5)
    return directions, solid_angle


def build_sampler(
    col_az: np.ndarray,
    col_el: np.ndarray,
    col_side: np.ndarray,
    *,
    width: int,
    height: int,
    hfov_deg: float = HFOV_DEG,
    fwhm_deg: float = FWHM_DEG,
    overlap_deg: float = OVERLAP_DEG,
    trunc_sigma: float = TRUNC_SIGMA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions, solid_angle = pixel_directions(hfov_deg, width, height)

    sigma = np.radians(fwhm_deg) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    analytic_mass = 2.0 * np.pi * sigma * sigma * (
        1.0 - np.exp(-(trunc_sigma * trunc_sigma) / 2.0)
    )

    az = np.radians(col_az.astype(np.float64))
    el = np.radians(col_el.astype(np.float64))
    column_direction = np.stack(
        [
            np.cos(el) * np.sin(az),
            np.sin(el),
            np.cos(el) * np.cos(az),
        ],
        axis=1,
    )

    horizontal_margin = np.radians(hfov_deg / 2.0) + trunc_sigma * sigma
    vfov = 2.0 * np.arctan(
        (height / 2.0)
        / ((width / 2.0) / np.tan(np.radians(hfov_deg / 2.0)))
    )
    vertical_margin = vfov / 2.0 + trunc_sigma * sigma

    candidate = (np.abs(az) <= horizontal_margin) & (np.abs(el) <= vertical_margin)
    split = np.where(
        col_side == 0,
        np.clip((overlap_deg / 2.0 - col_az) / overlap_deg, 0.0, 1.0),
        np.clip((overlap_deg / 2.0 + col_az) / overlap_deg, 0.0, 1.0),
    ).astype(np.float64)

    indptr = np.zeros(len(col_az) + 1, dtype=np.int64)
    indices_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    coverage = np.zeros(len(col_az), dtype=np.float64)

    for i in range(len(col_az)):
        if not candidate[i] or split[i] <= 0.0:
            indptr[i + 1] = indptr[i]
            continue

        angle = np.arccos(
            np.clip(directions @ column_direction[i], -1.0, 1.0)
        )
        selected = np.flatnonzero(angle < trunc_sigma * sigma)
        if selected.size == 0:
            indptr[i + 1] = indptr[i]
            continue

        raw = (
            np.exp(-0.5 * (angle[selected] / sigma) ** 2)
            * solid_angle[selected]
            * split[i]
        )
        normalized = raw / analytic_mass
        row_coverage = min(1.0, float(normalized.sum()))
        if normalized.sum() > 1.0:
            normalized /= normalized.sum()

        indices_parts.append(selected.astype(np.int32, copy=False))
        weight_parts.append(normalized.astype(np.float32, copy=False))
        coverage[i] = row_coverage
        indptr[i + 1] = indptr[i] + len(selected)

    indices = (
        np.concatenate(indices_parts)
        if indices_parts
        else np.zeros(0, dtype=np.int32)
    )
    weights = (
        np.concatenate(weight_parts)
        if weight_parts
        else np.zeros(0, dtype=np.float32)
    )
    return indptr, indices, weights, coverage.astype(np.float32)


@dataclass(frozen=True, slots=True)
class VisualGeometry:
    col_side: np.ndarray
    col_hex: np.ndarray
    col_az: np.ndarray
    col_el: np.ndarray
    nbr: np.ndarray
    coverage: np.ndarray
    driven: np.ndarray
    pix_indptr: np.ndarray
    pix_indices: np.ndarray
    pix_weights: np.ndarray
    rgb_indptr: np.ndarray
    rgb_indices: np.ndarray
    rgb_weights: np.ndarray
    rgb_coverage: np.ndarray
    side_gain: np.ndarray
    cell_body_id: np.ndarray
    cell_type_id: np.ndarray
    cell_side: np.ndarray
    cell_col: np.ndarray
    meta: dict

    @property
    def n_columns(self) -> int:
        return int(self.col_side.size)

    @property
    def n_cells(self) -> int:
        return int(self.cell_body_id.size)

    @property
    def frame_hw(self) -> tuple[int, int]:
        return int(self.meta["frame_h"]), int(self.meta["frame_w"])

    @property
    def rgb_hw(self) -> tuple[int, int]:
        return int(self.meta["rgb_h"]), int(self.meta["rgb_w"])

    def validate(self, *, strict: bool = True) -> None:
        c = self.n_columns
        m = self.n_cells

        if self.col_hex.shape != (c, 2):
            raise ValueError("col_hex has an incompatible shape.")
        if self.col_az.shape != (c,) or self.col_el.shape != (c,):
            raise ValueError("column angle arrays have incompatible shapes.")
        if self.nbr.shape != (c, 6):
            raise ValueError("neighbor table must have six slots per column.")
        if self.coverage.shape != (c,) or self.driven.shape != (c,):
            raise ValueError("coverage arrays have incompatible shapes.")
        if self.rgb_coverage.shape != (c,):
            raise ValueError("rgb_coverage has an incompatible shape.")
        if self.side_gain.shape != (2,):
            raise ValueError("side_gain must have exactly two values.")

        for name, array in {
            "cell_type_id": self.cell_type_id,
            "cell_side": self.cell_side,
            "cell_col": self.cell_col,
        }.items():
            if array.shape != (m,):
                raise ValueError(f"{name} has an incompatible shape.")

        if np.any(self.cell_body_id[1:] <= self.cell_body_id[:-1]):
            raise ValueError("visual cell body IDs must be strictly increasing.")

        if np.any(self.cell_type_id < 0) or np.any(self.cell_type_id >= len(CLAMPED_TYPES)):
            raise ValueError("cell_type_id contains an invalid type index.")

        assigned = self.cell_col >= 0
        if np.any(self.cell_col[assigned] >= c):
            raise ValueError("cell_col contains an invalid column index.")

        if np.any(self.nbr < -1) or np.any(self.nbr >= c):
            raise ValueError("neighbor table contains an invalid column index.")

        if self.pix_indptr.shape != (c + 1,) or self.pix_indptr[0] != 0:
            raise ValueError("invalid full-resolution sampler indptr.")
        if self.pix_indptr[-1] != len(self.pix_indices) or len(self.pix_indices) != len(self.pix_weights):
            raise ValueError("full-resolution sampler arrays disagree.")
        if self.rgb_indptr.shape != (c + 1,) or self.rgb_indptr[0] != 0:
            raise ValueError("invalid RGB sampler indptr.")
        if self.rgb_indptr[-1] != len(self.rgb_indices) or len(self.rgb_indices) != len(self.rgb_weights):
            raise ValueError("RGB sampler arrays disagree.")

        if np.any(self.coverage < 0) or np.any(self.coverage > 1.00001):
            raise ValueError("coverage must lie in [0,1].")
        if np.any(self.rgb_coverage < 0) or np.any(self.rgb_coverage > 1.00001):
            raise ValueError("rgb_coverage must lie in [0,1].")

        if strict:
            if c != EXPECTED_COLUMNS:
                raise ValueError(f"Expected {EXPECTED_COLUMNS} columns, got {c}.")
            if int((self.col_side == 0).sum()) != EXPECTED_LEFT_COLUMNS:
                raise ValueError("Unexpected number of left-eye columns.")
            if int((self.col_side == 1).sum()) != EXPECTED_RIGHT_COLUMNS:
                raise ValueError("Unexpected number of right-eye columns.")
            if m != EXPECTED_CELLS:
                raise ValueError(f"Expected {EXPECTED_CELLS} visual cells, got {m}.")
            if int((self.cell_col < 0).sum()) > 5:
                raise ValueError("Unexpected number of unassigned visual cells.")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            col_side=self.col_side,
            col_hex=self.col_hex,
            col_az=self.col_az,
            col_el=self.col_el,
            nbr=self.nbr,
            coverage=self.coverage,
            driven=self.driven,
            pix_indptr=self.pix_indptr,
            pix_indices=self.pix_indices,
            pix_weights=self.pix_weights,
            rgb_indptr=self.rgb_indptr,
            rgb_indices=self.rgb_indices,
            rgb_weights=self.rgb_weights,
            rgb_coverage=self.rgb_coverage,
            side_gain=self.side_gain,
            cell_body_id=self.cell_body_id,
            cell_type_id=self.cell_type_id,
            cell_side=self.cell_side,
            cell_col=self.cell_col,
            meta=json.dumps(self.meta, separators=(",", ":")),
        )

    @classmethod
    def load(cls, path: str | Path, *, strict: bool = True) -> "VisualGeometry":
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            geometry = cls(
                col_side=z["col_side"].astype(np.int8, copy=True),
                col_hex=z["col_hex"].astype(np.int16, copy=True),
                col_az=z["col_az"].astype(np.float32, copy=True),
                col_el=z["col_el"].astype(np.float32, copy=True),
                nbr=z["nbr"].astype(np.int32, copy=True),
                coverage=z["coverage"].astype(np.float32, copy=True),
                driven=z["driven"].astype(np.bool_, copy=True),
                pix_indptr=z["pix_indptr"].astype(np.int64, copy=True),
                pix_indices=z["pix_indices"].astype(np.int32, copy=True),
                pix_weights=z["pix_weights"].astype(np.float32, copy=True),
                rgb_indptr=z["rgb_indptr"].astype(np.int64, copy=True),
                rgb_indices=z["rgb_indices"].astype(np.int32, copy=True),
                rgb_weights=z["rgb_weights"].astype(np.float32, copy=True),
                rgb_coverage=z["rgb_coverage"].astype(np.float32, copy=True),
                side_gain=z["side_gain"].astype(np.float32, copy=True),
                cell_body_id=z["cell_body_id"].astype(np.int64, copy=True),
                cell_type_id=z["cell_type_id"].astype(np.int16, copy=True),
                cell_side=z["cell_side"].astype(np.int8, copy=True),
                cell_col=z["cell_col"].astype(np.int32, copy=True),
                meta=meta,
            )
        geometry.validate(strict=strict)
        return geometry
