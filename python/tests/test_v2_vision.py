from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fly_crossy.v2.substrate import MaleCNSV2Substrate
from fly_crossy.v2.vision_frontend import FrozenVisualFrontEnd
from fly_crossy.v2.vision_geometry import CLAMPED_TYPES, VisualGeometry


ROOT = Path(__file__).resolve().parents[2]
GEOMETRY = ROOT / "artifacts" / "malecns-crossy-v2" / "visual-geometry-160x120.npz"
SUBSTRATE = ROOT / "artifacts" / "malecns-crossy-v2" / "substrate-w5-h4.npz"


@pytest.mark.skipif(not GEOMETRY.exists(), reason="build V2 visual geometry first")
def test_real_visual_geometry_matches_frozen_substrate_exactly() -> None:
    geometry = VisualGeometry.load(GEOMETRY, strict=True)
    substrate = MaleCNSV2Substrate.load(SUBSTRATE, strict=True)

    assert geometry.n_columns == 1771
    assert geometry.n_cells == 30906
    assert int((geometry.cell_col < 0).sum()) == 5
    assert np.array_equal(
        geometry.cell_body_id,
        substrate.body_ids[substrate.is_clamped],
    )


@pytest.mark.skipif(not GEOMETRY.exists(), reason="build V2 visual geometry first")
def test_frontend_is_fixed_and_emits_ten_substeps() -> None:
    geometry = VisualGeometry.load(GEOMETRY, strict=True)
    frontend = FrozenVisualFrontEnd(
        geometry,
        decision_ms=200,
        internal_steps=10,
        device="cpu",
    )

    assert frontend.trainable_parameters == 0
    state = frontend.init_state(1)
    frame = np.full((1, 120, 160, 3), 128, dtype=np.uint8)
    rates, _ = frontend.process(frame, state)

    assert rates.shape == (10, 1, 30906)
    assert torch.isfinite(rates).all()
    assert float(rates.min()) >= 0.0


@pytest.mark.skipif(not GEOMETRY.exists(), reason="build V2 visual geometry first")
def test_uniform_static_gray_does_not_create_motion_or_color_activity() -> None:
    geometry = VisualGeometry.load(GEOMETRY, strict=True)
    frontend = FrozenVisualFrontEnd(
        geometry,
        decision_ms=200,
        internal_steps=10,
        device="cpu",
    )

    state = frontend.init_state(1)
    frame = np.full((1, 120, 160, 3), 128, dtype=np.uint8)

    out = None
    for _ in range(5):
        out, state = frontend.process(frame, state)

    assert out is not None
    assert float(out[-1].max()) < 1e-6


@pytest.mark.skipif(not GEOMETRY.exists(), reason="build V2 visual geometry first")
def test_visual_type_vocabulary_is_complete() -> None:
    geometry = VisualGeometry.load(GEOMETRY, strict=True)
    present = {
        CLAMPED_TYPES[int(i)]
        for i in np.unique(geometry.cell_type_id)
    }
    assert present == set(CLAMPED_TYPES)
