from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fly_crossy.v2 import FrozenMaleCNSCore, MaleCNSV2Substrate


def _tiny_substrate(tmp_path: Path, *, incoming_to_clamped: bool = False) -> Path:
    # Local nodes: 0 visual/clamped, 1..3 dynamic.
    # CSR rows are POST, columns are PRE.
    if incoming_to_clamped:
        indptr_post = np.array([0, 1, 3, 4, 5], dtype=np.int64)
        indices_pre = np.array([1, 0, 3, 1, 2], dtype=np.int32)
        weight = np.array([2, 5, 2, 3, 4], dtype=np.float32)
    else:
        indptr_post = np.array([0, 0, 2, 3, 4], dtype=np.int64)
        indices_pre = np.array([0, 3, 1, 2], dtype=np.int32)
        weight = np.array([5, 2, 3, 4], dtype=np.float32)

    e = len(indices_pre)
    # CSC companion arrays are only validated structurally by the loader.
    post = np.repeat(np.arange(4, dtype=np.int32), np.diff(indptr_post))
    perm = np.lexsort((post, indices_pre))
    counts = np.bincount(indices_pre, minlength=4)
    indptr_pre = np.zeros(5, dtype=np.int64)
    np.cumsum(counts, out=indptr_pre[1:])
    indices_post = post[perm].astype(np.int32)

    path = tmp_path / "tiny-v2.npz"
    np.savez(
        path,
        bodyId=np.array([101, 102, 103, 104], dtype=np.int64),
        node_idx=np.arange(4, dtype=np.int32),
        type_id=np.array([0, 1, 2, 3], dtype=np.int32),
        sign=np.array([1, 1, 1, -1], dtype=np.int8),
        is_clamped=np.array([True, False, False, False]),
        is_dynamic=np.array([False, True, True, True]),
        is_output=np.array([False, False, False, True]),
        is_visual_projection=np.array([False, False, True, False]),
        indptr_post=indptr_post,
        indices_pre=indices_pre,
        weight=weight,
        indptr_pre=indptr_pre,
        indices_post=indices_post,
        perm_csr_to_csc=perm.astype(np.int64),
        forward_hops=np.array([0, 1, 2, 3], dtype=np.int8),
        backward_hops=np.array([-1, 2, 1, 0], dtype=np.int8),
    )
    return path


def test_v2_substrate_rejects_recurrent_edges_into_clamped_cells(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="edges into clamped"):
        MaleCNSV2Substrate.load(
            _tiny_substrate(tmp_path, incoming_to_clamped=True),
            strict=False,
        )


def test_frozen_v2_core_has_zero_trainable_parameters(tmp_path: Path) -> None:
    substrate = MaleCNSV2Substrate.load(
        _tiny_substrate(tmp_path),
        strict=False,
    )
    core = FrozenMaleCNSCore(
        substrate,
        decision_ms=200,
        internal_steps=10,
        device="cpu",
    )

    assert sum(parameter.numel() for parameter in core.parameters()) == 0
    assert core.n_clamped == 1
    assert core.n_dynamic == 3
    assert core.n_readout == 2


def test_frozen_v2_core_runs_recurrent_decision_and_exposes_rate_deltas(
    tmp_path: Path,
) -> None:
    substrate = MaleCNSV2Substrate.load(
        _tiny_substrate(tmp_path),
        strict=False,
    )
    core = FrozenMaleCNSCore(
        substrate,
        decision_ms=200,
        internal_steps=10,
        device="cpu",
    )
    gain = core.match_rest_gain(target=0.5, iterations=20)

    state = core.init_state(1)
    state = core.step(state, torch.ones(1, core.n_clamped))
    features = core.decision_features(state)

    assert gain["w0"] > 0
    assert torch.isfinite(state.h).all()
    assert torch.isfinite(features).all()
    assert features.shape == (1, 2 * core.n_readout)
    assert core.rates(state).shape == (4, 1)


def test_v2_core_uses_ten_20ms_substeps_for_crossy(tmp_path: Path) -> None:
    substrate = MaleCNSV2Substrate.load(
        _tiny_substrate(tmp_path),
        strict=False,
    )
    core = FrozenMaleCNSCore(
        substrate,
        decision_ms=200,
        internal_steps=10,
        tau_ms=20,
        device="cpu",
    )

    assert core.internal_steps == 10
    assert core.decision_ms / core.internal_steps == pytest.approx(20.0)
    assert core.integration_coefficient == pytest.approx(1.0)
