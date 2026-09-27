from __future__ import annotations

import numpy as np

from fly_crossy.v4.policy import CHANNELS, FEATURES, IMAGE_H, IMAGE_W
from fly_crossy.v4.train_expo_specialist import (
    EXPO_SEED,
    SEMANTIC_DIM,
    resize_rgb,
    semantic_target,
    variant_state,
)


def test_resize_rgb_preserves_three_distinct_channels():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[..., 0] = 255
    frame[..., 1] = 64
    frame[..., 2] = 7
    resized = resize_rgb(frame)
    assert resized.shape == (IMAGE_H, IMAGE_W, CHANNELS)
    assert resized.dtype == np.float32
    assert np.allclose(resized[..., 0], 1.0)
    assert np.allclose(resized[..., 1], 64 / 255.0)
    assert np.allclose(resized[..., 2], 7 / 255.0)
    assert FEATURES == IMAGE_H * IMAGE_W * CHANNELS


def test_variant_state_changes_phase_and_column_without_changing_seed():
    state = variant_state(EXPO_SEED, phase_offset=0.8, initial_column=-1)
    assert state.seed == EXPO_SEED
    assert state.time == 0.8
    assert state.fly.column == -1.0


def test_semantic_target_has_expected_training_only_shape():
    state = variant_state(EXPO_SEED, phase_offset=0.0, initial_column=0)
    target = semantic_target(state)
    assert target.shape == (SEMANTIC_DIM,)
    assert target.dtype == np.float32
    assert np.isfinite(target).all()
    assert target[:4].sum() == 1.0
    assert target[4:8].sum() == 1.0
