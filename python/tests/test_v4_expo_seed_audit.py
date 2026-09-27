from __future__ import annotations

import numpy as np

from fly_crossy.v4.expo_seed_audit import (
    grayscale_rgb,
    lane_profile,
    rollout_expert,
)


def test_lane_profile_is_deterministic_and_reports_required_kinds_flag():
    a = lane_profile("v4-audit-test", rows=42)
    b = lane_profile("v4-audit-test", rows=42)
    assert a == b
    assert set(a["firstRowByKind"]) == {"grass", "road", "rail", "river"}
    assert sum(a["laneCounts"].values()) == 42


def test_grayscale_rgb_preserves_shape_and_channels_match():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[..., 0] = 255
    gray = grayscale_rgb(frame)
    assert gray.shape == frame.shape
    assert gray.dtype == np.uint8
    assert np.array_equal(gray[..., 0], gray[..., 1])
    assert np.array_equal(gray[..., 1], gray[..., 2])


def test_expert_rollout_smoke_is_deterministic():
    a = rollout_expert("v4-audit-rollout", depth=1, max_steps=3)
    b = rollout_expert("v4-audit-rollout", depth=1, max_steps=3)
    assert a == b
    assert 1 <= a["steps"] <= 3
    assert sum(a["actions"].values()) == a["steps"]
