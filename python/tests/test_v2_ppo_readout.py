from __future__ import annotations

import numpy as np
import torch

from fly_crossy.v2.ppo_readout import (
    ValueReadout,
    compute_gae,
    normalize_features,
)


def test_gae_stops_at_episode_boundary() -> None:
    rewards = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    values = np.zeros(3, dtype=np.float32)
    next_values = np.zeros(3, dtype=np.float32)
    dones = np.asarray([False, True, False])

    advantages, returns = compute_gae(
        rewards,
        values,
        next_values,
        dones,
        gamma=1.0,
        gae_lambda=1.0,
    )

    # t=1 is terminal, so t=0 may include reward at t=1, but nothing after it.
    assert np.allclose(advantages, [3.0, 2.0, 3.0])
    assert np.allclose(returns, advantages)


def test_feature_normalization_is_clipped() -> None:
    x = torch.tensor([100.0, -100.0, 1.0])
    mean = torch.zeros(3)
    std = torch.ones(3)

    normalized = normalize_features(x, mean, std)
    assert torch.allclose(normalized, torch.tensor([10.0, -10.0, 1.0]))


def test_critic_is_training_only_scalar_readout() -> None:
    critic = ValueReadout()
    x = torch.zeros(4, 21_022)
    y = critic(x)

    assert y.shape == (4,)
    assert sum(parameter.numel() for parameter in critic.parameters()) > 0
