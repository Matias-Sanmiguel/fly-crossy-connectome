from __future__ import annotations

import numpy as np

from fly_crossy.dagger_train import (
    _action_pools,
    _sample_hybrid_anchors,
)
from fly_crossy.schema import ACTION_ORDER


def test_hybrid_replay_includes_supported_backward_without_uniformly_balancing_all_data() -> None:
    actions = np.concatenate(
        [
            np.full(700, 0, dtype=np.int64),
            np.full(80, 1, dtype=np.int64),
            np.full(100, 2, dtype=np.int64),
            np.full(100, 3, dtype=np.int64),
            np.full(80, 4, dtype=np.int64),
        ]
    )
    indices = np.arange(len(actions), dtype=np.int64)
    pools = _action_pools(actions, indices)

    assert "backward" in pools
    assert set(pools) == {action.value for action in ACTION_ORDER}

    disagreements = indices[::3]
    anchors = _sample_hybrid_anchors(
        train_indices=indices,
        disagreement_indices=disagreements,
        action_pools=pools,
        count=1000,
        rng=np.random.default_rng(7),
        natural_fraction=0.5,
        disagreement_fraction=0.25,
    )

    sampled_actions = actions[anchors]
    assert len(anchors) == 1000
    assert np.any(sampled_actions == 1)
    assert np.any(np.isin(anchors, disagreements))
    # Hybrid replay must not become the old 20%-per-class uniform prior.
    assert float(np.mean(sampled_actions == 0)) > 0.30
