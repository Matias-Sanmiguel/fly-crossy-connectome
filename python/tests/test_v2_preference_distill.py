from __future__ import annotations

import numpy as np
import torch

from fly_crossy.env import create_game, step_game
from fly_crossy.expert_planner import plan_action
from fly_crossy.schema import ACTION_ORDER

from fly_crossy.v2.preference_distill import (
    pairwise_targets,
    planner_action_preferences,
    preference_loss,
    primary_acceptable_mask,
)


def test_preference_search_matches_canonical_planner() -> None:
    state = create_game("preference-unit-parity")

    for _ in range(6):
        _, best_index, _ = planner_action_preferences(state, depth=4)
        decision = plan_action(state, depth=4)
        assert ACTION_ORDER[best_index] == decision.action

        state = step_game(state, decision.action).state
        if state.terminal is not None:
            break


def test_primary_acceptable_set_keeps_secondary_ties_as_valid() -> None:
    values = np.asarray(
        [
            [1, 1, 1, 5, 1, 0],
            [1, 1, 1, 3, 0, -1],
            [1, 0, 0, 5, 1, 0],
            [0, 2, 2, 0, 0, 0],
            [1, 0, 1, 5, 1, 0],
        ],
        dtype=np.float32,
    )

    mask = primary_acceptable_mask(values)
    assert mask.tolist() == [True, True, False, False, False]

    signs, weights = pairwise_targets(values)
    assert (signs != 0).any()
    assert weights.max() >= 2.0


def test_preference_loss_rewards_valid_set_and_safe_ordering() -> None:
    acceptable = torch.tensor(
        [[True, True, False, False, False]], dtype=torch.bool
    )
    immediate_safe = torch.tensor(
        [[True, True, False, True, True]], dtype=torch.bool
    )

    pair_sign = torch.zeros((1, 10), dtype=torch.float32)
    pair_weight = torch.zeros((1, 10), dtype=torch.float32)
    pair_sign[:, 0] = 1.0
    pair_weight[:, 0] = 4.0
    sample_weight = torch.ones(1)

    good = torch.tensor([[4.0, 3.0, -3.0, 0.0, 0.0]])
    bad = torch.tensor([[-3.0, -2.0, 5.0, 0.0, 0.0]])

    good_loss, _ = preference_loss(
        good,
        acceptable,
        immediate_safe,
        pair_sign,
        pair_weight,
        sample_weight,
    )
    bad_loss, _ = preference_loss(
        bad,
        acceptable,
        immediate_safe,
        pair_sign,
        pair_weight,
        sample_weight,
    )

    assert good_loss < bad_loss
