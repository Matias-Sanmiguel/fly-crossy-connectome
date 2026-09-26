from __future__ import annotations

import numpy as np
import torch

from fly_crossy.v3.connectome_trainable import TrainableMeasuredConnectome
from fly_crossy.v3.policy import FullMaleCNSCrossyPolicy


def tiny_graph():
    return {
        "crow": np.asarray([0, 1, 2], dtype=np.int64),
        "col": np.asarray([1, 0], dtype=np.int64),
        "counts": np.asarray([3.0, 2.0], dtype=np.float32),
    }


def test_full_graph_core_has_trainable_edge_gains_and_leaks():
    core = TrainableMeasuredConnectome(**tiny_graph())
    assert core.edge_gain.requires_grad
    assert core.leak.requires_grad
    assert core.edge_gain.numel() == 2
    assert core.leak.numel() == 2


def test_state_is_recurrent_not_reset_inside_policy_step():
    policy = FullMaleCNSCrossyPolicy(
        tiny_graph(),
        sensory_ids=np.asarray([0]),
        motor_ids=np.asarray([1]),
        seed=7,
    )
    image = torch.ones(1, 32, 64)
    state0 = policy.zero_state(1)
    state1 = policy.step_state(state0, image)
    state2 = policy.step_state(state1, image)
    assert not torch.allclose(state0, state1)
    assert not torch.allclose(state1, state2)


def test_only_connectome_parameters_are_trainable():
    policy = FullMaleCNSCrossyPolicy(
        tiny_graph(),
        sensory_ids=np.asarray([0]),
        motor_ids=np.asarray([1]),
        seed=7,
    )
    assert {name for name, _ in policy.named_parameters()} == {
        "core.edge_gain",
        "core.leak",
    }
