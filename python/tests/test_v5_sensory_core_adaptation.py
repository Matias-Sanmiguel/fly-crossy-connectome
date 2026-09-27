from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from fly_crossy.v5.sensory_core_adaptation import (
    SensoryPlasticPolicy,
    _exact_key,
    build_plasticity_masks,
)


class FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 6
        self.edge_gain = nn.Parameter(torch.zeros(5))
        self.leak = nn.Parameter(torch.zeros(6))
        self.register_buffer("rows", torch.tensor([2, 3, 4, 5, 1], dtype=torch.long))

    def forward(self, state, *, steps=1, drive=None):
        return state if drive is None else state + drive


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = FakeCore()
        self.register_buffer("sensory_ids", torch.tensor([0, 1], dtype=torch.long))
        self.register_buffer("motor_ids", torch.tensor([4, 5], dtype=torch.long))
        self.register_buffer("feature_ids", torch.tensor([0, 1], dtype=torch.long))
        self.register_buffer("input_signs", torch.tensor([1.0, -1.0]))

    def zero_state(self, batch, *, device=None):
        return torch.zeros(6, batch, device=device)

    def motor_state(self, state):
        return state[self.motor_ids].T


def test_exact_key_prefers_real_progress():
    a = {"success": False, "score": 40.0, "steps": 54}
    b = {"success": False, "score": 41.0, "steps": 45}
    assert _exact_key(b) > _exact_key(a)


def test_plasticity_masks_select_sensory_outgoing_edges_and_first_hop_leaks():
    base = FakeBase()
    # Edge sources (CSR col): sensory neurons are 0 and 1.
    col = np.asarray([0, 2, 1, 4, 3], dtype=np.int64)
    edge, leak, report = build_plasticity_masks(
        base=base,
        graph_col=col,
        sensory_ids=np.asarray([0, 1], dtype=np.int64),
        device=torch.device("cpu"),
    )
    assert edge.tolist() == [True, False, True, False, False]
    # Sensory 0/1 plus postsynaptic rows 2 and 4.
    assert leak.tolist() == [True, True, True, False, True, False]
    assert report["sensoryOutgoingEdges"] == 2
    assert report["firstHopNeurons"] == 2


def test_adapter_starts_as_exact_baseline_drive(monkeypatch):
    # Patch module constants by feeding a full-sized image but only checking the
    # two mapped sensory entries. Zero residual + zero gain must preserve V4.
    base = FakeBase()
    policy = SensoryPlasticPolicy(base, rank=2, residual_scale=0.25)
    image = torch.full((1, 24, 48, 3), 0.75)
    drive = policy.drive_for(image)
    feature = ((0.75 - 0.5) * 2.0)
    assert torch.allclose(drive[0], torch.tensor([feature]))
    assert torch.allclose(drive[1], torch.tensor([-feature]))
