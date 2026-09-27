from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from fly_crossy.v6.structured_retina_core import (
    RETINA_CHANNELS,
    StructuredRetinaPolicy,
    _exact_key,
    _structured_slots,
    build_multihop_masks,
)


class FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 7
        self.edge_gain = nn.Parameter(torch.zeros(6))
        self.leak = nn.Parameter(torch.zeros(7))
        self.register_buffer("rows", torch.tensor([2, 3, 4, 5, 6, 1], dtype=torch.long))

    def forward(self, state, *, steps=1, drive=None):
        return state if drive is None else state + drive


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = FakeCore()
        self.register_buffer("sensory_ids", torch.tensor([0, 1], dtype=torch.long))
        self.register_buffer("motor_ids", torch.tensor([5, 6], dtype=torch.long))

    def zero_state(self, batch, *, device=None):
        return torch.zeros(7, batch, device=device)

    def motor_state(self, state):
        return state[self.motor_ids].T


def test_exact_key_prefers_more_progress():
    a = {"success": False, "score": 40.0, "steps": 60}
    b = {"success": False, "score": 41.0, "steps": 50}
    assert _exact_key(b) > _exact_key(a)


def test_structured_slots_cover_image_evenly():
    channels, yy, xx = _structured_slots(4114)
    assert len(channels) == 4114
    assert channels.min() >= 0 and channels.max() < RETINA_CHANNELS
    assert yy.min() == 0 and yy.max() == 23
    assert xx.min() == 0 and xx.max() == 47
    # Every image row and column must receive sensory receptors.
    assert len(np.unique(yy)) == 24
    assert len(np.unique(xx)) == 48


def test_retina_drive_is_local_and_has_expected_shape():
    base = FakeBase()
    policy = StructuredRetinaPolicy(base)
    image = torch.full((1, 24, 48, 3), 0.75)
    drive = policy.drive_for(image)
    assert drive.shape == (7, 1)
    assert torch.isfinite(drive).all()
    assert not torch.allclose(drive[0], torch.zeros_like(drive[0]))


def test_multihop_masks_expand_from_sensory_sources():
    base = FakeBase()
    # Edge sources. Sensory sources are 0/1, with rows 2 and 4 as first hop.
    col = np.asarray([0, 2, 1, 4, 3, 6], dtype=np.int64)
    masks = build_multihop_masks(
        base=base,
        graph_col=col,
        sensory_ids=np.asarray([0, 1], dtype=np.int64),
        device=torch.device("cpu"),
    )
    assert masks["edgeHop1"].tolist() == [True, False, True, False, False, False]
    # Hop2 sources are sensory 0/1 plus first-hop 2/4.
    assert masks["edgeHop2"].tolist() == [True, True, True, True, False, False]
    assert masks["report"]["firstHopNeurons"] == 2
