from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fly_crossy.schema import ACTION_ORDER
from fly_crossy.v2.decoder import (
    EXPECTED_INPUT_SIZE,
    HIDDEN_SIZE,
    MaleCNSActionDecoder,
    decoder_logits,
    load_decoder,
    save_decoder,
)


def test_decoder_architecture_is_only_256_hidden_units() -> None:
    model = MaleCNSActionDecoder()

    assert EXPECTED_INPUT_SIZE == 21_022
    assert HIDDEN_SIZE == 256
    assert sum(parameter.numel() for parameter in model.parameters()) > 0

    linear_layers = [
        module for module in model.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    assert len(linear_layers) == 2
    assert linear_layers[0].in_features == EXPECTED_INPUT_SIZE
    assert linear_layers[0].out_features == 256
    assert linear_layers[1].out_features == len(ACTION_ORDER)


def test_decoder_roundtrip_preserves_logits(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = MaleCNSActionDecoder()
    mean = np.zeros(EXPECTED_INPUT_SIZE, dtype=np.float32)
    std = np.ones(EXPECTED_INPUT_SIZE, dtype=np.float32)
    features = torch.randn(2, EXPECTED_INPUT_SIZE)

    path = tmp_path / "decoder.pt"
    save_decoder(
        path,
        model=model,
        mean=mean,
        std=std,
        metadata={"test": True},
    )
    loaded = load_decoder(path)

    with torch.no_grad():
        expected = model(features)
        actual = decoder_logits(loaded, features)

    assert torch.allclose(expected, actual)
    assert loaded.metadata["test"] is True
