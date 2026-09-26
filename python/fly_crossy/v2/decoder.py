from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from fly_crossy.schema import ACTION_ORDER


DECODER_VERSION = "malecns-crossy-v2-decoder-1"
FEATURE_KIND = "dn-vpn-current-plus-delta"
EXPECTED_INPUT_SIZE = 21_022
HIDDEN_SIZE = 256


class MaleCNSActionDecoder(nn.Module):
    """Small trainable readout over a completely frozen MaleCNS."""

    def __init__(self, input_size: int = EXPECTED_INPUT_SIZE) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.GELU(),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Linear(HIDDEN_SIZE, len(ACTION_ORDER)),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


@dataclass(frozen=True, slots=True)
class DecoderBundle:
    model: MaleCNSActionDecoder
    mean: Tensor
    std: Tensor
    metadata: dict[str, object]


def save_decoder(
    path: str | Path,
    *,
    model: MaleCNSActionDecoder,
    mean: np.ndarray,
    std: np.ndarray,
    metadata: dict[str, object],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": DECODER_VERSION,
        "featureKind": FEATURE_KIND,
        "inputSize": EXPECTED_INPUT_SIZE,
        "hiddenSize": HIDDEN_SIZE,
        "actions": [action.value for action in ACTION_ORDER],
        "mean": torch.as_tensor(mean, dtype=torch.float32).cpu(),
        "std": torch.as_tensor(std, dtype=torch.float32).cpu(),
        "stateDict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_decoder(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> DecoderBundle:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if payload.get("version") != DECODER_VERSION:
        raise ValueError("Unexpected MaleCNS V2 decoder version.")
    if payload.get("featureKind") != FEATURE_KIND:
        raise ValueError("Unexpected MaleCNS V2 decoder feature contract.")
    if payload.get("inputSize") != EXPECTED_INPUT_SIZE:
        raise ValueError("Unexpected MaleCNS V2 decoder input size.")
    if payload.get("hiddenSize") != HIDDEN_SIZE:
        raise ValueError("Unexpected MaleCNS V2 decoder hidden size.")
    if payload.get("actions") != [action.value for action in ACTION_ORDER]:
        raise ValueError("Unexpected MaleCNS V2 decoder action order.")

    dev = torch.device(device)
    model = MaleCNSActionDecoder(EXPECTED_INPUT_SIZE).to(dev)
    model.load_state_dict(payload["stateDict"])
    model.eval()

    mean = torch.as_tensor(payload["mean"], dtype=torch.float32, device=dev)
    std = torch.as_tensor(payload["std"], dtype=torch.float32, device=dev)
    if mean.shape != (EXPECTED_INPUT_SIZE,) or std.shape != (EXPECTED_INPUT_SIZE,):
        raise ValueError("Decoder normalization has an invalid shape.")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Decoder normalization contains non-finite values.")
    if torch.any(std <= 0):
        raise ValueError("Decoder standard deviations must be positive.")

    return DecoderBundle(
        model=model,
        mean=mean,
        std=std,
        metadata=dict(payload.get("metadata", {})),
    )


@torch.no_grad()
def decoder_logits(bundle: DecoderBundle, features: Tensor) -> Tensor:
    x = torch.as_tensor(
        features,
        dtype=torch.float32,
        device=bundle.mean.device,
    )
    if x.shape[-1] != EXPECTED_INPUT_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_INPUT_SIZE} decoder features, got {x.shape[-1]}."
        )
    return bundle.model((x - bundle.mean) / bundle.std)
