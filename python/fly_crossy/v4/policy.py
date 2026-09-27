from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fly_crossy.schema import ACTION_ORDER
from fly_crossy.v3.connectome_trainable import TrainableMeasuredConnectome

IMAGE_H = 24
IMAGE_W = 48
CHANNELS = 3
FEATURES = IMAGE_H * IMAGE_W * CHANNELS
GRAPH_UPDATES_PER_DECISION = 4


class FullMaleCNSRGBPolicy(nn.Module):
    """Full measured MaleCNS with a frozen RGB sensory and motor interface.

    Every RGB feature is guaranteed to reach at least one ol_sensory neuron when
    the traced sensory population is large enough. Extra sensory neurons sample
    repeated features. Only measured-edge gains and per-neuron leaks are trainable.
    """

    def __init__(self, graph, sensory_ids, motor_ids, *, seed: int = 404):
        super().__init__()
        self.core = TrainableMeasuredConnectome(
            graph["crow"], graph["col"], graph["counts"]
        )
        rng = np.random.default_rng(seed)

        self.register_buffer(
            "sensory_ids", torch.as_tensor(sensory_ids, dtype=torch.long)
        )
        self.register_buffer(
            "motor_ids", torch.as_tensor(motor_ids, dtype=torch.long)
        )

        sensory_count = len(sensory_ids)
        if sensory_count >= FEATURES:
            covered = rng.permutation(FEATURES)
            extra = rng.integers(0, FEATURES, sensory_count - FEATURES)
            feature_ids = np.concatenate([covered, extra])
            feature_ids = feature_ids[rng.permutation(sensory_count)]
        else:
            feature_ids = rng.choice(FEATURES, sensory_count, replace=False)

        self.register_buffer(
            "feature_ids", torch.as_tensor(feature_ids, dtype=torch.long)
        )
        self.register_buffer(
            "input_signs",
            torch.as_tensor(
                rng.choice([-1.0, 1.0], sensory_count), dtype=torch.float32
            ),
        )

        decoder = rng.normal(
            size=(len(ACTION_ORDER), len(motor_ids))
        ).astype(np.float32)
        decoder /= np.sqrt(max(1, len(motor_ids)))
        self.register_buffer("decoder", torch.as_tensor(decoder))

    def zero_state(self, batch: int, *, device=None) -> torch.Tensor:
        return torch.zeros(
            self.core.n,
            batch,
            dtype=torch.float32,
            device=device or self.decoder.device,
        )

    def drive_for(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (
            IMAGE_H,
            IMAGE_W,
            CHANNELS,
        ):
            raise ValueError(
                f"Expected RGB images [B,{IMAGE_H},{IMAGE_W},{CHANNELS}]."
            )
        features = ((images.flatten(1) - 0.5) * 2.0).clamp(-2.0, 2.0)
        drive = torch.zeros(
            self.core.n,
            len(images),
            dtype=torch.float32,
            device=images.device,
        )
        drive[self.sensory_ids] = (
            features[:, self.feature_ids].T * self.input_signs[:, None]
        )
        return drive

    def step_state(self, state: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        return self.core(
            state,
            steps=GRAPH_UPDATES_PER_DECISION,
            drive=self.drive_for(images),
        )

    def readout(self, state: torch.Tensor) -> torch.Tensor:
        return state[self.motor_ids].T @ self.decoder.T

    def motor_state(self, state: torch.Tensor) -> torch.Tensor:
        return state[self.motor_ids].T

    def run_window_with_motor(
        self,
        images_seq: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = []
        motor_states = []
        for k in range(images_seq.shape[0]):
            state = self.step_state(state, images_seq[k])
            logits.append(self.readout(state))
            motor_states.append(self.motor_state(state))
        return torch.stack(logits), torch.stack(motor_states), state

    def calibrate_decoder(self, images: torch.Tensor) -> None:
        with torch.no_grad():
            state = self.step_state(
                self.zero_state(len(images), device=images.device), images
            )
            raw = self.readout(state)
            self.decoder.mul_(0.75 / raw.std().clamp_min(1e-5))
