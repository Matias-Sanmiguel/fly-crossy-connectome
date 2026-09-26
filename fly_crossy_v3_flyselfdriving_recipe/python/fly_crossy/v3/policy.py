from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fly_crossy.schema import ACTION_ORDER
from .connectome_trainable import TrainableMeasuredConnectome

IMAGE_H = 32
IMAGE_W = 64
PIXELS = IMAGE_H * IMAGE_W
GRAPH_UPDATES_PER_DECISION = 4


class FullMaleCNSCrossyPolicy(nn.Module):
    def __init__(self, graph, sensory_ids, motor_ids, *, seed=123):
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
        self.register_buffer(
            "feature_ids",
            torch.as_tensor(
                rng.integers(0, PIXELS, len(sensory_ids)), dtype=torch.long
            ),
        )
        self.register_buffer(
            "input_signs",
            torch.as_tensor(
                rng.choice([-1.0, 1.0], len(sensory_ids)), dtype=torch.float32
            ),
        )
        decoder = rng.normal(
            size=(len(ACTION_ORDER), len(motor_ids))
        ).astype(np.float32)
        decoder /= np.sqrt(max(1, len(motor_ids)))
        self.register_buffer("decoder", torch.as_tensor(decoder))

    def zero_state(self, batch, *, device=None):
        return torch.zeros(
            self.core.n,
            batch,
            dtype=torch.float32,
            device=device or self.decoder.device,
        )

    def drive_for(self, images):
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

    def step_state(self, state, images):
        return self.core(
            state,
            steps=GRAPH_UPDATES_PER_DECISION,
            drive=self.drive_for(images),
        )

    def readout(self, state):
        return state[self.motor_ids].T @ self.decoder.T

    def run_window(self, images_seq, state):
        outputs = []
        for k in range(images_seq.shape[0]):
            state = self.step_state(state, images_seq[k])
            outputs.append(self.readout(state))
        return torch.stack(outputs), state

    def calibrate_decoder(self, images):
        with torch.no_grad():
            state = self.step_state(
                self.zero_state(len(images), device=images.device), images
            )
            raw = self.readout(state)
            self.decoder.mul_(0.75 / raw.std().clamp_min(1e-5))
