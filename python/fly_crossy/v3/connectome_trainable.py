from __future__ import annotations

import torch
from torch import nn

CUDA_EDGE_CHUNK = 8_388_608


class _EdgeSparseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, crow, col, rows, state):
        n = len(crow) - 1
        matrix = torch.sparse_csr_tensor(
            crow, col, values, size=(n, n), check_invariants=False
        )
        ctx.save_for_backward(values, crow, col, rows, state)
        return torch.sparse.mm(matrix, state)

    @staticmethod
    def backward(ctx, output_grad):
        values, crow, col, rows, state = ctx.saved_tensors
        value_grad = None
        if ctx.needs_input_grad[0]:
            value_grad = torch.empty_like(values)
            chunk = CUDA_EDGE_CHUNK if values.is_cuda else 262_144
            for start in range(0, len(values), chunk):
                end = min(start + chunk, len(values))
                value_grad[start:end] = (
                    output_grad[rows[start:end]] * state[col[start:end]]
                ).sum(dim=1)

        state_grad = None
        if ctx.needs_input_grad[4]:
            n = len(crow) - 1
            matrix = torch.sparse_csr_tensor(
                crow, col, values, size=(n, n), check_invariants=False
            )
            state_grad = torch.sparse.mm(matrix.transpose(0, 1), output_grad)

        return value_grad, None, None, None, state_grad


def edge_matmul(values, crow, col, rows, state):
    return _EdgeSparseMM.apply(values, crow, col, rows, state)


class TrainableMeasuredConnectome(nn.Module):
    def __init__(self, crow, col, counts, *, edge_init=0.0, leak_init=0.0):
        super().__init__()
        self.n = len(crow) - 1
        self.register_buffer("crow", torch.as_tensor(crow, dtype=torch.int64))
        self.register_buffer("col", torch.as_tensor(col, dtype=torch.int64))
        counts = torch.as_tensor(counts, dtype=torch.float32)
        rows = torch.repeat_interleave(
            torch.arange(self.n, dtype=torch.int64), torch.diff(self.crow)
        )
        self.register_buffer("rows", rows)
        totals = torch.zeros(self.n, dtype=torch.float32).index_add_(0, rows, counts)
        self.register_buffer("base", counts / totals[rows].clamp_min(1.0))
        self.edge_gain = nn.Parameter(torch.full_like(counts, float(edge_init)))
        self.leak = nn.Parameter(
            torch.full((self.n,), float(leak_init), dtype=torch.float32)
        )

    def edge_values(self):
        return self.base * (0.05 + 0.90 * torch.sigmoid(self.edge_gain))

    def leak_values(self):
        return 0.05 + 0.90 * torch.sigmoid(self.leak)

    def forward(self, state, *, steps=1, drive=None):
        values = self.edge_values()
        leak = self.leak_values()[:, None]
        for _ in range(steps):
            signal = edge_matmul(values, self.crow, self.col, self.rows, state)
            if drive is not None:
                signal = signal + drive
            state = (1.0 - leak) * state + leak * torch.tanh(signal)
        return state
