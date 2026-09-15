from __future__ import annotations

from types import SimpleNamespace

import pytest

from fly_crossy.backend import BackendUnavailable, resolve_backend


@pytest.fixture
def fake_torch_without_cuda() -> SimpleNamespace:
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))


def test_gpu_auto_falls_back_to_cpu(fake_torch_without_cuda: SimpleNamespace) -> None:
    status = resolve_backend("auto", fake_torch_without_cuda)

    assert (status.resolved, status.device, status.fallback_reason) == (
        "cpu",
        "cpu",
        "cuda-unavailable",
    )


def test_cpu_preference_does_not_report_a_cuda_fallback(fake_torch_without_cuda: SimpleNamespace) -> None:
    status = resolve_backend("cpu", fake_torch_without_cuda)

    assert status.fallback_reason is None


def test_gpu_strict_rejects_an_unavailable_cuda_device(fake_torch_without_cuda: SimpleNamespace) -> None:
    with pytest.raises(BackendUnavailable, match="requested strictly"):
        resolve_backend("gpu-strict", fake_torch_without_cuda)
