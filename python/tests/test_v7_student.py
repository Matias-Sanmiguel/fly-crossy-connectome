from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from fly_crossy.connectome import load_reduced_graph_variant
from fly_crossy.v7.contracts import FRAME_SHAPE
from fly_crossy.v7.student import (
    FullVisualConnectomeStudent,
    ReducedVisualConnectomeStudent,
    StructuredLocalRetina,
    build_visual_student,
)


@pytest.mark.parametrize("sensory_count", [17, 80])
def test_structured_retina_is_deterministic_local_and_has_no_global_projection(
    sensory_count: int,
) -> None:
    torch.manual_seed(9)
    first = StructuredLocalRetina(sensory_count)
    torch.manual_seed(999)
    second = StructuredLocalRetina(sensory_count)

    assert first.retina_kernel.shape == (8, 3, 3, 3)
    assert torch.equal(first.retina_channel, second.retina_channel)
    assert torch.equal(first.retina_y, second.retina_y)
    assert torch.equal(first.retina_x, second.retina_x)
    assert torch.equal(first.retina_kernel, second.retina_kernel)
    assert not any(
        parameter.ndim == 2 and parameter.numel() >= sensory_count * 24 * 48
        for parameter in first.parameters()
    )


@pytest.mark.parametrize("profile", ["80", "1k"])
def test_reduced_student_has_rgb_only_contract_and_trainable_interfaces(
    profile: str,
) -> None:
    student = build_visual_student(profile, device=torch.device("cpu"))
    assert isinstance(student, ReducedVisualConnectomeStudent)
    batch = 2
    state = student.zero_state(batch)
    dark = torch.zeros((batch, *FRAME_SHAPE))
    bright = torch.ones((batch, *FRAME_SHAPE))

    dark_output = student(dark, state)
    bright_output = student(bright, state)
    node_count = student.dynamics.graph.node_count
    assert dark_output.logits.shape == (batch, 5)
    assert dark_output.neuron_activity.shape == (batch, node_count)
    assert dark_output.next_recurrent_state.shape == (batch, node_count)
    assert all(
        torch.isfinite(value).all()
        for value in (
            dark_output.logits,
            dark_output.neuron_activity,
            dark_output.next_recurrent_state,
        )
    )
    assert not torch.equal(
        dark_output.neuron_activity, bright_output.neuron_activity
    )

    parameters = dict(student.named_parameters())
    assert parameters["retina.retina_kernel"].requires_grad
    assert parameters["retina.sensory_gain"].requires_grad
    assert parameters["retina.sensory_bias"].requires_grad
    assert parameters["dynamics.recurrent_gain"].requires_grad
    assert parameters["dynamics.time_constant"].requires_grad
    assert parameters["dynamics.expansion_logit"].requires_grad
    assert parameters["motor_head.0.weight"].requires_grad
    assert parameters["motor_head.2.weight"].requires_grad
    assert not student.dynamics.core_adjacency.requires_grad
    assert not student.dynamics.expansion_adjacency.requires_grad
    assert not any("observation" in key.lower() or "teacher" in key.lower() for key in student.state_dict())

    with pytest.raises(ValueError):
        student(torch.zeros(batch, 24, 48, 4), state)
    with pytest.raises(ValueError):
        student(dark, torch.zeros(batch, node_count + 1))


class _FakeCore(nn.Module):
    def __init__(self, neurons: int) -> None:
        super().__init__()
        self.n = neurons
        self.register_buffer("crow", torch.arange(neurons + 1))
        self.register_buffer("col", torch.arange(neurons))
        self.edge_gain = nn.Parameter(torch.zeros(neurons))
        self.leak = nn.Parameter(torch.zeros(neurons))

    def forward(self, state, *, steps=1, drive=None):
        assert steps == 4
        return torch.tanh(state + drive)


class _FakeFullPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = _FakeCore(12)
        self.register_buffer("sensory_ids", torch.tensor([0, 2, 4, 6]))
        self.register_buffer("motor_ids", torch.tensor([7, 8, 9, 10, 11]))


def test_full_student_matches_rgb_output_contract_and_freezes_measured_core() -> None:
    student = FullVisualConnectomeStudent(_FakeFullPolicy())
    frames = torch.rand(3, *FRAME_SHAPE)
    output = student(frames, student.zero_state(3))

    assert output.logits.shape == (3, 5)
    assert output.neuron_activity.shape == (3, 12)
    assert output.next_recurrent_state.shape == (3, 12)
    assert not student.base.core.edge_gain.requires_grad
    assert not student.base.core.leak.requires_grad
    assert student.retina.retina_kernel.requires_grad
    assert student.motor_head[0].weight.requires_grad


def test_student_forward_signatures_do_not_accept_privileged_features() -> None:
    for student_type in (ReducedVisualConnectomeStudent, FullVisualConnectomeStudent):
        parameters = inspect.signature(student_type.forward).parameters
        assert tuple(parameters) == ("self", "frames", "recurrent_state")


def test_reduced_constructor_rejects_mismatched_graphs() -> None:
    graph = load_reduced_graph_variant("1k")
    core = load_reduced_graph_variant("80")
    student = ReducedVisualConnectomeStudent(graph, core)
    assert student.dynamics.graph.node_count == graph.node_count
