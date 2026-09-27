from __future__ import annotations

import numpy as np
import torch

from fly_crossy.v4.audit_capacity import (
    _probe_action_metrics,
    _semantic_metrics,
    _summarize_trace,
)


def test_probe_action_metrics_accepts_alternative_teacher_action():
    logits = torch.tensor([[0.0, 3.0, 1.0, -1.0, -2.0]])
    labels = np.asarray([2], dtype=np.int64)
    acceptable = np.asarray([[False, True, True, False, False]])
    safe = np.asarray([[False, True, True, True, False]])
    result = _probe_action_metrics(logits, labels, acceptable, safe)
    assert result["exactActionRate"] == 0.0
    assert result["acceptableActionRate"] == 1.0
    assert result["immediateSafeActionRate"] == 1.0


def test_semantic_metrics_reports_lane_support_and_zero_mse():
    target = np.zeros((2, 34), dtype=np.float32)
    target[0, 0] = 1.0
    target[0, 5] = 1.0
    target[0, 8] = 1.0
    target[1, 3] = 1.0
    target[1, 4] = 1.0
    pred = target.copy()
    result = _semantic_metrics(pred, target)
    assert result["currentLaneAccuracy"] == 1.0
    assert result["nextLaneAccuracy"] == 1.0
    assert result["supportAccuracy"] == 1.0
    assert result["overallMse"] == 0.0


def test_trace_summary_distinguishes_exact_acceptable_and_safe():
    base_semantic = {
        "currentLaneCorrect": True,
        "nextLaneCorrect": True,
        "supportCorrect": True,
        "trafficMse": 0.0,
        "semanticMse": 0.0,
    }
    trace = [
        {
            "decision": 0,
            "gameStep": 0,
            "time": 0.0,
            "row": 0,
            "column": 0.0,
            "score": 0.0,
            "currentLane": "grass",
            "nextLane": "road",
            "teacherAction": "forward",
            "studentAction": "right",
            "studentAcceptable": True,
            "studentImmediateSafe": True,
            "studentImmediateFatal": False,
            "acceptableActions": ["forward", "right"],
            "immediateSafeActions": ["forward", "right"],
            "semantic": base_semantic,
        },
        {
            "decision": 1,
            "gameStep": 1,
            "time": 0.2,
            "row": 1,
            "column": 0.0,
            "score": 1.0,
            "currentLane": "road",
            "nextLane": "road",
            "teacherAction": "left",
            "studentAction": "forward",
            "studentAcceptable": False,
            "studentImmediateSafe": False,
            "studentImmediateFatal": True,
            "acceptableActions": ["left"],
            "immediateSafeActions": ["left", "wait"],
            "semantic": base_semantic,
        },
    ]
    result = _summarize_trace(trace)
    assert result["exactActionRate"] == 0.0
    assert result["acceptableActionRate"] == 0.5
    assert result["immediateSafeActionRate"] == 0.5
    assert result["immediateFatalChoiceRate"] == 0.5
    assert result["firstUnacceptableAction"]["decision"] == 1
