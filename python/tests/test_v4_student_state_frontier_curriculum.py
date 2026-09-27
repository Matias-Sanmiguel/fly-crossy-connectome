from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fly_crossy.v4.student_state_frontier_curriculum import (
    StudentSnapshot,
    choose_frontier_branch_actions,
    frontier_selection_key,
    select_frontier_roots,
)


def _snapshot(index: int, *, student: int = 0, acceptable=(True, True, True, True, True), safe=(True, True, True, True, True)) -> StudentSnapshot:
    state = SimpleNamespace(step=index, score=float(index), fly=SimpleNamespace(row=index, column=0.0))
    return StudentSnapshot(
        index=index,
        state=state,
        neural_after_frame=None,
        motor=np.zeros(3, dtype=np.float32),
        values=np.tile(np.arange(6, dtype=np.float32), (5, 1)),
        acceptable=np.asarray(acceptable, dtype=np.bool_),
        safe=np.asarray(safe, dtype=np.bool_),
        progress=np.zeros(5, dtype=np.bool_),
        pair_sign=np.zeros(10, dtype=np.int8),
        pair_weight=np.ones(10, dtype=np.float32),
        best_action=0,
        student_action=student,
        semantic=np.zeros(34, dtype=np.float32),
    )


def test_frontier_roots_include_unsafe_and_tail() -> None:
    snaps = [_snapshot(i) for i in range(10)]
    snaps[2] = _snapshot(2, student=1, acceptable=(True, True, True, True, True), safe=(True, False, True, True, True))
    roots = select_frontier_roots(snaps, tail=3, max_roots=5)
    assert 2 in roots
    assert {7, 8, 9}.issubset(set(roots))


def test_branch_actions_start_with_best_then_safe_student() -> None:
    snap = _snapshot(0, student=2, acceptable=(True, False, False, False, False), safe=(True, True, True, True, False))
    chosen = choose_frontier_branch_actions(snap, limit=3)
    assert chosen[0] == 0
    assert chosen[1] == 2
    assert 4 not in chosen


def test_selection_prefers_score_then_steps() -> None:
    def metrics(score: float, steps: int):
        return {
            "exactExpo": {"success": False, "score": score, "steps": steps},
            "robust": {"successes": 0, "meanScore": 1.0, "stalled": 0},
        }

    assert frontier_selection_key(metrics(41, 50)) > frontier_selection_key(metrics(40, 200))
    assert frontier_selection_key(metrics(41, 70)) > frontier_selection_key(metrics(41, 50))
