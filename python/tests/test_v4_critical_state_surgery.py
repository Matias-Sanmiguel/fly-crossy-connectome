from types import SimpleNamespace

import numpy as np

from fly_crossy.v4.critical_state_surgery import (
    _surgery_key,
    nearest_rows,
    select_correction_indices,
    select_primary_critical,
)


def _snap(*, safe=True, acceptable=True, student=0, best=0):
    safe_mask = np.zeros(5, dtype=np.bool_)
    acceptable_mask = np.zeros(5, dtype=np.bool_)
    safe_mask[student] = safe
    acceptable_mask[student] = acceptable
    if best != student:
        safe_mask[best] = True
        acceptable_mask[best] = True
    return SimpleNamespace(
        safe=safe_mask,
        acceptable=acceptable_mask,
        student_action=student,
        best_action=best,
    )


def test_primary_prefers_first_unsafe():
    snaps = [
        _snap(),
        _snap(acceptable=False, student=0, best=1),
        _snap(safe=False, acceptable=False, student=0, best=2),
        _snap(safe=False, acceptable=False, student=0, best=3),
    ]
    assert select_primary_critical(snaps) == (2, "first-unsafe")


def test_corrections_stay_small_and_include_primary():
    snaps = [_snap() for _ in range(10)]
    snaps[6] = _snap(acceptable=False, student=0, best=1)
    snaps[8] = _snap(acceptable=False, student=0, best=2)
    snaps[9] = _snap(safe=False, acceptable=False, student=0, best=3)
    chosen = select_correction_indices(snaps, primary=9, tail=4, max_states=2)
    assert 9 in chosen
    assert len(chosen) == 2


def test_nearest_rows_orders_by_cosine_then_l2():
    q = np.asarray([1.0, 0.0], dtype=np.float32)
    m = np.asarray([[1.0, 0.0], [0.5, 0.5], [-1.0, 0.0]], dtype=np.float32)
    meta = [{"id": 0}, {"id": 1}, {"id": 2}]
    rows = nearest_rows(q, m, meta, k=2)
    assert [row["id"] for row in rows] == [0, 1]


def test_surgery_key_rewards_real_progress():
    base = {"success": False, "score": 40.0, "steps": 54}
    better = {"success": False, "score": 41.0, "steps": 45}
    assert _surgery_key(better, 44) > _surgery_key(base, 53)
