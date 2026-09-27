import numpy as np

from fly_crossy.v4.exact_seed_branch_curriculum import (
    _parse_stage_ends,
    _stage_windows,
    choose_branch_actions,
)


def test_parse_stage_ends_appends_max_steps_and_sorts():
    assert _parse_stage_ends("100,50,150", 200) == [50, 100, 150, 200]


def test_parse_stage_ends_caps_to_max_steps():
    assert _parse_stage_ends("20,40,100", 40) == [20, 40]


def test_stage_windows_are_incremental():
    assert _stage_windows([50, 100, 200]) == [(0, 50), (50, 100), (100, 200)]


def test_branch_actions_prioritize_safe_unacceptable_and_exclude_best():
    values = np.asarray(
        [
            [1, 3, 3, 3, 1, 0],
            [1, 0, 0, 1, 0, 0],
            [1, 1, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 3, 3, 3, 1, -1],
        ],
        dtype=np.float32,
    )
    acceptable = np.asarray([True, False, False, False, True], dtype=np.bool_)
    safe = np.asarray([True, True, True, False, True], dtype=np.bool_)
    chosen = choose_branch_actions(
        values=values,
        acceptable=acceptable,
        safe=safe,
        best_index=0,
        limit=2,
    )
    assert chosen == [1, 2]


def test_branch_actions_fall_back_to_safe_acceptable_alternate():
    values = np.ones((5, 6), dtype=np.float32)
    acceptable = np.asarray([True, False, False, False, True], dtype=np.bool_)
    safe = np.asarray([True, False, False, False, True], dtype=np.bool_)
    chosen = choose_branch_actions(
        values=values,
        acceptable=acceptable,
        safe=safe,
        best_index=0,
        limit=2,
    )
    assert chosen == [4]
