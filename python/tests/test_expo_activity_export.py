import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("expo_export", Path(__file__).resolve().parents[2] / "scripts/export-expo-activity.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_quantization_preserves_zero_and_signed_state_magnitude():
    state = np.array([0.0, 1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    np.testing.assert_array_equal(export.quantize_activity(state), [0, 255, 255, 128, 128])
    np.testing.assert_array_equal(export.quantize_activity(state), export.quantize_activity(state))


@pytest.mark.parametrize("state", [[float("nan")], [float("inf")], [1.1]])
def test_export_rejects_invalid_state(state):
    with pytest.raises(ValueError):
        export.quantize_activity(np.array(state))


def test_display_selection_is_deterministic_and_preserves_only_real_strongest_values():
    ids = [30, 10, 20, 40]
    frames = np.array([[5, 5, 5, 0], [7, 0, 6, 8]], dtype=np.uint8)
    pool, compact = export.select_display_frames(ids, frames, 2)
    assert pool == ids
    # Ties select lower body IDs; values are not rescaled or fabricated.
    np.testing.assert_array_equal(compact, [[0, 5, 5, 0], [7, 0, 0, 8]])
    repeat_pool, repeat = export.select_display_frames(ids, frames, 2)
    assert repeat_pool == pool
    np.testing.assert_array_equal(repeat, compact)
    np.testing.assert_array_equal(frames, [[5, 5, 5, 0], [7, 0, 6, 8]])


def test_display_pool_excludes_never_selected_ids_and_keeps_zero_dark():
    pool, compact = export.select_display_frames([1, 2, 3], np.array([[0, 9, 1], [0, 2, 8]], dtype=np.uint8), 1)
    assert pool == [2, 3]
    np.testing.assert_array_equal(compact, [[9, 0], [0, 8]])
    pool, compact = export.select_display_frames([1, 2], np.zeros((2, 2), dtype=np.uint8), 2)
    assert pool == []
    assert compact.shape == (2, 0)


def test_invalid_display_limit_is_rejected():
    with pytest.raises(ValueError):
        export.select_display_frames([1], np.array([[1]], dtype=np.uint8), 0)


def test_regional_selection_does_not_let_optic_signals_crowd_out_central_or_descending():
    ids = [1, 2, 3, 4, 5, 6]
    regions = np.array([0, 0, 1, 1, 2, 2])
    frames = np.array([[90, 80, 3, 1, 2, 0], [90, 80, 0, 1, 2, 0]], dtype=np.uint8)
    pool, compact = export.select_display_frames(ids, frames, 3, regions)
    assert pool == [1, 3, 4, 5]
    np.testing.assert_array_equal(compact, [[90, 3, 0, 2], [90, 0, 1, 2]])
    assert (np.count_nonzero(compact, axis=1) <= 3).all()
    repeat_pool, repeat = export.select_display_frames(ids, frames, 3, regions)
    assert repeat_pool == pool
    np.testing.assert_array_equal(repeat, compact)


def test_actual_changes_get_a_slot_but_no_timed_blinks_are_invented():
    ids = [1, 2, 3]
    regions = np.array([1, 1, 1])
    frames = np.array([[9, 8, 1], [9, 8, 2], [9, 8, 2]], dtype=np.uint8)
    pool, compact = export.select_display_frames(ids, frames, 6, regions)
    assert pool == [1, 2, 3]
    np.testing.assert_array_equal(compact, [[9, 8, 0], [9, 0, 2], [9, 8, 0]])
    # After settling, identical states produce identical selections, not a fake oscillator.
    repeated = np.repeat(frames[-1:], 4, axis=0)
    _, settled = export.select_display_frames(ids, repeated, 6, regions)
    np.testing.assert_array_equal(settled[1:], np.repeat(settled[1:2], 3, axis=0))


def test_fixed_region_gain_reveals_weak_real_states_without_turning_zero_on():
    state = np.array([0, 0.001, -0.001])
    np.testing.assert_array_equal(export.quantize_activity(state, np.array([16, 16, 16])), [0, 4, 4])
    with pytest.raises(ValueError):
        export.quantize_activity(state, 0)
