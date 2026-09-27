from __future__ import annotations

import numpy as np
import pytest

from fly_crossy.v4.recurrent_state_drift_audit import (
    ModeSpec,
    parse_modes,
    representation_similarity,
)


def test_parse_modes_maps_reset_and_short_histories() -> None:
    modes = parse_modes("stateful,reset,short4,short8,short16")
    assert modes == [
        ModeSpec("stateful", None),
        ModeSpec("reset", 1),
        ModeSpec("short4", 4),
        ModeSpec("short8", 8),
        ModeSpec("short16", 16),
    ]


def test_parse_modes_deduplicates() -> None:
    modes = parse_modes("stateful,short4,short4")
    assert modes == [ModeSpec("stateful", None), ModeSpec("short4", 4)]


def test_parse_modes_rejects_invalid_short_history() -> None:
    with pytest.raises(ValueError):
        parse_modes("stateful,short0")


def test_representation_similarity_identity() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = representation_similarity(values, values.copy())
    assert result["meanCosineToStateful"] == pytest.approx(1.0)
    assert result["p10CosineToStateful"] == pytest.approx(1.0)
    assert result["meanRelativeL2ToStateful"] == pytest.approx(0.0)
