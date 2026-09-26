from __future__ import annotations

import numpy as np
import torch

from fly_crossy.v2.brain_readout_audit import (
    interpretation,
    _mix64,
)


def test_hash_projection_seed_is_deterministic() -> None:
    ids = np.asarray([10, 20, 30, 40], dtype=np.int64)
    a = _mix64(ids, 109)
    b = _mix64(ids, 109)
    c = _mix64(ids, 110)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_interpretation_requires_both_core_and_correction_gain() -> None:
    baseline = {
        "name": "dn_vpn_readout",
        "oof": {"coreMacroAccuracy": 0.50},
        "disagreementRecovery": 0.35,
        "criticalCorrection": 0.40,
    }
    broad_core_only = {
        "name": "near_dn_hop1",
        "oof": {"coreMacroAccuracy": 0.61},
        "disagreementRecovery": 0.38,
        "criticalCorrection": 0.44,
    }
    broad_corrections = {
        "name": "all_dynamic",
        "oof": {"coreMacroAccuracy": 0.59},
        "disagreementRecovery": 0.46,
        "criticalCorrection": 0.50,
    }

    conclusion = interpretation(
        {
            "dn_vpn_readout": baseline,
            "near_dn_hop1": broad_core_only,
            "near_dn_hop2": broad_corrections,
            "all_dynamic": broad_corrections,
        }
    )
    assert conclusion["readoutBottleneckEvidence"] is True


def test_interpretation_rejects_small_broad_population_gain() -> None:
    baseline = {
        "name": "dn_vpn_readout",
        "oof": {"coreMacroAccuracy": 0.51},
        "disagreementRecovery": 0.35,
        "criticalCorrection": 0.40,
    }
    almost_same = {
        "name": "near_dn_hop1",
        "oof": {"coreMacroAccuracy": 0.55},
        "disagreementRecovery": 0.39,
        "criticalCorrection": 0.43,
    }

    conclusion = interpretation(
        {
            "dn_vpn_readout": baseline,
            "near_dn_hop1": almost_same,
            "near_dn_hop2": almost_same,
            "all_dynamic": almost_same,
        }
    )
    assert conclusion["readoutBottleneckEvidence"] is False
