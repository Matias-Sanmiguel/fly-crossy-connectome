from __future__ import annotations

import pytest

from fly_crossy.connectome import (
    load_default_reduced_graph,
    load_reduced_graph_variant,
)


def test_historical_default_connectome_remains_80_cells() -> None:
    default = load_default_reduced_graph()
    explicit = load_reduced_graph_variant("80")

    assert default.node_count == 80
    assert explicit.node_count == 80
    assert default.artifact_sha256 == explicit.artifact_sha256
    assert set(default.body_ids.tolist()) == set(explicit.body_ids.tolist())


def test_capacity_connectome_1k_contains_full_80_cell_core() -> None:
    core = load_reduced_graph_variant("80")
    capacity = load_reduced_graph_variant("1k")

    assert capacity.node_count == 1000
    assert set(core.body_ids.tolist()).issubset(
        set(capacity.body_ids.tolist())
    )
    assert set(core.sensory_body_ids.tolist()) == set(
        capacity.sensory_body_ids.tolist()
    )
    assert set(core.readout_body_ids.tolist()) == set(
        capacity.readout_body_ids.tolist()
    )
    assert capacity.edge_count >= core.edge_count


def test_unknown_connectome_variant_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unknown connectome graph variant"):
        load_reduced_graph_variant("bogus")
