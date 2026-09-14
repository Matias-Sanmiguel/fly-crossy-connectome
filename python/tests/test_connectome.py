from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fly_crossy.connectome import (
    ReducedGraphArtifact,
    build_reduced_graph,
    load_default_reduced_graph,
)


SELECTED_FIXTURE_IDS = (101, 102, 103)


def _write_source(tmp_path: Path, source: object) -> Path:
    source_path = tmp_path / "graph.json"
    source_path.write_text(
        json.dumps(source, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "MaleCNS fixture v1.0",
                "source": "https://example.test/malecns-fixture",
                "license": "CC BY 4.0",
                "selection": "Declared fixture cells; all internal edges retained.",
                "graphSha256": source_sha256,
            }
        ),
        encoding="utf-8",
    )
    return source_path


@pytest.fixture
def source_fixture(tmp_path: Path) -> Path:
    source = {
        "version": "fixture-circuit-v1",
        "nodes": [
            {"id": 103, "sign": -1, "role": "output"},
            {"id": 101, "sign": 1, "role": "input"},
            {"id": 102, "sign": 1, "role": "interneuron"},
        ],
        "edges": [
            [1, 2, 2],
            [2, 0, 4],
            [0, 2, 2],
        ],
        "inputs": [[1, 0]],
        "outputs": [0],
    }
    return _write_source(tmp_path, source)


def test_body_id_edge_fields_cannot_be_misread_as_small_node_indices(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path,
        {
            "version": "body-id-fixture-v1",
            "nodes": [
                {"id": 2, "sign": 1, "role": "output"},
                {"id": 3, "sign": 1, "role": "interneuron"},
                {"id": 1, "sign": 1, "role": "input"},
            ],
            "edges": [{"body_pre": 1, "body_post": 2, "weight": 4}],
            "inputs": [[2, 0]],
            "outputs": [0],
        },
    )

    graph = build_reduced_graph(
        source, selected_ids=(1, 2, 3), atlas_visible_ids=(1, 2, 3)
    )

    assert graph.edge_index.tolist() == [[0], [1]]


def test_source_target_edge_fields_are_explicit_node_indices(tmp_path: Path) -> None:
    source = _write_source(
        tmp_path,
        {
            "version": "index-fixture-v1",
            "nodes": [
                {"id": 2, "sign": 1, "role": "output"},
                {"id": 3, "sign": 1, "role": "interneuron"},
                {"id": 1, "sign": 1, "role": "input"},
            ],
            "edges": [{"source": 1, "target": 2, "weight": 4}],
            "inputs": [[2, 0]],
            "outputs": [0],
        },
    )

    graph = build_reduced_graph(
        source, selected_ids=(1, 2, 3), atlas_visible_ids=(1, 2, 3)
    )

    assert graph.edge_index.tolist() == [[2], [0]]


def test_reduced_graph_preserves_declared_edges_and_ids(source_fixture: Path) -> None:
    source_sha256 = hashlib.sha256(source_fixture.read_bytes()).hexdigest()

    graph = build_reduced_graph(
        source_fixture,
        selected_ids=SELECTED_FIXTURE_IDS,
        atlas_visible_ids=SELECTED_FIXTURE_IDS,
    )

    assert graph.body_ids.tolist() == sorted(SELECTED_FIXTURE_IDS)
    assert graph.edge_index.shape == (2, 3)
    assert np.isfinite(graph.edge_weight).all()
    assert graph.source_sha256 == source_sha256
    assert graph.dataset_version == "MaleCNS fixture v1.0"
    assert graph.source_url == "https://example.test/malecns-fixture"
    assert graph.license == "CC BY 4.0"
    assert graph.minimum_edge_threshold == 1
    assert graph.sensory_body_ids.tolist() == [101]
    assert graph.readout_body_ids.tolist() == [103]
    assert len(graph.artifact_sha256) == 64


def test_reduced_graph_normalizes_signed_contacts_by_target(
    source_fixture: Path,
) -> None:
    graph = build_reduced_graph(
        source_fixture,
        selected_ids=SELECTED_FIXTURE_IDS,
        atlas_visible_ids=SELECTED_FIXTURE_IDS,
    )
    edges = {
        (int(source), int(target)): float(weight)
        for (source, target), weight in zip(graph.edge_index.T, graph.edge_weight, strict=True)
    }

    assert edges[(0, 1)] == pytest.approx(0.5)
    assert edges[(1, 2)] == pytest.approx(1.0)
    assert edges[(2, 1)] == pytest.approx(-0.5)


def test_reduced_graph_rejects_unknown_atlas_ids_and_empty_populations(
    source_fixture: Path,
) -> None:
    with pytest.raises(ValueError, match="atlas-visible"):
        build_reduced_graph(
            source_fixture,
            selected_ids=SELECTED_FIXTURE_IDS,
            atlas_visible_ids=(101, 102),
        )
    with pytest.raises(ValueError, match="sensory"):
        build_reduced_graph(
            source_fixture,
            selected_ids=(102, 103),
            atlas_visible_ids=SELECTED_FIXTURE_IDS,
        )
    with pytest.raises(ValueError, match="readout"):
        build_reduced_graph(
            source_fixture,
            selected_ids=(101, 102),
            atlas_visible_ids=SELECTED_FIXTURE_IDS,
        )


@pytest.mark.parametrize("invalid", [101.9, "101", True])
def test_reduced_graph_rejects_coercible_non_integer_selected_ids(
    source_fixture: Path, invalid: object
) -> None:
    with pytest.raises(ValueError, match="selected IDs.*positive integers"):
        build_reduced_graph(
            source_fixture,
            selected_ids=(invalid, 102, 103),
            atlas_visible_ids=SELECTED_FIXTURE_IDS,
        )


@pytest.mark.parametrize("invalid", [101.9, "101", True])
def test_reduced_graph_rejects_coercible_non_integer_atlas_ids(
    source_fixture: Path, invalid: object
) -> None:
    with pytest.raises(ValueError, match="atlas-visible body IDs.*positive integers"):
        build_reduced_graph(
            source_fixture,
            selected_ids=SELECTED_FIXTURE_IDS,
            atlas_visible_ids=(invalid, 102, 103),
        )


@pytest.mark.parametrize(
    ("argument", "values", "label"),
    [
        ("sensory_ids", (101, 999), "sensory"),
        ("readout_ids", (103, 999), "readout"),
        ("sensory_ids", (101.9,), "sensory"),
    ],
)
def test_reduced_graph_rejects_every_invalid_explicit_population_id(
    source_fixture: Path, argument: str, values: tuple[object, ...], label: str
) -> None:
    with pytest.raises(ValueError, match=rf"{label}.*(?:positive integers|selected)"):
        build_reduced_graph(
            source_fixture,
            selected_ids=SELECTED_FIXTURE_IDS,
            atlas_visible_ids=SELECTED_FIXTURE_IDS,
            **{argument: values},
        )


def test_reduced_graph_rejects_checksum_mismatch_and_non_finite_weights(
    source_fixture: Path,
) -> None:
    manifest_path = source_fixture.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["graphSha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        build_reduced_graph(
            source_fixture,
            selected_ids=SELECTED_FIXTURE_IDS,
            atlas_visible_ids=SELECTED_FIXTURE_IDS,
        )

    manifest["graphSha256"] = hashlib.sha256(source_fixture.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source = json.loads(source_fixture.read_text(encoding="utf-8"))
    source["edges"][0][2] = float("nan")
    source_fixture.write_text(json.dumps(source), encoding="utf-8")
    manifest["graphSha256"] = hashlib.sha256(source_fixture.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        build_reduced_graph(
            source_fixture,
            selected_ids=SELECTED_FIXTURE_IDS,
            atlas_visible_ids=SELECTED_FIXTURE_IDS,
        )


def test_reduced_graph_artifact_rejects_duplicate_topology(
    source_fixture: Path,
) -> None:
    graph = build_reduced_graph(
        source_fixture,
        selected_ids=SELECTED_FIXTURE_IDS,
        atlas_visible_ids=SELECTED_FIXTURE_IDS,
    )
    duplicate_edges = np.concatenate((graph.edge_index, graph.edge_index[:, :1]), axis=1)
    duplicate_weights = np.concatenate((graph.edge_weight, graph.edge_weight[:1]))

    with pytest.raises(ValueError, match="duplicate"):
        replace(graph, edge_index=duplicate_edges, edge_weight=duplicate_weights)


def test_reduced_graph_filters_edges_below_declared_threshold(
    source_fixture: Path,
) -> None:
    graph = build_reduced_graph(
        source_fixture,
        selected_ids=SELECTED_FIXTURE_IDS,
        atlas_visible_ids=SELECTED_FIXTURE_IDS,
        minimum_edge_threshold=3,
    )

    assert graph.edge_index.shape == (2, 1)
    assert graph.minimum_edge_threshold == 3


def test_reduced_graph_type_is_public() -> None:
    assert ReducedGraphArtifact.__name__ == "ReducedGraphArtifact"


def test_pinned_reduced_graph_matches_its_declared_integrity_counts() -> None:
    graph = load_default_reduced_graph()

    assert graph.node_count == 80
    assert graph.edge_count == 1296
    assert graph.sensory_body_ids.size == 32
    assert graph.readout_body_ids.size == 16
    assert graph.source_sha256 == (
        "2424c9dd2e44534e600aeda1a9058039b1f22a4bd983284a27adc10b22130719"
    )
    assert graph.artifact_sha256 == (
        "e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862"
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("body_ids", [True, 102, 103]),
        ("body_ids", ["101", 102, 103]),
        ("body_ids", [101.5, 102, 103]),
        ("edge_index", [[0, 1, 2], [1, False, 1]]),
        ("edge_index", [[0, 1, 2], [1, "2", 1]]),
        ("sensory_body_ids", [True]),
        ("readout_body_ids", [103.0]),
        ("edge_weight", ["0.5", 1.0, -0.5]),
        ("edge_weight", [True, 1.0, -0.5]),
        ("edge_weight", [float("inf"), 1.0, -0.5]),
    ],
)
def test_checkpoint_graph_rejects_values_that_numpy_would_coerce(
    source_fixture: Path, field: str, invalid: object
) -> None:
    graph = build_reduced_graph(
        source_fixture,
        selected_ids=SELECTED_FIXTURE_IDS,
        atlas_visible_ids=SELECTED_FIXTURE_IDS,
    )
    payload = graph.to_checkpoint()
    payload[field] = invalid

    with pytest.raises(ValueError, match="incompatible reduced graph"):
        ReducedGraphArtifact.from_checkpoint(payload)


@pytest.mark.parametrize(
    "digest",
    ["A" * 64, "g" * 64, "0" * 63, 123],
)
def test_checkpoint_graph_requires_exact_lowercase_artifact_digest(
    source_fixture: Path, digest: object
) -> None:
    graph = build_reduced_graph(
        source_fixture,
        selected_ids=SELECTED_FIXTURE_IDS,
        atlas_visible_ids=SELECTED_FIXTURE_IDS,
    )
    payload = graph.to_checkpoint()
    payload["artifact_sha256"] = digest

    with pytest.raises(ValueError, match="incompatible reduced graph"):
        ReducedGraphArtifact.from_checkpoint(payload)


@pytest.mark.parametrize("invalid", [True, "1", float("nan")])
def test_reduced_graph_threshold_rejects_coercible_non_numbers(
    source_fixture: Path, invalid: object
) -> None:
    graph = build_reduced_graph(
        source_fixture,
        selected_ids=SELECTED_FIXTURE_IDS,
        atlas_visible_ids=SELECTED_FIXTURE_IDS,
    )
    payload = graph.to_checkpoint()
    payload["minimum_edge_threshold"] = invalid

    with pytest.raises(ValueError, match="incompatible reduced graph"):
        ReducedGraphArtifact.from_checkpoint(payload)
