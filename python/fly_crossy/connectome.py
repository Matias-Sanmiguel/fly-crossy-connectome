from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor


DEFAULT_GRAPH_PATH = (
    Path(__file__).resolve().parents[2] / "public" / "data" / "connectome" / "graph.json"
)
DEFAULT_ATLAS_PATH = (
    Path(__file__).resolve().parents[2] / "public" / "data" / "brain-atlas"
)


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Reduced graph {label} must be declared.")
    return value.strip()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True, slots=True)
class ReducedGraphArtifact:
    """Validated fixed-topology graph plus complete source and derivation metadata."""

    dataset_version: str
    source_url: str
    license: str
    source_sha256: str
    selection_rule: str
    minimum_edge_threshold: float
    body_ids: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    sensory_body_ids: np.ndarray
    readout_body_ids: np.ndarray
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for value, label in (
            (self.dataset_version, "dataset version"),
            (self.source_url, "source URL"),
            (self.license, "license"),
            (self.selection_rule, "selection rule"),
        ):
            _required_text(value, label)
        if not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64:
            raise ValueError("Reduced graph source SHA-256 must be a 64-character digest.")
        try:
            int(self.source_sha256, 16)
        except ValueError as error:
            raise ValueError("Reduced graph source SHA-256 is invalid.") from error
        if (
            not np.isfinite(self.minimum_edge_threshold)
            or self.minimum_edge_threshold <= 0
        ):
            raise ValueError("Reduced graph minimum edge threshold must be positive and finite.")

        body_ids = np.asarray(self.body_ids, dtype=np.int64)
        edge_index = np.asarray(self.edge_index, dtype=np.int64)
        edge_weight = np.asarray(self.edge_weight, dtype=np.float32)
        sensory_ids = np.asarray(self.sensory_body_ids, dtype=np.int64)
        readout_ids = np.asarray(self.readout_body_ids, dtype=np.int64)
        if body_ids.ndim != 1 or body_ids.size == 0 or np.any(body_ids <= 0):
            raise ValueError("Reduced graph body IDs must be a non-empty positive vector.")
        if body_ids.tolist() != sorted(set(int(item) for item in body_ids)):
            raise ValueError("Reduced graph body IDs must be sorted and unique.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Reduced graph edge_index must have shape [2, edge_count].")
        if edge_weight.shape != (edge_index.shape[1],):
            raise ValueError("Reduced graph edge weights must match edge_index.")
        if not np.isfinite(edge_weight).all():
            raise ValueError("Reduced graph edge weights must be finite.")
        if edge_index.size and (
            np.any(edge_index < 0) or np.any(edge_index >= body_ids.size)
        ):
            raise ValueError("Reduced graph topology contains an invalid node index.")
        topology = [tuple(int(item) for item in edge) for edge in edge_index.T]
        if len(topology) != len(set(topology)):
            raise ValueError("Reduced graph topology contains a duplicate edge after aggregation.")
        node_set = set(int(item) for item in body_ids)
        for population, label in (
            (sensory_ids, "sensory"),
            (readout_ids, "readout"),
        ):
            if population.ndim != 1 or population.size == 0:
                raise ValueError(f"Reduced graph {label} population must not be empty.")
            values = [int(item) for item in population]
            if len(values) != len(set(values)) or not set(values).issubset(node_set):
                raise ValueError(
                    f"Reduced graph {label} population must contain unique included body IDs."
                )

        for array in (body_ids, edge_index, edge_weight, sensory_ids, readout_ids):
            array.setflags(write=False)
        object.__setattr__(self, "body_ids", body_ids)
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "edge_weight", edge_weight)
        object.__setattr__(self, "sensory_body_ids", sensory_ids)
        object.__setattr__(self, "readout_body_ids", readout_ids)
        digest = _sha256_bytes(
            json.dumps(
                self._canonical_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        )
        if self.artifact_sha256 and self.artifact_sha256 != digest:
            raise ValueError("Reduced graph artifact SHA-256 does not match its contents.")
        object.__setattr__(self, "artifact_sha256", digest)

    @property
    def node_count(self) -> int:
        return int(self.body_ids.size)

    @property
    def edge_count(self) -> int:
        return int(self.edge_weight.size)

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "source_url": self.source_url,
            "license": self.license,
            "source_sha256": self.source_sha256,
            "selection_rule": self.selection_rule,
            "minimum_edge_threshold": float(self.minimum_edge_threshold),
            "body_ids": [int(item) for item in self.body_ids],
            "edge_index": [[int(item) for item in row] for row in self.edge_index],
            "edge_weight": [float(item) for item in self.edge_weight],
            "sensory_body_ids": [int(item) for item in self.sensory_body_ids],
            "readout_body_ids": [int(item) for item in self.readout_body_ids],
        }

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            **self._canonical_payload(),
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_checkpoint(cls, value: Mapping[str, object]) -> ReducedGraphArtifact:
        try:
            return cls(
                dataset_version=_required_text(
                    value["dataset_version"], "dataset version"
                ),
                source_url=_required_text(value["source_url"], "source URL"),
                license=_required_text(value["license"], "license"),
                source_sha256=_required_text(
                    value["source_sha256"], "source SHA-256"
                ),
                selection_rule=_required_text(
                    value["selection_rule"], "selection rule"
                ),
                minimum_edge_threshold=float(value["minimum_edge_threshold"]),
                body_ids=np.asarray(value["body_ids"], dtype=np.int64),
                edge_index=np.asarray(value["edge_index"], dtype=np.int64),
                edge_weight=np.asarray(value["edge_weight"], dtype=np.float32),
                sensory_body_ids=np.asarray(value["sensory_body_ids"], dtype=np.int64),
                readout_body_ids=np.asarray(value["readout_body_ids"], dtype=np.int64),
                artifact_sha256=str(value["artifact_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Checkpoint contains an incompatible reduced graph.") from error

    def sparse_adjacency(self) -> Tensor:
        """Return target-by-source adjacency for ``torch.sparse.mm``."""
        indices = torch.from_numpy(self.edge_index[[1, 0]].copy())
        values = torch.from_numpy(self.edge_weight.copy())
        with torch.sparse.check_sparse_tensor_invariants():
            return torch.sparse_coo_tensor(
                indices,
                values,
                (self.node_count, self.node_count),
                dtype=torch.float32,
                check_invariants=True,
            ).coalesce()


def _load_source(
    source: str | Path | Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], str]:
    if isinstance(source, Mapping):
        payload = dict(source)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        metadata_value = payload.get("metadata", payload)
        if not isinstance(metadata_value, Mapping):
            raise ValueError("Reduced graph source metadata must be an object.")
        return payload, dict(metadata_value), _sha256_bytes(encoded)

    path = Path(source)
    content = path.read_bytes()
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Reduced graph source must be valid JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError("Reduced graph source must be a JSON object.")
    manifest_path = path.with_name("manifest.json")
    if not manifest_path.is_file():
        raise ValueError("Reduced graph source requires a sibling manifest.json.")
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Reduced graph manifest must be valid JSON.") from error
    if not isinstance(metadata, dict):
        raise ValueError("Reduced graph manifest must be an object.")
    return payload, metadata, _sha256_bytes(content)


def _metadata_value(
    metadata: Mapping[str, object], explicit: str | None, *keys: str
) -> str:
    if explicit is not None:
        return _required_text(explicit, keys[0])
    for key in keys:
        if key in metadata:
            return _required_text(metadata[key], keys[0])
    raise ValueError(f"Reduced graph {keys[0]} must be declared.")


def _node_endpoint(value: object, body_ids: list[int], id_to_source_index: dict[int, int]) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise ValueError("Reduced graph edge endpoints must be integer node indices or body IDs.")
    endpoint = int(value)
    if 0 <= endpoint < len(body_ids):
        return endpoint
    if endpoint in id_to_source_index:
        return id_to_source_index[endpoint]
    raise ValueError("Reduced graph edge topology references an unknown node.")


def build_reduced_graph(
    source: str | Path | Mapping[str, object],
    *,
    selected_ids: Iterable[int],
    atlas_visible_ids: Iterable[int] | None = None,
    dataset_version: str | None = None,
    source_url: str | None = None,
    license: str | None = None,
    source_sha256: str | None = None,
    selection_rule: str | None = None,
    minimum_edge_threshold: float = 1,
    sensory_ids: Iterable[int] | None = None,
    readout_ids: Iterable[int] | None = None,
) -> ReducedGraphArtifact:
    """Construct a deterministic signed, incoming-normalized graph without inventing edges."""
    payload, metadata, computed_source_hash = _load_source(source)
    expected_hash_value = source_sha256 or metadata.get("graphSha256") or metadata.get(
        "source_sha256"
    )
    expected_hash = _required_text(expected_hash_value, "source SHA-256")
    if expected_hash != computed_source_hash:
        raise ValueError("Reduced graph source SHA-256 does not match the source file.")
    if not np.isfinite(minimum_edge_threshold) or minimum_edge_threshold <= 0:
        raise ValueError("Reduced graph minimum edge threshold must be positive and finite.")

    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Reduced graph source must declare a non-empty node table.")
    source_body_ids: list[int] = []
    source_signs: list[float] = []
    source_roles: list[str] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            raise ValueError("Reduced graph source nodes must be objects.")
        body_id_value = node.get("id", node.get("bodyId"))
        sign_value = node.get("sign")
        role_value = node.get("role", "interneuron")
        if (
            not isinstance(body_id_value, (int, np.integer))
            or isinstance(body_id_value, bool)
            or int(body_id_value) <= 0
        ):
            raise ValueError("Reduced graph source node body IDs must be positive integers.")
        if not isinstance(sign_value, (int, float, np.integer, np.floating)) or not np.isfinite(
            sign_value
        ):
            raise ValueError("Reduced graph source node signs must be finite.")
        if role_value not in ("input", "interneuron", "output"):
            raise ValueError("Reduced graph source node roles are invalid.")
        source_body_ids.append(int(body_id_value))
        source_signs.append(float(sign_value))
        source_roles.append(str(role_value))
    if len(source_body_ids) != len(set(source_body_ids)):
        raise ValueError("Reduced graph source contains a duplicate body ID.")
    id_to_source_index = {body_id: index for index, body_id in enumerate(source_body_ids)}

    selected = sorted(set(int(item) for item in selected_ids))
    if not selected or any(item <= 0 for item in selected):
        raise ValueError("Reduced graph selected IDs must be positive and non-empty.")
    missing_source = set(selected) - set(source_body_ids)
    if missing_source:
        raise ValueError("Reduced graph selected IDs are absent from the source node table.")
    if atlas_visible_ids is None:
        declared_atlas = payload.get("atlasVisibleBodyIds", payload.get("atlas_visible_body_ids"))
        if not isinstance(declared_atlas, list):
            raise ValueError("Reduced graph atlas-visible body IDs must be supplied.")
        atlas_visible_ids = declared_atlas
    visible = {int(item) for item in atlas_visible_ids}
    if not set(selected).issubset(visible):
        raise ValueError("Reduced graph selected IDs must all be atlas-visible body IDs.")

    source_sensory = {
        source_body_ids[index]
        for index, role in enumerate(source_roles)
        if role == "input"
    }
    inputs = payload.get("inputs")
    if isinstance(inputs, list):
        for item in inputs:
            if isinstance(item, list) and item:
                source_sensory.add(
                    source_body_ids[_node_endpoint(item[0], source_body_ids, id_to_source_index)]
                )
    source_readout = {
        source_body_ids[index]
        for index, role in enumerate(source_roles)
        if role == "output"
    }
    outputs = payload.get("outputs")
    if isinstance(outputs, list):
        for item in outputs:
            source_readout.add(
                source_body_ids[_node_endpoint(item, source_body_ids, id_to_source_index)]
            )
    selected_sensory = sorted(
        set(int(item) for item in sensory_ids) if sensory_ids is not None else source_sensory
    )
    selected_readout = sorted(
        set(int(item) for item in readout_ids) if readout_ids is not None else source_readout
    )
    selected_sensory = [item for item in selected_sensory if item in set(selected)]
    selected_readout = [item for item in selected_readout if item in set(selected)]
    if not selected_sensory:
        raise ValueError("Reduced graph sensory population must not be empty.")
    if not selected_readout:
        raise ValueError("Reduced graph readout population must not be empty.")

    raw_edges = payload.get("edges")
    if not isinstance(raw_edges, list):
        raise ValueError("Reduced graph source must declare an edge table.")
    aggregated: dict[tuple[int, int], float] = {}
    selected_set = set(selected)
    for edge in raw_edges:
        if isinstance(edge, Mapping):
            source_value = edge.get("source", edge.get("body_pre"))
            target_value = edge.get("target", edge.get("body_post"))
            weight_value = edge.get("weight", edge.get("contacts"))
        elif isinstance(edge, list) and len(edge) == 3:
            source_value, target_value, weight_value = edge
        else:
            raise ValueError("Reduced graph source edges must contain source, target, and weight.")
        if not isinstance(weight_value, (int, float, np.integer, np.floating)) or not np.isfinite(
            weight_value
        ):
            raise ValueError("Reduced graph source edge weights must be finite.")
        source_index = _node_endpoint(source_value, source_body_ids, id_to_source_index)
        target_index = _node_endpoint(target_value, source_body_ids, id_to_source_index)
        source_id = source_body_ids[source_index]
        target_id = source_body_ids[target_index]
        raw_weight = float(weight_value)
        if (
            source_id in selected_set
            and target_id in selected_set
            and raw_weight >= minimum_edge_threshold
        ):
            key = (source_id, target_id)
            aggregated[key] = aggregated.get(key, 0.0) + raw_weight

    output_index = {body_id: index for index, body_id in enumerate(selected)}
    retained = sorted(aggregated.items(), key=lambda item: (output_index[item[0][0]], output_index[item[0][1]]))
    denominators: dict[int, float] = {}
    for (source_id, target_id), contacts in retained:
        source_sign = source_signs[id_to_source_index[source_id]]
        denominators[target_id] = denominators.get(target_id, 0.0) + abs(
            contacts * source_sign
        )
    edge_pairs: list[tuple[int, int]] = []
    modeled_weights: list[float] = []
    for (source_id, target_id), contacts in retained:
        denominator = denominators.get(target_id, 0.0)
        source_sign = source_signs[id_to_source_index[source_id]]
        edge_pairs.append((output_index[source_id], output_index[target_id]))
        modeled_weights.append(
            0.0 if denominator == 0 else contacts * source_sign / denominator
        )

    edge_index = (
        np.asarray(edge_pairs, dtype=np.int64).T
        if edge_pairs
        else np.empty((2, 0), dtype=np.int64)
    )
    return ReducedGraphArtifact(
        dataset_version=_metadata_value(
            metadata, dataset_version, "dataset version", "dataset", "datasetVersion"
        ),
        source_url=_metadata_value(
            metadata, source_url, "source URL", "source", "sourceUrl"
        ),
        license=_metadata_value(metadata, license, "license"),
        source_sha256=computed_source_hash,
        selection_rule=_metadata_value(
            metadata, selection_rule, "selection rule", "selection", "selectionRule"
        ),
        minimum_edge_threshold=float(minimum_edge_threshold),
        body_ids=np.asarray(selected, dtype=np.int64),
        edge_index=edge_index,
        edge_weight=np.asarray(modeled_weights, dtype=np.float32),
        sensory_body_ids=np.asarray(selected_sensory, dtype=np.int64),
        readout_body_ids=np.asarray(selected_readout, dtype=np.int64),
    )


def load_default_reduced_graph() -> ReducedGraphArtifact:
    """Load the pinned 80-cell FlyDino-selected MaleCNS subset against this atlas."""
    graph_payload = json.loads(DEFAULT_GRAPH_PATH.read_text(encoding="utf-8"))
    nodes = graph_payload.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Pinned reduced graph has no node table.")
    ids = np.frombuffer((DEFAULT_ATLAS_PATH / "ids.bin").read_bytes(), dtype="<u4")
    groups = np.frombuffer((DEFAULT_ATLAS_PATH / "groups.bin").read_bytes(), dtype="u1")
    if ids.shape != groups.shape:
        raise ValueError("MaleCNS atlas IDs and groups have incompatible lengths.")
    visible = ids[groups < 3]
    selected = [int(node["id"]) for node in nodes if isinstance(node, Mapping)]
    return build_reduced_graph(
        DEFAULT_GRAPH_PATH,
        selected_ids=selected,
        atlas_visible_ids=visible,
        minimum_edge_threshold=1,
    )
