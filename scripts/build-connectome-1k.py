from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

import numpy as np
import pyarrow.feather as feather


TARGET_NODES = 1000
CORE_NODES = 80
ADDITIONAL_NODES = TARGET_NODES - CORE_NODES

DATASET = "FlyEM MaleCNS v1.0, min confidence 0.5"
LICENSE = "CC BY 4.0"
SOURCE = "https://male-cns.janelia.org/download/"

SOURCES = {
    "annotations": {
        "filename": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "sha256": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    },
    "edges": {
        "filename": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "sha256": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    },
    "neurotransmitters": {
        "filename": "body-neurotransmitters-male-cns-v1.0.feather",
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather",
        "sha256": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    },
}

SIGNS = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_repo() -> Path:
    candidates = [
        Path.cwd(),
        Path(r"C:\Users\Nico\Documents\GitHub\fly-crossy-connectome"),
        Path.home() / "Documents" / "GitHub" / "fly-crossy-connectome",
    ]
    for candidate in candidates:
        if (
            (candidate / "python/fly_crossy/connectome.py").is_file()
            and (candidate / "public/data/connectome/graph.json").is_file()
            and (candidate / "public/data/brain-atlas/ids.bin").is_file()
        ):
            return candidate.resolve()
    raise SystemExit("No encontre fly-crossy-connectome.")


def download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        current = sha256(destination)
        if current == expected_sha256:
            print(f"[cache] {destination.name}")
            return
        print(f"[cache invalida] {destination.name}: {current}")
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)

    print(f"[download] {destination.name}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "fly-crossy-connectome/1k-builder"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None
        copied = 0
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            copied += len(chunk)
            if total_bytes:
                print(
                    f"  {copied / (1024**2):8.1f} / {total_bytes / (1024**2):8.1f} MiB",
                    end="\r",
                    flush=True,
                )
    if total_bytes:
        print()

    actual = sha256(partial)
    if actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise SystemExit(
            f"SHA-256 incorrecto para {destination.name}:\n"
            f"esperado {expected_sha256}\n"
            f"obtenido {actual}"
        )
    partial.replace(destination)


def summed_strength(ids: np.ndarray, weights: np.ndarray) -> dict[int, int]:
    if ids.size == 0:
        return {}
    unique, inverse = np.unique(ids, return_inverse=True)
    sums = np.bincount(inverse, weights=weights.astype(np.float64, copy=False))
    return {int(body_id): int(round(value)) for body_id, value in zip(unique, sums, strict=True)}


ROOT = find_repo()
CORE_GRAPH_PATH = ROOT / "public" / "data" / "connectome" / "graph.json"
CORE_MANIFEST_PATH = ROOT / "public" / "data" / "connectome" / "manifest.json"
ATLAS_DIR = ROOT / "public" / "data" / "brain-atlas"
OUTPUT_DIR = ROOT / "public" / "data" / "connectome-1k"
SCRIPT_DESTINATION = ROOT / "scripts" / "build-connectome-1k.py"

cache_dir = Path.home() / ".cache" / "fly-crossy-connectome" / "malecns-v1"

print("=== MaleCNS 1k builder ===")
print(f"repo:  {ROOT}")
print(f"cache: {cache_dir}")
print()

for name, spec in SOURCES.items():
    download_verified(
        spec["url"],
        cache_dir / spec["filename"],
        spec["sha256"],
    )

print("\n[1/6] Verificando nucleo historico de 80 neuronas...")
core_graph = json.loads(CORE_GRAPH_PATH.read_text(encoding="utf-8"))
core_manifest = json.loads(CORE_MANIFEST_PATH.read_text(encoding="utf-8"))

if len(core_graph.get("nodes", [])) != CORE_NODES:
    raise SystemExit(
        f"El grafo historico ya no tiene {CORE_NODES} nodos."
    )
if sha256(CORE_GRAPH_PATH) != core_manifest.get("graphSha256"):
    raise SystemExit("graph.json 80n no coincide con su manifest.")

core_nodes = core_graph["nodes"]
core_ids = [int(node["id"]) for node in core_nodes]
if len(core_ids) != len(set(core_ids)):
    raise SystemExit("El nucleo 80n contiene body IDs duplicados.")

core_input_ids = {
    core_ids[int(entry[0])]
    for entry in core_graph.get("inputs", [])
}
core_output_ids = {
    core_ids[int(index)]
    for index in core_graph.get("outputs", [])
}
if len(core_input_ids) != 32 or len(core_output_ids) != 16:
    raise SystemExit(
        f"Esperaba 32 inputs y 16 outputs; encontre "
        f"{len(core_input_ids)} y {len(core_output_ids)}."
    )

print("[2/6] Leyendo atlas y tablas MaleCNS verificadas...")
atlas_ids = np.frombuffer((ATLAS_DIR / "ids.bin").read_bytes(), dtype="<u4").astype(np.int64)
atlas_groups = np.frombuffer((ATLAS_DIR / "groups.bin").read_bytes(), dtype="u1")
if atlas_ids.shape != atlas_groups.shape:
    raise SystemExit("ids.bin y groups.bin tienen tamanos incompatibles.")
visible_ids = set(int(value) for value in atlas_ids[atlas_groups < 3])

annotation_path = cache_dir / SOURCES["annotations"]["filename"]
edge_path = cache_dir / SOURCES["edges"]["filename"]
nt_path = cache_dir / SOURCES["neurotransmitters"]["filename"]

annotation_rows = feather.read_table(
    annotation_path,
    columns=["bodyId", "type", "superclass", "somaLocation"],
).to_pylist()
annotations = {
    int(row["bodyId"]): row
    for row in annotation_rows
    if row.get("somaLocation") is not None
}

nt_rows = feather.read_table(
    nt_path,
    columns=["body", "consensus_nt"],
).to_pylist()
neurotransmitters = {
    int(row["body"]): row.get("consensus_nt")
    for row in nt_rows
}

edge_table = feather.read_table(
    edge_path,
    columns=["body_pre", "body_post", "weight"],
)
pre = edge_table["body_pre"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
post = edge_table["body_post"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
weight = edge_table["weight"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

if not set(core_ids).issubset(visible_ids):
    raise SystemExit("Alguna de las 80 neuronas historicas no es atlas-visible.")
if not set(core_ids).issubset(annotations):
    raise SystemExit("Alguna de las 80 neuronas historicas falta en annotations.")

print("[3/6] Seleccionando 920 vecinos anatomicos directos...")
core_array = np.asarray(core_ids, dtype=np.int64)
pre_is_core = np.isin(pre, core_array)
post_is_core = np.isin(post, core_array)

core_to_candidate = summed_strength(
    post[pre_is_core & ~post_is_core],
    weight[pre_is_core & ~post_is_core],
)
candidate_to_core = summed_strength(
    pre[post_is_core & ~pre_is_core],
    weight[post_is_core & ~pre_is_core],
)

eligible = (visible_ids & set(annotations)) - set(core_ids)
direct_neighbors = (
    set(core_to_candidate) | set(candidate_to_core)
) & eligible

if len(direct_neighbors) < ADDITIONAL_NODES:
    raise SystemExit(
        f"Solo encontre {len(direct_neighbors)} vecinos directos atlas-validos; "
        f"necesito {ADDITIONAL_NODES}. No aplico un fallback silencioso."
    )

def candidate_key(body_id: int) -> tuple[int, int, int, int]:
    outward = core_to_candidate.get(body_id, 0)
    inward = candidate_to_core.get(body_id, 0)
    bidirectional_bottleneck = min(outward, inward)
    total = outward + inward
    strongest_direction = max(outward, inward)
    return (
        -bidirectional_bottleneck,
        -total,
        -strongest_direction,
        body_id,
    )

ranked_candidates = sorted(direct_neighbors, key=candidate_key)
added_ids = ranked_candidates[:ADDITIONAL_NODES]
selected_ids = sorted(set(core_ids) | set(added_ids))

if len(selected_ids) != TARGET_NODES:
    raise SystemExit(f"Seleccion inesperada: {len(selected_ids)} nodos.")

bidirectional_count = sum(
    int(core_to_candidate.get(body_id, 0) > 0 and candidate_to_core.get(body_id, 0) > 0)
    for body_id in added_ids
)

print(f"  candidatos directos elegibles: {len(direct_neighbors)}")
print(f"  agregados:                    {len(added_ids)}")
print(f"  agregados bidireccionales:    {bidirectional_count}")

print("[4/6] Extrayendo todas las conexiones internas medidas...")
selected_set = set(selected_ids)
selected_array = np.asarray(selected_ids, dtype=np.int64)
internal_mask = np.isin(pre, selected_array) & np.isin(post, selected_array)
internal_pre = pre[internal_mask]
internal_post = post[internal_mask]
internal_weight = weight[internal_mask]

index_by_id = {body_id: index for index, body_id in enumerate(selected_ids)}
aggregated_edges: dict[tuple[int, int], int] = defaultdict(int)
for source_id, target_id, contacts in zip(
    internal_pre, internal_post, internal_weight, strict=True
):
    aggregated_edges[
        (index_by_id[int(source_id)], index_by_id[int(target_id)])
    ] += int(contacts)

edges = [
    [source, target, contacts]
    for (source, target), contacts in sorted(aggregated_edges.items())
    if contacts > 0
]
if not edges:
    raise SystemExit("El subgrafo 1k quedo sin edges.")

core_node_by_id = {int(node["id"]): dict(node) for node in core_nodes}
nodes: list[dict[str, object]] = []

for body_id in selected_ids:
    if body_id in core_node_by_id:
        node = dict(core_node_by_id[body_id])
        node["role"] = (
            "input"
            if body_id in core_input_ids
            else "output"
            if body_id in core_output_ids
            else "interneuron"
        )
        nodes.append(node)
        continue

    row = annotations[body_id]
    nt = neurotransmitters.get(body_id)
    nodes.append(
        {
            "id": body_id,
            "type": row.get("type"),
            "position": row.get("somaLocation"),
            "nt": nt,
            "sign": SIGNS.get(nt, 0),
            # Deliberadamente no creamos nuevas interfaces artificiales.
            "role": "interneuron",
        }
    )

old_index_to_id = {
    index: int(node["id"])
    for index, node in enumerate(core_nodes)
}
inputs = sorted(
    [
        [index_by_id[old_index_to_id[int(old_index)]], int(channel)]
        for old_index, channel in core_graph.get("inputs", [])
    ],
    key=lambda item: (item[1], item[0]),
)
outputs = [
    index_by_id[old_index_to_id[int(old_index)]]
    for old_index in core_graph.get("outputs", [])
]

graph = {
    "version": "malecns-crossy-capacity-1k-v1",
    "nodes": nodes,
    "edges": edges,
    "inputs": inputs,
    "outputs": outputs,
    "channels": core_graph.get("channels", []),
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
graph_path = OUTPUT_DIR / "graph.json"
graph_path.write_text(
    json.dumps(graph, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
)
graph_hash = sha256(graph_path)

selection_rule = (
    "Capacity expansion from the frozen 80-cell FlyDino-derived core. "
    "Retain all 80 core body IDs and their 32 engineered input cells plus 16 "
    "engineered readout cells. Add 920 atlas-visible MaleCNS v1.0 cells with "
    "soma annotations, ranked only by measured direct synaptic connectivity "
    "to the 80-cell core: descending order of min(core->candidate contacts, "
    "candidate->core contacts), then total bidirectional contacts, then strongest "
    "single direction, then ascending body ID. No Fly Crossy rewards, scores, "
    "training seeds, evaluation seeds, or game outcomes are used. Retain every "
    "measured directed internal edge among the selected 1000 cells; aggregate "
    "duplicate source-target rows by summed contact count."
)

manifest = {
    "dataset": DATASET,
    "license": LICENSE,
    "source": SOURCE,
    "nodes": TARGET_NODES,
    "edges": len(edges),
    "synapticContacts": int(sum(edge[2] for edge in edges)),
    "inputCells": len(inputs),
    "readoutCells": len(outputs),
    "coreCells": CORE_NODES,
    "addedCells": ADDITIONAL_NODES,
    "directNeighborCandidates": len(direct_neighbors),
    "addedBidirectionalWithCore": bidirectional_count,
    "graphSha256": graph_hash,
    "coreGraph": {
        "path": "../connectome/graph.json",
        "sha256": sha256(CORE_GRAPH_PATH),
    },
    "sources": {
        key: spec["sha256"]
        for key, spec in SOURCES.items()
    },
    "sourceUrls": {
        key: spec["url"]
        for key, spec in SOURCES.items()
    },
    "selection": selection_rule,
    "assumptions": (
        "The original 32 engineered sensory interfaces and 16 engineered readout "
        "interfaces are preserved exactly from the 80-cell baseline. Added cells "
        "receive no new engineered game-specific role. Acetylcholine is modeled "
        "as +1; GABA/glutamate as -1; unknown/modulatory transmitters as 0. "
        "Runtime dynamics remain simplified signed incoming-normalized leaky tanh "
        "rate units and are not claimed to be physiological."
    ),
    "scopeWarning": (
        "This is a deterministic 1000-cell capacity experiment around the historical "
        "80-cell core, not a representative whole-brain MaleCNS model."
    ),
}

manifest_path = OUTPUT_DIR / "manifest.json"
manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
)

notice = f"""# MaleCNS v1.0 1000-cell capacity subset

Data creators: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology and Google Research.
Dataset/project: {SOURCE}

License: Creative Commons Attribution 4.0 International.
No endorsement of this experiment is implied.

This artifact is derived from the verified MaleCNS v1.0 minimum-confidence-0.5
annotation and directed connectivity tables and the v1.0 neurotransmitter table.
Exact source URLs and SHA-256 values are recorded in `manifest.json`.

The historical 80-cell circuit is retained in full. The additional 920 cells are
selected deterministically using measured anatomical connectivity to that frozen
core only. Fly Crossy rewards, scores, policies, training seeds, evaluation seeds
and game outcomes are not used for selection.

All measured directed internal source-target connections among the selected 1000
cells are retained, with duplicate rows aggregated by contact count. The original
32 engineered input interfaces and 16 engineered readout interfaces are preserved;
new cells receive no game-specific input/output role.

The runtime's signed normalization and recurrent leaky-tanh dynamics are modeling
assumptions, not biological measurements. This 1000-cell circuit is a bounded
capacity experiment, not a complete or representative fly brain.
"""
(OUTPUT_DIR / "NOTICE.md").write_text(notice, encoding="utf-8")

print("[5/6] Validando el artefacto con el loader real del proyecto...")
sys.path.insert(0, str(ROOT / "python"))
from fly_crossy.connectome import build_reduced_graph  # noqa: E402

validated = build_reduced_graph(
    graph_path,
    selected_ids=selected_ids,
    atlas_visible_ids=visible_ids,
    minimum_edge_threshold=1,
)

assert validated.node_count == TARGET_NODES
assert set(core_ids).issubset(set(int(value) for value in validated.body_ids))
assert len(validated.sensory_body_ids) == 32
assert len(validated.readout_body_ids) == 16
assert set(int(value) for value in validated.sensory_body_ids) == core_input_ids
assert set(int(value) for value in validated.readout_body_ids) == core_output_ids
assert set(selected_ids).issubset(visible_ids)
assert sha256(graph_path) == json.loads(
    manifest_path.read_text(encoding="utf-8")
)["graphSha256"]

print("[6/6] Guardando el builder reproducible dentro del repo...")
SCRIPT_DESTINATION.parent.mkdir(parents=True, exist_ok=True)
source_script = Path(__file__).resolve()
if source_script != SCRIPT_DESTINATION.resolve():
    if SCRIPT_DESTINATION.exists():
        existing = SCRIPT_DESTINATION.read_text(encoding="utf-8")
        current = source_script.read_text(encoding="utf-8")
        if existing != current:
            raise SystemExit(
                f"Ya existe un builder distinto en {SCRIPT_DESTINATION}; no lo sobrescribo."
            )
    else:
        shutil.copy2(source_script, SCRIPT_DESTINATION)

print()
print("=== CONNECTOME 1K CONSTRUIDO Y VALIDADO ===")
print(f"nodes:                 {validated.node_count}")
print(f"raw internal edges:    {len(edges)}")
print(f"normalized edges:      {validated.edge_count}")
print(f"core retained:         {len(set(core_ids) & set(selected_ids))}/80")
print(f"input population:      {len(validated.sensory_body_ids)}")
print(f"readout population:    {len(validated.readout_body_ids)}")
print(f"atlas valid:           {set(selected_ids).issubset(visible_ids)}")
print(f"graph sha256:          {graph_hash}")
print(f"graph:                 {graph_path}")
print(f"manifest:              {manifest_path}")
print(f"builder:               {SCRIPT_DESTINATION}")
print()
print("NO entrenes todavia.")
print("Primero:")
print(f'git -C "{ROOT}" status --short')
