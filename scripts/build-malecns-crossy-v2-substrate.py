from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.feather as feather
import scipy.sparse as sp


W_MIN = 5.0
HOPS = 4
OUTPUT_SUPERCLASS = "descending_neuron"
VISUAL_PROJECTION_SUPERCLASS = "visual_projection"
EXCLUDE_SUPERCLASS_PREFIX = "vnc_"

VISUAL_INPUT_TYPES = (
    "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
    "Mi1", "Mi4", "Mi9",
    "Tm1", "Tm2", "Tm4", "Tm9", "Tm20",
    "Tm5a", "Tm5b", "Tm5c", "TmY5a",
)

EXPECTED = {
    "neurons_full": 166_700,
    "edges_full": 25_582_938,
    "visual_inputs": 30_906,
    "selected_nodes": 138_968,
    "dynamic_nodes": 108_062,
    "stored_edges": 4_639_332,
    "reachable_descending": 1_312,
}

SOURCES = {
    "annotations": {
        "filename": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "sha256": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    },
    "edges": {
        "filename": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "sha256": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    },
    "neurotransmitters": {
        "filename": "body-neurotransmitters-male-cns-v1.0.feather",
        "sha256": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    },
}

NT_SIGN = {
    "acetylcholine": 1,
    "dopamine": 1,
    "octopamine": 1,
    "serotonin": 1,
    "unclear": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def content_hash(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        h.update(name.encode("utf-8"))
        h.update(str(array.dtype).encode("ascii"))
        h.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        h.update(array.view(np.uint8))
    return h.hexdigest()


def find_repo() -> Path:
    candidates = [
        Path.cwd(),
        Path(r"C:\Users\Nico\Documents\GitHub\fly-crossy-connectome"),
        Path.home() / "Documents" / "GitHub" / "fly-crossy-connectome",
    ]
    for candidate in candidates:
        if (candidate / "python" / "fly_crossy" / "env.py").is_file():
            return candidate.resolve()
    raise SystemExit("No encontre la raiz de fly-crossy-connectome.")


def find_cache(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.append(Path.home() / ".cache" / "fly-crossy-connectome" / "malecns-v1")
    for candidate in candidates:
        if all((candidate / spec["filename"]).is_file() for spec in SOURCES.values()):
            return candidate.resolve()
    raise SystemExit("No encontre los tres Feather verificados de MaleCNS v1.0.")


def map_body_ids(sorted_body: np.ndarray, sort_order: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(sorted_body, values)
    valid = positions < len(sorted_body)
    source_rows = np.flatnonzero(valid)
    valid_positions = positions[valid]
    exact = sorted_body[valid_positions] == values[valid]
    mapped = np.full(len(values), -1, dtype=np.int32)
    mapped[source_rows[exact]] = sort_order[valid_positions[exact]].astype(np.int32)
    return mapped, mapped >= 0


def hop_distances(matrix: sp.csr_matrix, seeds: np.ndarray, hops: int) -> np.ndarray:
    if seeds.dtype != np.bool_ or seeds.shape != (matrix.shape[0],):
        raise ValueError("Expected one boolean seed per annotation object.")
    distance = np.full(matrix.shape[0], -1, dtype=np.int16)
    distance[seeds] = 0
    frontier = seeds.astype(np.float32)
    for hop in range(1, hops + 1):
        reached = ((matrix @ frontier) > 0) & (distance < 0)
        if not reached.any():
            break
        distance[reached] = hop
        frontier = reached.astype(np.float32)
    return distance


def build_csc(node_count: int, post: np.ndarray, pre: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = np.lexsort((post, pre))
    counts = np.bincount(pre, minlength=node_count)
    indptr_pre = np.zeros(node_count + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr_pre[1:])
    indices_post = post[perm].astype(np.int32, copy=False)
    return indptr_pre, indices_post, perm.astype(np.int64, copy=False)


def require_equal(label: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise SystemExit(f"{label}: esperaba {expected:,}, obtuve {actual:,}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen MaleCNS Crossy V2 w>=5 / 4-hop visual-to-DN substrate."
    )
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    root = find_repo()
    cache = find_cache(args.cache)
    out = args.output or root / "artifacts" / "malecns-crossy-v2" / "substrate-w5-h4.npz"
    out = out.resolve()
    manifest_path = out.with_suffix(".manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    print("=== MaleCNS Crossy V2 substrate builder ===", flush=True)
    print(f"repo:   {root}", flush=True)
    print(f"cache:  {cache}", flush=True)
    print(f"output: {out}", flush=True)

    source_hashes: dict[str, str] = {}
    if not args.skip_hash:
        for name, spec in SOURCES.items():
            path = cache / spec["filename"]
            print(f"[hash] {path.name}", flush=True)
            actual = sha256_file(path)
            if actual != spec["sha256"]:
                raise SystemExit(
                    f"SHA-256 incorrecto para {path.name}\n"
                    f"esperado: {spec['sha256']}\n"
                    f"actual:   {actual}"
                )
            source_hashes[name] = actual
    else:
        source_hashes = {name: "skipped" for name in SOURCES}

    t0 = time.time()

    print("[1/7] Cargando anotaciones y aplicando ledger neuronal...", flush=True)
    ann = feather.read_table(
        cache / SOURCES["annotations"]["filename"],
        columns=["bodyId", "type", "superclass", "status", "somaSide"],
    )
    body = ann["bodyId"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    type_name = np.asarray([x or "" for x in ann["type"].to_pylist()], dtype=object)
    superclass = np.asarray([x or "" for x in ann["superclass"].to_pylist()], dtype=object)
    status = np.asarray([x or "" for x in ann["status"].to_pylist()], dtype=object)

    if len(body) != len(np.unique(body)):
        raise SystemExit("Annotations contiene bodyId duplicados.")

    has_superclass = np.fromiter((bool(x) for x in superclass), dtype=bool, count=len(superclass))
    glia = status == "Glia"
    neuron = has_superclass & ~glia
    require_equal("neurons_full", int(neuron.sum()), EXPECTED["neurons_full"])

    visual = neuron & np.isin(type_name, VISUAL_INPUT_TYPES)
    output_dn = neuron & (superclass == OUTPUT_SUPERCLASS)
    vnc = neuron & np.fromiter(
        (x.startswith(EXCLUDE_SUPERCLASS_PREFIX) for x in superclass),
        dtype=bool,
        count=len(superclass),
    )
    excluded = ~neuron | vnc
    require_equal("visual_inputs", int(visual.sum()), EXPECTED["visual_inputs"])

    print(
        f"  annotationObjects={len(body):,} neurons={int(neuron.sum()):,} "
        f"visual={int(visual.sum()):,} descending={int(output_dn.sum()):,} "
        f"vncExcluded={int(vnc.sum()):,}",
        flush=True,
    )

    print("[2/7] Cargando y verificando grafo neuronal...", flush=True)
    edge_table = feather.read_table(
        cache / SOURCES["edges"]["filename"],
        columns=["body_pre", "body_post", "weight"],
    )
    pre_body = edge_table["body_pre"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    post_body = edge_table["body_post"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    contacts_all = edge_table["weight"].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

    sort_order = np.argsort(body)
    sorted_body = body[sort_order]
    pre_global, ok_pre = map_body_ids(sorted_body, sort_order, pre_body)
    post_global, ok_post = map_body_ids(sorted_body, sort_order, post_body)
    mapped = ok_pre & ok_post & (contacts_all > 0)
    pre_global = pre_global[mapped]
    post_global = post_global[mapped]
    contacts = contacts_all[mapped]
    neuronal_edge = neuron[pre_global] & neuron[post_global]
    pre_global = pre_global[neuronal_edge]
    post_global = post_global[neuronal_edge]
    contacts = contacts[neuronal_edge]
    require_equal("edges_full", len(contacts), EXPECTED["edges_full"])

    del edge_table, pre_body, post_body, contacts_all

    print("[3/7] Seleccionando rutas visual -> descending (w>=5, hops=4)...", flush=True)
    keep = (
        (contacts >= W_MIN)
        & ~visual[post_global]
        & ~excluded[pre_global]
        & ~excluded[post_global]
    )
    binary_data = np.ones(int(keep.sum()), dtype=np.float32)
    forward = sp.csr_matrix(
        (binary_data, (post_global[keep], pre_global[keep])),
        shape=(len(body), len(body)),
    )
    reverse = sp.csr_matrix(
        (binary_data, (pre_global[keep], post_global[keep])),
        shape=(len(body), len(body)),
    )

    fwd = hop_distances(forward, visual, HOPS)
    bwd = hop_distances(reverse, output_dn, HOPS)
    core = (fwd >= 0) & (bwd >= 0) & ~visual & ~excluded
    selected = visual | core
    reachable_dn = output_dn & selected

    require_equal("dynamic_nodes", int(core.sum()), EXPECTED["dynamic_nodes"])
    require_equal("selected_nodes", int(selected.sum()), EXPECTED["selected_nodes"])
    require_equal(
        "reachable_descending",
        int(reachable_dn.sum()),
        EXPECTED["reachable_descending"],
    )

    stored = keep & selected[pre_global] & core[post_global]
    require_equal("stored_edges", int(stored.sum()), EXPECTED["stored_edges"])

    print(
        f"  selected={int(selected.sum()):,} dynamic={int(core.sum()):,} "
        f"DN={int(reachable_dn.sum()):,}/{int(output_dn.sum()):,} "
        f"edges={int(stored.sum()):,}",
        flush=True,
    )

    del forward, reverse, binary_data

    print("[4/7] Construyendo indices locales deterministas...", flush=True)
    selected_global = np.flatnonzero(selected)
    # Canonical node order is ascending MaleCNS bodyId, not annotation row order.
    selected_global = selected_global[np.argsort(body[selected_global], kind="stable")]
    local_of_global = np.full(len(body), -1, dtype=np.int32)
    local_of_global[selected_global] = np.arange(len(selected_global), dtype=np.int32)

    pre = local_of_global[pre_global[stored]]
    post = local_of_global[post_global[stored]]
    edge_weight = contacts[stored].astype(np.float32, copy=False)

    order = np.lexsort((pre, post))
    pre = pre[order].astype(np.int32, copy=False)
    post = post[order].astype(np.int32, copy=False)
    edge_weight = edge_weight[order]

    # The released weight table should already contain one aggregate row per body pair.
    if len(pre) > 1:
        duplicate = (pre[1:] == pre[:-1]) & (post[1:] == post[:-1])
        if duplicate.any():
            raise SystemExit(
                f"Encontre {int(duplicate.sum()):,} pares de edges duplicados; "
                "no los agrego silenciosamente."
            )

    counts_post = np.bincount(post, minlength=len(selected_global))
    indptr_post = np.zeros(len(selected_global) + 1, dtype=np.int64)
    np.cumsum(counts_post, out=indptr_post[1:])
    indptr_pre, indices_post, perm_csr_to_csc = build_csc(
        len(selected_global), post, pre
    )

    print("[5/7] Asignando signs, tipos y poblaciones de readout...", flush=True)
    nt = feather.read_table(
        cache / SOURCES["neurotransmitters"]["filename"],
        columns=["body", "consensus_nt"],
    )
    nt_body = nt["body"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    nt_name = np.asarray([x if x is not None else None for x in nt["consensus_nt"].to_pylist()], dtype=object)
    nt_order = np.argsort(nt_body)
    nt_body_sorted = nt_body[nt_order]
    nt_name_sorted = nt_name[nt_order]

    selected_body = body[selected_global].astype(np.int64, copy=False)
    pos = np.searchsorted(nt_body_sorted, selected_body)
    has_nt = pos < len(nt_body_sorted)
    exact = np.zeros(len(selected_body), dtype=bool)
    exact[has_nt] = nt_body_sorted[pos[has_nt]] == selected_body[has_nt]

    selected_nt = np.empty(len(selected_body), dtype=object)
    selected_nt[:] = None
    selected_nt[exact] = nt_name_sorted[pos[exact]]

    sign = np.empty(len(selected_body), dtype=np.int8)
    for i, name in enumerate(selected_nt):
        if name is None:
            sign[i] = 1
        elif name in NT_SIGN:
            sign[i] = NT_SIGN[name]
        else:
            raise SystemExit(f"Neurotransmisor desconocido: {name!r}")

    selected_type = type_name[selected_global]
    vocabulary = sorted({str(x) for x in selected_type if str(x)})
    type_to_id = {name: i for i, name in enumerate(vocabulary)}
    type_id = np.asarray(
        [type_to_id.get(str(name), -1) for name in selected_type],
        dtype=np.int32,
    )

    is_clamped = visual[selected_global].astype(np.bool_)
    is_dynamic = core[selected_global].astype(np.bool_)
    is_output = reachable_dn[selected_global].astype(np.bool_)
    is_visual_projection = (
        neuron[selected_global]
        & (superclass[selected_global] == VISUAL_PROJECTION_SUPERCLASS)
    ).astype(np.bool_)

    arrays = {
        "bodyId": selected_body,
        "node_idx": selected_global.astype(np.int32),
        "type_id": type_id,
        "sign": sign,
        "is_clamped": is_clamped,
        "is_dynamic": is_dynamic,
        "is_output": is_output,
        "is_visual_projection": is_visual_projection,
        "indptr_post": indptr_post,
        "indices_pre": pre,
        "weight": edge_weight,
        "indptr_pre": indptr_pre,
        "indices_post": indices_post,
        "perm_csr_to_csc": perm_csr_to_csc,
        "forward_hops": fwd[selected_global].astype(np.int8),
        "backward_hops": bwd[selected_global].astype(np.int8),
    }

    print(
        f"  visualProjection={int(is_visual_projection.sum()):,} "
        f"excitatory={int((sign > 0).sum()):,} inhibitory={int((sign < 0).sum()):,}",
        flush=True,
    )

    print("[6/7] Escribiendo substrate NPZ...", flush=True)
    artifact_content_sha = content_hash(arrays)
    np.savez(out, **arrays)

    manifest = {
        "version": "imas-malecns-crossy-v2-substrate-1",
        "dataset": "FlyEM MaleCNS v1.0, min confidence 0.5",
        "source": "https://male-cns.janelia.org/download/",
        "license": "CC BY 4.0",
        "sourceHashes": source_hashes,
        "selection": {
            "visualInputTypes": list(VISUAL_INPUT_TYPES),
            "outputSuperclass": OUTPUT_SUPERCLASS,
            "excludeSuperclassPrefix": EXCLUDE_SUPERCLASS_PREFIX,
            "minimumSynapses": W_MIN,
            "hops": HOPS,
            "rule": (
                "visual cells are clamped; discard edges into clamped cells; "
                "dynamic core is intersection of <=4-hop forward reachability from "
                "visual inputs and <=4-hop backward reachability from descending neurons; "
                "VNC, unresolved annotation objects and glia are excluded"
            ),
        },
        "counts": {
            "fullNeurons": int(neuron.sum()),
            "fullWeightedEdges": len(contacts),
            "selectedNodes": len(selected_global),
            "clampedVisual": int(is_clamped.sum()),
            "dynamicNodes": int(is_dynamic.sum()),
            "reachableDescending": int(is_output.sum()),
            "visualProjection": int(is_visual_projection.sum()),
            "storedEdges": len(pre),
        },
        "typeVocabulary": vocabulary,
        "artifactContentSha256": artifact_content_sha,
        "artifactFileSha256": sha256_file(out),
        "elapsedSeconds": time.time() - t0,
        "notes": [
            "Raw synapse contact counts are preserved. No incoming normalization is applied.",
            "Presynaptic signs are derived from MaleCNS consensus neurotransmitter annotations.",
            "No FlyDino core, Crossy observation vector, learned connectome weights or gameplay labels are used.",
            "This file is generated data and should not be committed unless the repository explicitly tracks large artifacts.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("[7/7] Verificacion final...", flush=True)
    with np.load(out, allow_pickle=False) as z:
        check_arrays = {name: z[name] for name in arrays}
    check_hash = content_hash(check_arrays)
    if check_hash != artifact_content_sha:
        raise SystemExit("El content hash cambio al releer el artifact.")

    print()
    print("SUBSTRATE V2: OK", flush=True)
    print(f"artifact: {out}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    print(f"content sha256: {artifact_content_sha}", flush=True)
    print(f"file sha256:    {manifest['artifactFileSha256']}", flush=True)
    print(
        f"counts: {len(selected_global):,} nodes / {len(pre):,} edges / "
        f"{int(is_clamped.sum()):,} visual / {int(is_output.sum()):,} DN / "
        f"{int(is_visual_projection.sum()):,} VPN",
        flush=True,
    )
    print("\nNo se entreno nada y no se modifico la arquitectura vieja.", flush=True)


if __name__ == "__main__":
    main()
