from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.feather as feather

from .substrate import MaleCNSV2Substrate
from .vision_geometry import (
    CLAMPED_TYPES,
    E1,
    E2,
    FRAME_H,
    FRAME_W,
    HEX_TYPES,
    HFOV_DEG,
    LATTICE_STEPS,
    NBR_NAMES,
    OVERLAP_DEG,
    RGB_DOWNSAMPLE,
    SIDES,
    SPACING_DEG,
    FWHM_DEG,
    TRUNC_SIGMA,
    VisualGeometry,
    build_sampler,
    hex_to_raw,
)


ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"


def find_repo() -> Path:
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
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
        if (candidate / ANNOTATIONS).is_file() and (candidate / WEIGHTS).is_file():
            return candidate.resolve()
    raise SystemExit("No encontre los Feather de MaleCNS v1.0.")


def _strongest_partner(key: np.ndarray, partner: np.ndarray, weight: np.ndarray) -> dict[int, int]:
    if len(key) == 0:
        return {}
    pair = np.stack([key, partner], axis=1)
    unique_pair, inverse = np.unique(pair, axis=0, return_inverse=True)
    total = np.bincount(inverse, weights=weight, minlength=len(unique_pair))
    order = np.lexsort((unique_pair[:, 1], -total, unique_pair[:, 0]))
    pair_sorted = unique_pair[order]
    first = np.r_[True, pair_sorted[1:, 0] != pair_sorted[:-1, 0]]
    return {int(k): int(v) for k, v in pair_sorted[first].tolist()}


def _mean_offset(post, pre_type, pre_col, weight, raw, tip, base):
    rows = []
    for body in np.unique(post):
        mask = post == body
        tip_mask = mask & (pre_type == tip)
        base_mask = mask & np.isin(pre_type, base)
        if not tip_mask.any() or not base_mask.any():
            continue
        tip_centroid = (
            raw[pre_col[tip_mask]] * weight[tip_mask, None]
        ).sum(0) / weight[tip_mask].sum()
        base_centroid = (
            raw[pre_col[base_mask]] * weight[base_mask, None]
        ).sum(0) / weight[base_mask].sum()
        rows.append(tip_centroid - base_centroid)
    return None if not rows else np.mean(np.stack(rows), axis=0)


def _infer_frame(evidence: dict[str, np.ndarray | None]) -> tuple[np.ndarray, np.ndarray, dict]:
    if any(evidence.get("T4" + s) is None for s in "abcd"):
        raise RuntimeError("No hay evidencia T4 completa para inferir el frame visual.")

    da, db, dc, dd = (np.asarray(evidence["T4" + s]) for s in "abcd")
    f_raw = (da - db) / 2.0
    u_raw = (dd - dc) / 2.0

    f_norm = np.linalg.norm(f_raw)
    u_norm = np.linalg.norm(u_raw)
    if f_norm <= 1e-12 or u_norm <= 1e-12:
        raise RuntimeError("El frame visual T4 es degenerado.")

    f_unit = f_raw / f_norm
    u_unit = u_raw / u_norm
    steps = hex_to_raw(LATTICE_STEPS)
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)

    k = int(np.argmax(steps @ f_unit))
    front = steps[k]
    up = np.array([-front[1], front[0]], dtype=np.float64)
    if up @ u_unit < 0:
        up = -up

    snap = float(np.degrees(np.arccos(np.clip(front @ f_unit, -1.0, 1.0))))
    orth = float(f_unit @ u_unit)

    if snap > 20.0 or abs(orth) > 0.5:
        raise RuntimeError(
            f"Frame visual no confiable: snap={snap:.3f}deg orth={orth:.3f}"
        )

    diagnostics = {
        "snapDegrees": snap,
        "frontUpCos": orth,
        "front": front.tolist(),
        "up": up.tolist(),
    }

    if all(evidence.get("T5" + s) is not None for s in "abcd"):
        a5, b5, c5, d5 = (np.asarray(evidence["T5" + s]) for s in "abcd")
        f5 = (a5 - b5) / 2.0
        u5 = (d5 - c5) / 2.0
        diagnostics["t5FrontCos"] = float(
            f5 @ front / max(np.linalg.norm(f5), 1e-12)
        )
        diagnostics["t5UpCos"] = float(
            u5 @ up / max(np.linalg.norm(u5), 1e-12)
        )

    return front, up, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen measured visual geometry for MaleCNS Crossy V2."
    )
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--substrate", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    repo = find_repo()
    cache = find_cache(args.cache)
    substrate_path = args.substrate or repo / "artifacts" / "malecns-crossy-v2" / "substrate-w5-h4.npz"
    output = args.output or repo / "artifacts" / "malecns-crossy-v2" / "visual-geometry-160x120.npz"

    substrate = MaleCNSV2Substrate.load(substrate_path, strict=True)
    expected_visual_body = substrate.body_ids[substrate.is_clamped]

    started = time.time()
    print("=== MaleCNS Crossy V2 / Build Visual Geometry ===", flush=True)
    print(f"substrate: {substrate_path.resolve()}", flush=True)
    print(f"output:    {output.resolve()}", flush=True)

    print("[1/6] Leyendo anotaciones...", flush=True)
    table = feather.read_table(
        cache / ANNOTATIONS,
        columns=["bodyId", "type", "somaSide", "assignedOlHex1", "assignedOlHex2"],
    )

    body = table["bodyId"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    typ = np.asarray([x or "" for x in table["type"].to_pylist()], dtype=object)
    side_name = np.asarray([x or "" for x in table["somaSide"].to_pylist()], dtype=object)
    h1 = np.asarray(table["assignedOlHex1"].to_pylist(), dtype=object)
    h2 = np.asarray(table["assignedOlHex2"].to_pylist(), dtype=object)

    relevant = np.isin(typ, tuple(set(CLAMPED_TYPES) | set(HEX_TYPES))) & np.isin(side_name, SIDES)
    body, typ, side_name, h1, h2 = (
        body[relevant], typ[relevant], side_name[relevant], h1[relevant], h2[relevant]
    )
    side = (side_name == "R").astype(np.int8)
    has_hex = np.fromiter(
        (a is not None and b is not None for a, b in zip(h1, h2, strict=True)),
        dtype=bool,
        count=len(body),
    )

    hex_rows = np.flatnonzero(has_hex)
    column_key = np.stack(
        [
            side[hex_rows],
            np.asarray([int(h1[i]) for i in hex_rows]),
            np.asarray([int(h2[i]) for i in hex_rows]),
        ],
        axis=1,
    )
    unique_columns, inverse = np.unique(column_key, axis=0, return_inverse=True)
    col_side = unique_columns[:, 0].astype(np.int8)
    col_hex = unique_columns[:, 1:].astype(np.int16)

    if len(unique_columns) != 1771:
        raise SystemExit(f"Esperaba 1,771 columnas, obtuve {len(unique_columns):,}.")

    hex_body = body[hex_rows]
    hex_col = inverse.astype(np.int32)
    hex_type = typ[hex_rows]

    visual_mask = np.isin(typ, CLAMPED_TYPES)
    vbody = body[visual_mask]
    vtype = typ[visual_mask]
    vside = side[visual_mask]
    vhas_hex = has_hex[visual_mask]
    vh1 = h1[visual_mask]
    vh2 = h2[visual_mask]

    if len(vbody) != 30_906:
        raise SystemExit(f"Esperaba 30,906 visual cells, obtuve {len(vbody):,}.")
    if set(vbody.tolist()) != set(expected_visual_body.tolist()):
        raise SystemExit("El set de visual body IDs no coincide con el substrate congelado.")

    print("[2/6] Leyendo wiring para ubicar cells sin hex...", flush=True)
    wt = feather.read_table(cache / WEIGHTS, columns=["body_pre", "body_post", "weight"])
    pre = wt["body_pre"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    post = wt["body_post"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    weight = wt["weight"].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

    need = vbody[~vhas_hex]
    in_hex = np.unique(hex_body)
    in_need = np.unique(need)

    edge_in = np.isin(pre, in_hex) & np.isin(post, in_need)
    edge_out = np.isin(pre, in_need) & np.isin(post, in_hex)
    edges_in = (pre[edge_in], post[edge_in], weight[edge_in])
    edges_out = (pre[edge_out], post[edge_out], weight[edge_out])

    body_to_col = {int(b): int(c) for b, c in zip(hex_body, hex_col, strict=True)}
    body_to_type = {int(b): str(t) for b, t in zip(hex_body, hex_type, strict=True)}
    key_to_col = {
        (int(s), int(a), int(b)): i
        for i, (s, a, b) in enumerate(unique_columns.tolist())
    }

    vcol = np.full(len(vbody), -1, dtype=np.int32)
    assign_kind = np.full(len(vbody), 3, dtype=np.int8)

    for i in np.flatnonzero(vhas_hex):
        vcol[i] = key_to_col[(int(vside[i]), int(vh1[i]), int(vh2[i]))]
        assign_kind[i] = 0

    best_pre = _strongest_partner(edges_in[1], edges_in[0], edges_in[2])
    best_post = _strongest_partner(edges_out[0], edges_out[1], edges_out[2])

    for i in np.flatnonzero(~vhas_hex):
        b = int(vbody[i])
        if b in best_pre and best_pre[b] in body_to_col:
            vcol[i] = body_to_col[best_pre[b]]
            assign_kind[i] = 1
        elif b in best_post and best_post[b] in body_to_col:
            vcol[i] = body_to_col[best_post[b]]
            assign_kind[i] = 2

    assigned = vcol >= 0
    wrong_side = assigned & (col_side[np.maximum(vcol, 0)] != vside)
    vcol[wrong_side] = -1
    assign_kind[wrong_side] = 3

    if int((vcol < 0).sum()) != 5:
        raise SystemExit(f"Esperaba 5 cells visuales sin columna, obtuve {int((vcol < 0).sum())}.")

    print("[3/6] Infiriendo orientacion desde T4/T5...", flush=True)
    pre_e, post_e, w_e = edges_in
    pre_col = np.asarray([body_to_col.get(int(b), -1) for b in pre_e], dtype=np.int32)
    ok = pre_col >= 0
    pre_e, post_e, w_e, pre_col = pre_e[ok], post_e[ok], w_e[ok], pre_col[ok]
    pre_type = np.asarray([body_to_type.get(int(b), "") for b in pre_e], dtype=object)

    visual_type_by_body = {int(b): str(t) for b, t in zip(vbody, vtype, strict=True)}
    visual_side_by_body = {int(b): int(s) for b, s in zip(vbody, vside, strict=True)}
    post_type = np.asarray([visual_type_by_body.get(int(b), "") for b in post_e], dtype=object)
    post_side = np.asarray([visual_side_by_body.get(int(b), -1) for b in post_e], dtype=np.int8)
    raw = hex_to_raw(col_hex)

    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    frame_meta = {}
    for si in (0, 1):
        evidence = {}
        groups = {
            "T4": ("Mi9", ("Mi4", "C3")),
            "T5": ("Tm9", ("Tm1", "Tm2", "Tm4")),
        }
        for family, (tip, base) in groups.items():
            for sub in "abcd":
                cell_name = family + sub
                m = (post_side == si) & (post_type == cell_name)
                evidence[cell_name] = _mean_offset(
                    post_e[m], pre_type[m], pre_col[m], w_e[m], raw, tip, base
                )

        front, up, diagnostics = _infer_frame(evidence)
        frames[si] = (front, up)
        frame_meta[SIDES[si]] = diagnostics
        print(
            f"  {SIDES[si]}: front={np.round(front, 4).tolist()} "
            f"up={np.round(up, 4).tolist()} snap={diagnostics['snapDegrees']:.3f}",
            flush=True,
        )

    print("[4/6] Construyendo columnas angulares y vecindad...", flush=True)
    col_az = np.zeros(len(col_side), dtype=np.float32)
    col_el = np.zeros(len(col_side), dtype=np.float32)
    nbr = np.full((len(col_side), 6), -1, dtype=np.int32)
    lut = {
        (int(s), int(a), int(b)): i
        for i, (s, (a, b)) in enumerate(zip(col_side.tolist(), col_hex.tolist(), strict=True))
    }
    lattice_raw = hex_to_raw(LATTICE_STEPS)

    for si in (0, 1):
        front, up = frames[si]
        m = col_side == si
        posterior = -(raw[m] @ front)
        dorsal = raw[m] @ up
        posterior -= posterior.min()
        dorsal -= dorsal.mean()

        sign = 1.0 if si == 1 else -1.0
        col_az[m] = sign * (-OVERLAP_DEG / 2.0 + posterior * SPACING_DEG)
        col_el[m] = dorsal * SPACING_DEG

        component = np.stack([lattice_raw @ front, lattice_raw @ up], axis=1)
        slot = np.empty(6, dtype=np.int64)
        for k, (cf, cu) in enumerate(component):
            if abs(cu) < 1e-6:
                slot[k] = 0 if cf > 0 else 1
            elif cu > 0:
                slot[k] = 2 if cf > 0 else 3
            else:
                slot[k] = 4 if cf > 0 else 5

        for i in np.flatnonzero(m):
            a, b = map(int, col_hex[i])
            for k, (da, db) in enumerate(LATTICE_STEPS.tolist()):
                nbr[i, slot[k]] = lut.get((si, a + da, b + db), -1)

    print("[5/6] Construyendo samplers retina 160x120 + RGB 80x60...", flush=True)
    pix_indptr, pix_indices, pix_weights, coverage = build_sampler(
        col_az,
        col_el,
        col_side,
        width=FRAME_W,
        height=FRAME_H,
    )
    rgb_h, rgb_w = FRAME_H // RGB_DOWNSAMPLE, FRAME_W // RGB_DOWNSAMPLE
    rgb_indptr, rgb_indices, rgb_weights, rgb_coverage = build_sampler(
        col_az,
        col_el,
        col_side,
        width=rgb_w,
        height=rgb_h,
    )

    driven = coverage >= 0.5
    n_driven = np.asarray(
        [int((driven & (col_side == s)).sum()) for s in (0, 1)],
        dtype=np.float64,
    )
    side_gain = (n_driven.mean() / np.maximum(n_driven, 1.0)).astype(np.float32)

    print(
        f"  driven L/R={int(n_driven[0])}/{int(n_driven[1])} "
        f"sampler nnz={len(pix_indices):,} rgb nnz={len(rgb_indices):,}",
        flush=True,
    )

    print("[6/6] Ordenando cells exactamente como el substrate y guardando...", flush=True)
    by_body = {int(b): i for i, b in enumerate(vbody)}
    order = np.asarray([by_body[int(b)] for b in expected_visual_body], dtype=np.int64)

    cell_type_id = np.asarray(
        [CLAMPED_TYPES.index(str(vtype[i])) for i in order],
        dtype=np.int16,
    )

    geometry = VisualGeometry(
        col_side=col_side,
        col_hex=col_hex,
        col_az=col_az,
        col_el=col_el,
        nbr=nbr,
        coverage=coverage,
        driven=driven,
        pix_indptr=pix_indptr,
        pix_indices=pix_indices,
        pix_weights=pix_weights,
        rgb_indptr=rgb_indptr,
        rgb_indices=rgb_indices,
        rgb_weights=rgb_weights,
        rgb_coverage=rgb_coverage,
        side_gain=side_gain,
        cell_body_id=vbody[order].astype(np.int64),
        cell_type_id=cell_type_id,
        cell_side=vside[order].astype(np.int8),
        cell_col=vcol[order].astype(np.int32),
        meta={
            "version": "imas-malecns-crossy-v2-visual-geometry-1",
            "frame_w": FRAME_W,
            "frame_h": FRAME_H,
            "rgb_w": rgb_w,
            "rgb_h": rgb_h,
            "hfov_deg": HFOV_DEG,
            "spacing_deg": SPACING_DEG,
            "fwhm_deg": FWHM_DEG,
            "overlap_deg": OVERLAP_DEG,
            "trunc_sigma": TRUNC_SIGMA,
            "neighbor_names": list(NBR_NAMES),
            "clamped_types": list(CLAMPED_TYPES),
            "frame_evidence": frame_meta,
            "assignment_counts": {
                "own_hex": int((assign_kind == 0).sum()),
                "pre_partner": int((assign_kind == 1).sum()),
                "post_partner": int((assign_kind == 2).sum()),
                "unassigned": int((assign_kind == 3).sum()),
            },
            "cross_side_rejected": int(wrong_side.sum()),
            "build_seconds": time.time() - started,
            "input_contract": (
                "Perspective egocentric RGB frame, 160x120, nominal 108 degree HFOV. "
                "This artifact defines the fly retina; it is intentionally separate "
                "from the human-facing isometric Three.js camera."
            ),
        },
    )
    geometry.validate(strict=True)
    geometry.save(output)

    loaded = VisualGeometry.load(output, strict=True)
    if not np.array_equal(loaded.cell_body_id, expected_visual_body):
        raise SystemExit("El artifact re-leido perdio el orden exacto del substrate.")

    print()
    print("VISUAL GEOMETRY BUILD: OK", flush=True)
    print(f"artifact: {output.resolve()}", flush=True)
    print(
        f"counts: {loaded.n_columns:,} columns / {loaded.n_cells:,} cells / "
        f"{int(loaded.driven.sum()):,} driven columns",
        flush=True,
    )
    print(f"unassigned: {int((loaded.cell_col < 0).sum())}", flush=True)


if __name__ == "__main__":
    main()
