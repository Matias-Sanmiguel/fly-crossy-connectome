from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXPECTED_COUNTS = {
    "nodes": 138_968,
    "edges": 4_639_332,
    "clamped": 30_906,
    "dynamic": 108_062,
    "descending": 1_312,
    "visual_projection": 9_199,
    "decision_readout": 10_511,
}


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class MaleCNSV2Substrate:
    body_ids: np.ndarray
    type_id: np.ndarray
    sign: np.ndarray
    is_clamped: np.ndarray
    is_dynamic: np.ndarray
    is_output: np.ndarray
    is_visual_projection: np.ndarray
    indptr_post: np.ndarray
    indices_pre: np.ndarray
    weight: np.ndarray
    indptr_pre: np.ndarray
    indices_post: np.ndarray
    perm_csr_to_csc: np.ndarray
    forward_hops: np.ndarray
    backward_hops: np.ndarray

    @classmethod
    def load(cls, path: str | Path, *, strict: bool = True) -> "MaleCNSV2Substrate":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as z:
            required = (
                "bodyId",
                "type_id",
                "sign",
                "is_clamped",
                "is_dynamic",
                "is_output",
                "is_visual_projection",
                "indptr_post",
                "indices_pre",
                "weight",
                "indptr_pre",
                "indices_post",
                "perm_csr_to_csc",
                "forward_hops",
                "backward_hops",
            )
            missing = [name for name in required if name not in z.files]
            if missing:
                raise ValueError(f"MaleCNS V2 substrate is missing arrays: {missing}")

            substrate = cls(
                body_ids=_readonly(z["bodyId"].astype(np.int64, copy=True)),
                type_id=_readonly(z["type_id"].astype(np.int32, copy=True)),
                sign=_readonly(z["sign"].astype(np.int8, copy=True)),
                is_clamped=_readonly(z["is_clamped"].astype(np.bool_, copy=True)),
                is_dynamic=_readonly(z["is_dynamic"].astype(np.bool_, copy=True)),
                is_output=_readonly(z["is_output"].astype(np.bool_, copy=True)),
                is_visual_projection=_readonly(
                    z["is_visual_projection"].astype(np.bool_, copy=True)
                ),
                indptr_post=_readonly(z["indptr_post"].astype(np.int64, copy=True)),
                indices_pre=_readonly(z["indices_pre"].astype(np.int32, copy=True)),
                weight=_readonly(z["weight"].astype(np.float32, copy=True)),
                indptr_pre=_readonly(z["indptr_pre"].astype(np.int64, copy=True)),
                indices_post=_readonly(z["indices_post"].astype(np.int32, copy=True)),
                perm_csr_to_csc=_readonly(
                    z["perm_csr_to_csc"].astype(np.int64, copy=True)
                ),
                forward_hops=_readonly(z["forward_hops"].astype(np.int8, copy=True)),
                backward_hops=_readonly(z["backward_hops"].astype(np.int8, copy=True)),
            )

        substrate.validate(strict=strict)
        return substrate

    @property
    def node_count(self) -> int:
        return int(self.body_ids.size)

    @property
    def edge_count(self) -> int:
        return int(self.indices_pre.size)

    @property
    def decision_readout_mask(self) -> np.ndarray:
        return np.logical_or(self.is_output, self.is_visual_projection)

    @property
    def decision_readout_count(self) -> int:
        return int(self.decision_readout_mask.sum())

    def validate(self, *, strict: bool = True) -> None:
        n = self.node_count
        e = self.edge_count

        vectors = {
            "type_id": self.type_id,
            "sign": self.sign,
            "is_clamped": self.is_clamped,
            "is_dynamic": self.is_dynamic,
            "is_output": self.is_output,
            "is_visual_projection": self.is_visual_projection,
            "forward_hops": self.forward_hops,
            "backward_hops": self.backward_hops,
        }
        for name, array in vectors.items():
            if array.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {array.shape}.")

        if n == 0 or e == 0:
            raise ValueError("MaleCNS V2 substrate cannot be empty.")
        if np.any(self.body_ids <= 0):
            raise ValueError("MaleCNS V2 body IDs must be positive.")
        if np.any(self.body_ids[1:] <= self.body_ids[:-1]):
            raise ValueError("MaleCNS V2 body IDs must be strictly increasing.")
        if not np.all(np.isin(self.sign, (-1, 1))):
            raise ValueError("MaleCNS V2 signs must be exactly -1 or +1.")
        if np.any(~np.isfinite(self.weight)) or np.any(self.weight <= 0):
            raise ValueError("MaleCNS V2 synaptic contact counts must be positive finite values.")

        if self.indptr_post.shape != (n + 1,):
            raise ValueError("indptr_post has an incompatible shape.")
        if self.indptr_post[0] != 0 or self.indptr_post[-1] != e:
            raise ValueError("indptr_post boundaries do not match the edge count.")
        if np.any(np.diff(self.indptr_post) < 0):
            raise ValueError("indptr_post must be monotonic.")
        if self.indices_pre.shape != (e,):
            raise ValueError("indices_pre has an incompatible shape.")
        if np.any(self.indices_pre < 0) or np.any(self.indices_pre >= n):
            raise ValueError("indices_pre contains an invalid node index.")

        if self.indptr_pre.shape != (n + 1,):
            raise ValueError("indptr_pre has an incompatible shape.")
        if self.indptr_pre[0] != 0 or self.indptr_pre[-1] != e:
            raise ValueError("indptr_pre boundaries do not match the edge count.")
        if self.indices_post.shape != (e,) or self.perm_csr_to_csc.shape != (e,):
            raise ValueError("CSC companion arrays have incompatible shapes.")

        if np.any(self.is_clamped & self.is_dynamic):
            raise ValueError("Clamped visual cells cannot also be dynamic.")
        if not np.all(self.is_clamped | self.is_dynamic):
            raise ValueError("Every selected cell must be clamped or dynamic.")
        if np.any(self.is_output & self.is_clamped):
            raise ValueError("Descending readout cells must be dynamic.")
        if np.any(self.is_visual_projection & self.is_clamped):
            raise ValueError("Visual-projection readout cells must be dynamic.")

        incoming = np.diff(self.indptr_post)
        if np.any(incoming[self.is_clamped] != 0):
            raise ValueError(
                "The frozen substrate must not retain recurrent edges into clamped visual cells."
            )

        if strict:
            observed = {
                "nodes": n,
                "edges": e,
                "clamped": int(self.is_clamped.sum()),
                "dynamic": int(self.is_dynamic.sum()),
                "descending": int(self.is_output.sum()),
                "visual_projection": int(self.is_visual_projection.sum()),
                "decision_readout": self.decision_readout_count,
            }
            mismatches = {
                key: (observed[key], expected)
                for key, expected in EXPECTED_COUNTS.items()
                if observed[key] != expected
            }
            if mismatches:
                raise ValueError(f"MaleCNS V2 frozen count mismatch: {mismatches}")
