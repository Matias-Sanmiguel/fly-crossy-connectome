"""Export checkpoint state, never train or fabricate activity. Run from repo root."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.feather as feather
import torch
from fly_crossy.env import step_game
from fly_crossy.schema import ACTION_ORDER
from fly_crossy.v2.crossy_camera import render_crossy_neural_frame
from fly_crossy.v4.exact_seed_branch_curriculum import load_decoder_checkpoint
from fly_crossy.v4.train_expo_specialist import default_flyhard_root, resize_rgb, variant_state
from fly_crossy.v5.sensory_core_adaptation import SensoryPlasticPolicy, _load_base_policy, load_adaptation_checkpoint
# Saved traces reference __main__.TraceStep; importing defines it without running training.
from repair_full_malecns_cached_frontier_v12_10 import TraceStep


def quantize_activity(state: np.ndarray, gain=1) -> np.ndarray:
    """Signed recurrent state -> magnitude, fixed [0,1] scale, nearest uint8."""
    if not np.isfinite(state).all() or (np.abs(state) > 1.000001).any():
        raise ValueError("Invalid recurrent state")
    if not np.isfinite(gain).all() or (np.asarray(gain) <= 0).any():
        raise ValueError("Invalid fixed display gain")
    return np.rint(np.clip(np.abs(state) * gain, 0, 1) * 255).astype(np.uint8)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_display_frames(ids: list[int], frames: np.ndarray, limit: int, regions: np.ndarray | None = None) -> tuple[list[int], np.ndarray]:
    """Keep strongest real magnitudes; ties use ascending body ID, never noise."""
    if limit < 1 or frames.dtype != np.uint8 or frames.ndim != 2 or frames.shape[1] != len(ids):
        raise ValueError("Invalid display selection")
    body_ids = np.asarray(ids, dtype=np.int64)
    if regions is not None and (regions.shape != (len(ids),) or not np.isin(regions, [0, 1, 2]).all()):
        raise ValueError("Invalid measured regions")
    selected = []
    previous = np.zeros(len(ids), dtype=np.uint8)
    for frame in frames:
        if regions is None:
            rank = np.lexsort((body_ids, -frame.astype(np.int16)))[:limit]
            chosen = rank[frame[rank] > 0]
        else:
            chosen_parts = []
            change = np.abs(frame.astype(np.int16) - previous.astype(np.int16))
            for region in range(3):
                quota = limit // 3 + int(region < limit % 3)
                candidates = np.flatnonzero((regions == region) & (frame > 0))
                moving = candidates[change[candidates] > 0]
                moving = moving[np.lexsort((body_ids[moving], -change[moving]))][:quota // 2]
                remaining = candidates[~np.isin(candidates, moving)]
                remaining = remaining[np.lexsort((body_ids[remaining], -frame[remaining].astype(np.int16)))][:quota - len(moving)]
                chosen_parts.append(np.concatenate([moving, remaining]))
            chosen = np.concatenate(chosen_parts)
        selected.append(chosen)
        previous = frame
    union = np.unique(np.concatenate(selected))
    lookup = np.zeros(len(ids), dtype=np.int64)
    lookup[union] = np.arange(len(union))
    compact = np.zeros((len(frames), len(union)), dtype=np.uint8)
    for i, indices in enumerate(selected):
        compact[i, lookup[indices]] = frames[i, indices]
    return [ids[int(i)] for i in union], compact


def publish_activity(directory: Path, manifest: dict, frames: np.ndarray, ids: list[int], limit: int, regions: np.ndarray | None = None) -> None:
    manifest["sourceFullDataSha256"] = hashlib.sha256(frames.tobytes()).hexdigest()
    manifest["sourceNonzeroPerFrame"] = manifest["nonzeroPerFrame"]
    region_by_id = dict(zip(ids, regions.tolist())) if regions is not None else {}
    ids, frames = select_display_frames(ids, frames, limit, regions)
    raw = frames.tobytes()
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    manifest.update({
        "bodyIds": ids, "dataSha256": hashlib.sha256(raw).hexdigest(),
        "displayNeuronLimit": limit,
        "nonzeroPerFrame": {"min": int(np.count_nonzero(frames, axis=1).min()), "max": int(np.count_nonzero(frames, axis=1).max())},
    })
    manifest["source"]["selectionRule"] = f"At most {limit} strongest nonzero magnitudes per frame; ties by ascending body ID. Omitted signals are not displayed."
    if regions is not None:
        manifest["source"]["selectionRule"] = f"At most {limit} nonzero signals, equal quotas for measured optic/central/descending regions; half each quota prioritizes actual magnitude changes, rest strongest nonzero state; ties by body ID. No random or timed pulses."
        manifest["bodyRegions"] = [region_by_id[body_id] for body_id in ids]
        compact_regions = np.asarray(manifest["bodyRegions"])
        manifest["regionNonzeroPerFrame"] = {
            name: {"min": int(np.count_nonzero(frames[:, compact_regions == region], axis=1).min()), "max": int(np.count_nonzero(frames[:, compact_regions == region], axis=1).max())}
            for region, name in enumerate(["optic", "central", "descending"])
        }
    (directory / manifest["dataFile"]).write_bytes(compressed)
    (directory / "full-malecns-activity.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"displayPool": len(ids), "frames": len(frames), "decodedBytes": len(raw), "compressedBytes": len(compressed), "nonzeroPerFrame": manifest["nonzeroPerFrame"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--display-neurons", type=int, default=1536)
    parser.add_argument("--compact-existing", action="store_true", help="Compact the already verified export without rerunning inference")
    args = parser.parse_args()
    if args.display_neurons < 1:
        parser.error("--display-neurons must be positive")
    directory = ROOT / "public/assets/expo"
    replay = json.loads((directory / "full-malecns-replay.json").read_text())
    if args.compact_existing:
        manifest = json.loads((directory / "full-malecns-activity.json").read_text())
        if "displayNeuronLimit" in manifest:
            raise ValueError("Export is already compact; rerun inference to change selection")
        raw = gzip.decompress((directory / manifest["dataFile"]).read_bytes())
        action_hash = hashlib.sha256(json.dumps(replay["actions"], separators=(",", ":")).encode()).hexdigest()
        if hashlib.sha256(raw).hexdigest() != manifest["dataSha256"] or action_hash != manifest["actionsSha256"] or manifest["seed"] != replay["seed"] or manifest["frameCount"] != len(replay["actions"]):
            raise ValueError("Original neural export does not match replay")
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(manifest["frameCount"], len(manifest["bodyIds"]))
        publish_activity(directory, manifest, frames, manifest["bodyIds"], args.display_neurons)
        return
    checkpoint = Path(replay["sourceCheckpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    flyhard = default_flyhard_root()
    base, _, _, _ = _load_base_policy(checkpoint_path=Path(payload["sourceCheckpoint"]), flyhard_root=flyhard, device=device)
    policy = SensoryPlasticPolicy(base, rank=payload["rank"], residual_scale=payload["residualScale"]).to(device)
    decoder, _ = load_decoder_checkpoint(Path(payload["sourceDecoder"]), device=device)
    load_adaptation_checkpoint(checkpoint, policy=policy, decoder=decoder)
    run_path = checkpoint.parent / "run_state.pt"
    run = torch.load(run_path, map_location="cpu", weights_only=False)
    trace, cache = run["committedTrace"], run["committedCache"]
    if run["signature"]["seed"] != replay["seed"] or len(trace) != len(replay["actions"]):
        raise ValueError("Committed trace does not match replay")
    if not all(torch.equal(v, payload["decoder"][k]) for k, v in run["committedDecoder"].items()):
        raise ValueError("Committed decoder does not match checkpoint")
    if [ACTION_ORDER[t.student_action] for t in trace] != replay["actions"]:
        raise ValueError("Committed actions do not match replay")
    policy.eval()
    decoder.eval()
    atlas = ROOT / "public/data/brain-atlas"
    atlas_ids = np.fromfile(atlas / "ids.bin", dtype="<u4")
    groups = np.fromfile(atlas / "groups.bin", dtype="u1")
    visible = set(atlas_ids[groups < 3].tolist())
    node_ids = feather.read_table(flyhard / "data/graph-traced-v1/nodes.feather")["bodyId"].to_numpy()
    indices = np.array([i for i, body_id in enumerate(node_ids) if int(body_id) in visible])
    ids = [int(v) for v in node_ids[indices]]
    region_by_id = dict(zip(atlas_ids.tolist(), groups.tolist()))
    regions = np.array([region_by_id[body_id] for body_id in ids], dtype=np.uint8)
    fixed_gains = np.array([1, 16, 16], dtype=np.float32)[regions]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Invalid body-ID mapping")
    state = variant_state(replay["seed"], phase_offset=0.0, initial_column=0)
    neural = policy.zero_state(1, device=device)
    frames = []
    with torch.inference_mode():
        for index, expected in enumerate(replay["actions"]):
            if index in cache:
                neural = cache[index].to(device=device, dtype=torch.float32).clone()
            if state.terminal is not None:
                raise ValueError(f"Early terminal at {index}")
            if state != trace[index].state:
                raise ValueError(f"Replay game state differs from committed observation at {index}")
            # Match the cache reconstruction used by the original run: its stored
            # observation snapshots, not a second reconstruction of camera state.
            image = torch.as_tensor(resize_rgb(render_crossy_neural_frame(trace[index].state))[None], dtype=torch.float32, device=device)
            neural = policy.step_state(neural, image)
            action = ACTION_ORDER[int(decoder(policy.motor_state(neural))[0].argmax().item())]
            if action != expected:
                raise ValueError(f"Checkpoint/replay action mismatch at {index}: {action} != {expected}; logits={decoder(policy.motor_state(neural))[0].cpu().tolist()}")
            frames.append(quantize_activity(neural[:, 0].cpu().numpy()[indices], fixed_gains))
            state = step_game(state, action).state
            if index % 50 == 0:
                print(f"Verified {index + 1}/{len(replay['actions'])} frames", flush=True)
    print("All actions verified; compacting display state", flush=True)
    manifest = {
        "version": 1, "kind": "full-malecns-neural-state-replay", "seed": replay["seed"],
        "sourceCheckpoint": checkpoint.name,
        "checkpointSha256": file_sha256(checkpoint),
        "runStateSha256": file_sha256(run_path),
        "reconstruction": "committed float32 recurrent-state anchors and committed trace RGB snapshots, four updates per decision",
        "actionsSha256": hashlib.sha256(json.dumps(replay["actions"], separators=(",", ":")).encode()).hexdigest(),
        "source": {"kind": "predicted", "name": "Full MaleCNS checkpoint recurrent state", "normalization": "abs(state), fixed [0,1], nearest uint8 / 255; signed model state magnitude, not biological recordings"},
        "bodyIds": ids, "frameCount": len(frames), "dataFile": "full-malecns-activity.bin.gz",
        "encoding": "frame-major-uint8-gzip",
        "nonzeroPerFrame": {"min": min(int(np.count_nonzero(v)) for v in frames), "max": max(int(np.count_nonzero(v)) for v in frames)},
    }
    manifest["source"]["normalization"] = "abs(state) with fixed regional display gains optic=1, central=16, descending=16; clipped [0,1], nearest uint8 / 255; model magnitude, not biological recordings"
    manifest["regionalDisplayGains"] = {"optic": 1, "central": 16, "descending": 16}
    # Nothing is published until every checkpoint action has matched.
    publish_activity(directory, manifest, np.stack(frames), ids, args.display_neurons, regions)


if __name__ == "__main__":
    main()
