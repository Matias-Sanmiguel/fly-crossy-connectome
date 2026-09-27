from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping

import torch

from .contracts import ACTION_NAMES, ProfileName, profile_budget


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    free_disk_bytes: int
    available_ram_bytes: int
    device: str
    torch_version: str
    thread_count: int


@dataclass(frozen=True, slots=True)
class CheckpointProvenance:
    version: str
    profile: ProfileName
    stage: str
    action_order: tuple[str, ...]
    graph_sha256: str | None
    teacher_sha256: str | None
    dataset_manifest_sha256: str
    seed_manifest_sha256: str
    completed_round: int
    completed_epoch: int


def _available_ram_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return int(pages * page_size)


def preflight_resources(
    profile: ProfileName,
    output: Path,
    device: torch.device,
) -> ResourceSnapshot:
    budget = profile_budget(profile)
    output.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(output).free)
    if free < budget.minimum_free_bytes:
        raise RuntimeError(
            f"insufficient free disk: {free} bytes available, "
            f"{budget.minimum_free_bytes} required for {profile}"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return ResourceSnapshot(
        free_disk_bytes=free,
        available_ram_bytes=_available_ram_bytes(),
        device=str(device),
        torch_version=torch.__version__,
        thread_count=torch.get_num_threads(),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_verified_checkpoint(
    path: Path,
    *,
    payload: Mapping[str, Any],
    verify: Callable[[Mapping[str, Any]], None],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=".v7-checkpoint-", suffix=".pt"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        if not isinstance(loaded, Mapping):
            raise ValueError("checkpoint payload must be a mapping")
        verify(loaded)
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _validate_provenance_payload(
    payload: Mapping[str, Any],
    expected: CheckpointProvenance,
) -> None:
    raw = payload.get("provenance")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint provenance is missing")
    action_order = tuple(raw.get("action_order", ()))
    if action_order != ACTION_NAMES:
        raise ValueError("checkpoint action order is not canonical")
    expected_payload = asdict(expected)
    expected_payload["action_order"] = tuple(expected.action_order)
    actual_payload = dict(raw)
    actual_payload["action_order"] = action_order
    if actual_payload != expected_payload:
        differing = sorted(
            key
            for key in set(actual_payload) | set(expected_payload)
            if actual_payload.get(key) != expected_payload.get(key)
        )
        raise ValueError(
            f"checkpoint provenance does not match expected fields: {differing}"
        )


def load_verified_checkpoint(
    path: Path,
    *,
    expected: CheckpointProvenance,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    _validate_provenance_payload(payload, expected)
    return payload


def retain_run_artifacts(run_dir: Path) -> tuple[Path, ...]:
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    fixed_names = {
        "best.pt",
        "latest.pt",
        "report.json",
        "dataset-manifest.json",
    }
    logs = [path for path in run_dir.iterdir() if path.is_file() and path.suffix == ".log"]
    newest_log = max(logs, key=lambda path: path.stat().st_mtime_ns, default=None)
    retained: list[Path] = []
    for path in run_dir.iterdir():
        keep = path.is_file() and (
            path.name in fixed_names or (newest_log is not None and path == newest_log)
        )
        if keep:
            retained.append(path)
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return tuple(sorted(retained, key=lambda path: path.name))
