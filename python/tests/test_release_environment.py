from __future__ import annotations

from collections import deque
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import platform
import sys

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest
import torch


PYTHON_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PYTHON_ROOT / "release-environment-linux-x86_64.json"


def _read_exact_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert line.count("==") == 1, f"release dependency is not exact: {line}"
        name, version = line.split("==")
        assert name and version and not any(char in version for char in "<>=!*~ ")
        normalized = canonicalize_name(name)
        assert normalized not in pins
        pins[normalized] = version
    return pins


def _active_dependency_closure() -> dict[str, str]:
    environment = default_environment()
    queue = deque(
        [("numpy", set()), ("torch", set()), ("pytest", set())]
    )
    expanded_contexts: dict[str, set[str]] = {}
    result: dict[str, str] = {}
    while queue:
        raw_name, requested_extras = queue.popleft()
        name = canonicalize_name(raw_name)
        contexts = {"", *requested_extras}
        previous = expanded_contexts.get(name, set())
        if contexts <= previous:
            continue
        expanded_contexts[name] = previous | contexts
        package = distribution(name)
        result[name] = package.version
        for raw_requirement in package.requires or ():
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None and not any(
                requirement.marker.evaluate({**environment, "extra": extra})
                for extra in contexts
            ):
                continue
            queue.append((requirement.name, set(requirement.extras)))
    return result


def test_release_environment_manifest_and_active_dependency_closure_are_complete() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    declared = manifest["declaredPlatform"]
    running = {
        "implementation": platform.python_implementation(),
        "pythonFullVersion": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    if running != declared:
        pytest.skip(f"release closure applies only to {declared}, running {running}")

    requirements_path = PYTHON_ROOT / manifest["requirementsFile"]
    requirements_bytes = requirements_path.read_bytes()
    assert hashlib.sha256(requirements_bytes).hexdigest() == manifest["requirementsSha256"]
    pins = _read_exact_pins(requirements_path)
    closure = _active_dependency_closure()

    assert pins == closure
    assert manifest["roots"] == ["numpy", "torch", "pytest"]
    libc_name, libc_version = platform.libc_ver()
    assert manifest["libc"] == {"name": libc_name, "version": libc_version}
    assert manifest["pipVersion"] == distribution("pip").version
    assert manifest["torch"]["distributionVersion"] == distribution("torch").version
    assert manifest["torch"]["runtimeVersion"] == torch.__version__
    assert manifest["torch"]["cudaBuild"] == torch.version.cuda
    assert manifest["torch"]["cudaAvailableDuringRelease"] is torch.cuda.is_available()
    assert manifest["torch"]["executionDevice"] == "cpu"
    assert manifest["integrityBoundary"] == "exact-versions-without-wheel-hashes"
    assert manifest["indexUrl"] == "https://pypi.org/simple"
