"""Executable compatibility check for the minimum supported Pydantic release.

Run this directly in an isolated environment after installing the exact
Pydantic version being verified. It deliberately loads the protocol module by
path so the check does not need the simulation package's NumPy/Torch runtime.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pydantic


def load_protocol() -> object:
    path = Path(__file__).parents[1] / "fly_crossy" / "protocol.py"
    spec = importlib.util.spec_from_file_location("protocol_minimum_check", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load protocol module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    expected_version = os.environ.get("EXPECTED_PYDANTIC_VERSION")
    if expected_version is not None:
        assert pydantic.__version__ == expected_version, pydantic.__version__

    protocol = load_protocol()
    hello = {
        "type": "hello",
        "version": 2,
        "sessionId": "s-01234567",
        "episodeId": "e-01234567",
        "sequence": 0,
        "simulationTime": 0,
        "supportedVersions": [2],
        "uiBuild": "minimum-check",
    }
    assert protocol.parse_client_message(hello).session_id == "s-01234567"

    for field, value in (("sequence", "0"), ("sequence", True), ("simulationTime", "0"), ("simulationTime", True)):
        try:
            protocol.parse_client_message({**hello, field: value})
        except protocol.ValidationError:
            continue
        raise AssertionError(f"{field} accepted non-numeric value {value!r}")

    print(f"protocol minimum compatibility passed with Pydantic {pydantic.__version__}")


if __name__ == "__main__":
    main()
