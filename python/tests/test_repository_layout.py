"""Historical tools must locate a clean checkout without generated run files."""

import ast
from pathlib import Path

import pytest


SCRIPTS = sorted(
    (Path(__file__).resolve().parents[2] / "scripts/experiments/cached_frontier")
    .glob("*.py")
)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_research_root_detection_without_local_runs(script, tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    (root / "python/fly_crossy").mkdir(parents=True)
    (root / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(root / "python")

    # Load only the actual discovery function: no GPU/model dependencies needed.
    tree = ast.parse(script.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "find_repo_root")
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(script), "exec"), namespace)

    assert namespace["find_repo_root"]() == root
    assert not (root / "runs").exists()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="inside the fly-crossy-connectome repository"):
        namespace["find_repo_root"]()
