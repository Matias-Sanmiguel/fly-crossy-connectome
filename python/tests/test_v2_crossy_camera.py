from __future__ import annotations

from dataclasses import replace
import ast
import inspect

import numpy as np

from fly_crossy.env import GameState, create_game
from fly_crossy.v2.crossy_camera import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HFOV_DEG,
    LANE_COLORS,
    frame_sha256,
    project_world_point,
    render_crossy_neural_frame,
)


def test_perspective_neural_camera_is_deterministic_and_fixed_size() -> None:
    state = create_game("perspective-camera-test")
    a = render_crossy_neural_frame(state)
    b = render_crossy_neural_frame(state)

    assert a.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    assert a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert frame_sha256(a) == frame_sha256(b)
    assert HFOV_DEG == 108.0


def test_perspective_neural_camera_exposes_lateral_position() -> None:
    state = create_game("perspective-camera-lateral")
    shifted = replace(
        state,
        fly=replace(state.fly, column=state.fly.column + 1.0),
    )

    assert not np.array_equal(
        render_crossy_neural_frame(state),
        render_crossy_neural_frame(shifted),
    )


def test_far_rows_are_perspectively_smaller_than_near_rows() -> None:
    state = create_game("perspective-camera-depth")

    near_a = project_world_point(state, 0.0, 0.0, state.fly.row + 1.0)
    near_b = project_world_point(state, 0.0, 0.0, state.fly.row + 2.0)
    far_a = project_world_point(state, 0.0, 0.0, state.fly.row + 9.0)
    far_b = project_world_point(state, 0.0, 0.0, state.fly.row + 10.0)

    assert all(value is not None for value in (near_a, near_b, far_a, far_b))
    near_pixels = abs(near_a[1] - near_b[1])
    far_pixels = abs(far_a[1] - far_b[1])

    assert far_pixels > 0
    assert far_pixels < 0.55 * near_pixels


def test_opening_grass_lane_surface_is_actually_rasterized() -> None:
    # Regression guard for the signed-area/barycentric bug that previously
    # discarded ground triangles while still drawing some box faces.
    state = create_game("perspective-camera-ground")
    frame = render_crossy_neural_frame(state)
    grass = np.asarray(LANE_COLORS["grass"], dtype=np.uint8)
    grass_pixels = int(np.all(frame == grass[None, None, :], axis=2).sum())

    assert grass_pixels >= 100


def _camera_ast() -> ast.AST:
    import fly_crossy.v2.crossy_camera as camera
    return ast.parse(inspect.getsource(camera))


def _imported_modules(tree: ast.AST) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    return imported


def _used_identifiers(tree: ast.AST) -> set[str]:
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
    return identifiers


def test_perspective_camera_has_no_observation_or_planner_channel() -> None:
    tree = _camera_ast()
    imported = _imported_modules(tree)
    identifiers = _used_identifiers(tree)

    # Inspect executable structure, not comments/docstrings.
    assert not any("expert_planner" in module for module in imported)
    assert not any("observation" in module.lower() for module in imported)
    assert "plan_action" not in identifiers
    assert "flatten_observation" not in identifiers
    assert not any(name.startswith("ObservationV") for name in identifiers)


def test_presentation_ui_cannot_affect_python_neural_camera() -> None:
    import fly_crossy.v2.crossy_camera as camera

    tree = _camera_ast()
    imported = _imported_modules(tree)
    identifiers = _used_identifiers(tree)

    signature = inspect.signature(camera.render_crossy_neural_frame)
    assert list(signature.parameters) == ["state"]
    assert signature.parameters["state"].annotation in (GameState, "GameState")
    assert signature.return_annotation in (np.ndarray, "np.ndarray")

    forbidden_import_fragments = (
        "three",
        "react",
        "browser",
        "selenium",
        "playwright",
    )
    assert not any(
        fragment in module.lower()
        for module in imported
        for fragment in forbidden_import_fragments
    )
    for forbidden_identifier in ("window", "document", "canvas", "css"):
        assert forbidden_identifier not in identifiers
