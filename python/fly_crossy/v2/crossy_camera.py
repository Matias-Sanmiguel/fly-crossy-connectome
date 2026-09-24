from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import numpy as np

from fly_crossy.env import (
    GameState,
    Lane,
    TRAIN_WARNING_SECONDS,
    WORLD_HALF_WIDTH,
    _hazard_sweeps_column,
    _scenery_obstacle_columns,
    hazard_position_at,
)


FRAME_WIDTH = 160
FRAME_HEIGHT = 120
HFOV_DEG = 108.0
CAMERA_HEIGHT = 1.15
CAMERA_PITCH_DEG = 25.0
NEAR_PLANE = 0.05

# The neural camera is attached to the fly. +Z is forward (increasing game row),
# +X is game-right, +Y is up.
ROWS_BEHIND = 0
ROWS_AHEAD = 14

SKY_COLOR = np.array([21, 39, 48], dtype=np.uint8)
OUTSIDE_GROUND_COLOR = (35, 66, 38)

LANE_COLORS: dict[str, tuple[int, int, int]] = {
    "grass": (88, 164, 74),
    "road": (32, 42, 41),
    "rail": (64, 58, 52),
    "river": (36, 152, 189),
}

HAZARD_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (235, 107, 66),
    "truck": (227, 179, 65),
    "train": (217, 225, 230),
    "log": (117, 72, 41),
}

ROAD_MARKING = (174, 182, 170)
RAIL_DETAIL = (121, 134, 129)
SCENERY_BLOCKER = (43, 83, 42)
WARNING_RED = (255, 45, 38)
LOG_END = (180, 123, 69)


def _focal_px() -> float:
    return FRAME_WIDTH / (2.0 * np.tan(np.radians(HFOV_DEG / 2.0)))


_FOCAL_PX = _focal_px()
_CX = FRAME_WIDTH / 2.0
_CY = FRAME_HEIGHT / 2.0

_pitch = np.radians(CAMERA_PITCH_DEG)
_CAMERA_FORWARD = np.array(
    [0.0, -np.sin(_pitch), np.cos(_pitch)],
    dtype=np.float64,
)
_CAMERA_RIGHT = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_CAMERA_UP = np.cross(_CAMERA_FORWARD, _CAMERA_RIGHT)
_CAMERA_UP /= np.linalg.norm(_CAMERA_UP)


def _camera_position(state: GameState) -> np.ndarray:
    return np.array(
        [float(state.fly.column), CAMERA_HEIGHT, float(state.fly.row)],
        dtype=np.float64,
    )


def _world_to_camera(state: GameState, points: np.ndarray) -> np.ndarray:
    rel = np.asarray(points, dtype=np.float64) - _camera_position(state)[None, :]
    return np.stack(
        [
            rel @ _CAMERA_RIGHT,
            rel @ _CAMERA_UP,
            rel @ _CAMERA_FORWARD,
        ],
        axis=1,
    )


def _project_camera(points_camera: np.ndarray) -> np.ndarray:
    points_camera = np.asarray(points_camera, dtype=np.float64)
    z = points_camera[:, 2]
    if np.any(z <= 0):
        raise ValueError("Projection expects positive camera-space depth.")
    return np.stack(
        [
            _CX + _FOCAL_PX * points_camera[:, 0] / z,
            _CY - _FOCAL_PX * points_camera[:, 1] / z,
            z,
        ],
        axis=1,
    )


def project_world_point(
    state: GameState,
    x: float,
    y: float,
    row: float,
) -> tuple[float, float, float] | None:
    """Project one world point into the fixed 160x120 neural camera."""
    camera = _world_to_camera(
        state,
        np.array([[x, y, row]], dtype=np.float64),
    )[0]
    if camera[2] <= NEAR_PLANE:
        return None
    projected = _project_camera(camera[None, :])[0]
    return float(projected[0]), float(projected[1]), float(projected[2])


class _Raster:
    def __init__(self) -> None:
        self.color = np.empty((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self.color[:] = SKY_COLOR
        self.depth = np.full((FRAME_HEIGHT, FRAME_WIDTH), np.inf, dtype=np.float32)

    def triangle(
        self,
        state: GameState,
        vertices_world: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        camera = _world_to_camera(state, vertices_world)
        if np.any(camera[:, 2] <= NEAR_PLANE):
            # The canonical camera is positioned so all submitted geometry is
            # expected to be in front of the near plane. Silently dropping a
            # crossing triangle is safer than inventing a different clipping
            # rule between training and inference.
            return

        projected = _project_camera(camera)
        x = projected[:, 0]
        y = projected[:, 1]
        z = projected[:, 2]

        min_x = max(0, int(np.floor(x.min())))
        max_x = min(FRAME_WIDTH - 1, int(np.ceil(x.max())))
        min_y = max(0, int(np.floor(y.min())))
        max_y = min(FRAME_HEIGHT - 1, int(np.ceil(y.max())))
        if max_x < min_x or max_y < min_y:
            return

        area = (
            (x[1] - x[0]) * (y[2] - y[0])
            - (y[1] - y[0]) * (x[2] - x[0])
        )
        if abs(area) < 1e-9:
            return

        xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
        px, py = np.meshgrid(xs, ys)

        w0 = ((x[1] - px) * (y[2] - py) - (y[1] - py) * (x[2] - px)) / area
        w1 = ((x[2] - px) * (y[0] - py) - (y[2] - py) * (x[0] - px)) / area
        w2 = 1.0 - w0 - w1

        # Barycentric coordinates are divided by the signed triangle area,
        # therefore points inside the triangle have non-negative weights
        # regardless of clockwise/counter-clockwise screen winding.
        #
        # The previous implementation flipped this predicate for negative-area
        # triangles and silently discarded most ground/lane quads.
        epsilon = -1e-6
        inside = (w0 >= epsilon) & (w1 >= epsilon) & (w2 >= epsilon)

        if not inside.any():
            return

        # Perspective-correct depth for visibility. For a planar triangle,
        # reciprocal depth interpolates linearly in screen space.
        inv_z = w0 / z[0] + w1 / z[1] + w2 / z[2]
        valid = inside & (inv_z > 0)
        if not valid.any():
            return
        depth = np.where(valid, 1.0 / np.maximum(inv_z, 1e-12), np.inf).astype(np.float32)

        region_depth = self.depth[min_y:max_y + 1, min_x:max_x + 1]
        nearer = valid & (depth < region_depth)
        if not nearer.any():
            return

        region_depth[nearer] = depth[nearer]
        region_color = self.color[min_y:max_y + 1, min_x:max_x + 1]
        region_color[nearer] = color

    def quad(
        self,
        state: GameState,
        vertices_world: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        v = np.asarray(vertices_world, dtype=np.float64)
        self.triangle(state, v[[0, 1, 2]], color)
        self.triangle(state, v[[0, 2, 3]], color)

    def box(
        self,
        state: GameState,
        *,
        center_x: float,
        center_row: float,
        width: float,
        depth: float,
        height: float,
        color: tuple[int, int, int],
    ) -> None:
        x0, x1 = center_x - width / 2.0, center_x + width / 2.0
        z0, z1 = center_row - depth / 2.0, center_row + depth / 2.0
        y0, y1 = 0.015, height

        p = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y0, z1],
                [x0, y0, z1],
                [x0, y1, z0],
                [x1, y1, z0],
                [x1, y1, z1],
                [x0, y1, z1],
            ],
            dtype=np.float64,
        )

        def shade(scale: float) -> tuple[int, int, int]:
            return tuple(
                int(np.clip(round(channel * scale), 0, 255))
                for channel in color
            )

        faces = (
            ((4, 5, 6, 7), shade(1.14)),  # top
            ((0, 1, 5, 4), shade(0.80)),
            ((1, 2, 6, 5), shade(0.92)),
            ((2, 3, 7, 6), shade(0.74)),
            ((3, 0, 4, 7), shade(0.88)),
        )
        for idx, face_color in faces:
            self.quad(state, p[list(idx)], face_color)


def _ground_quad(
    x0: float,
    x1: float,
    row0: float,
    row1: float,
    *,
    y: float = 0.0,
) -> np.ndarray:
    return np.array(
        [
            [x0, y, row0],
            [x1, y, row0],
            [x1, y, row1],
            [x0, y, row1],
        ],
        dtype=np.float64,
    )


def _draw_lane(raster: _Raster, state: GameState, lane: Lane) -> None:
    row0 = float(lane.row) - 0.5
    row1 = float(lane.row) + 0.5

    # On the current row, crop only the tiny sliver that would cross behind the
    # eye plane. This keeps the camera genuinely attached to the fly.
    if lane.row == state.fly.row:
        row0 = max(row0, float(state.fly.row) - 0.22)

    raster.quad(
        state,
        _ground_quad(
            -WORLD_HALF_WIDTH - 0.5,
            WORLD_HALF_WIDTH + 0.5,
            row0,
            row1,
        ),
        LANE_COLORS[lane.kind],
    )

    if lane.kind == "road":
        # Broken center stripe. This is visual-only and repeats in X.
        for x in np.arange(-WORLD_HALF_WIDTH - 0.3, WORLD_HALF_WIDTH + 0.3, 1.5):
            raster.quad(
                state,
                _ground_quad(
                    float(x),
                    float(min(x + 0.68, WORLD_HALF_WIDTH + 0.45)),
                    lane.row - 0.035,
                    lane.row + 0.035,
                    y=0.012,
                ),
                ROAD_MARKING,
            )
    elif lane.kind == "rail":
        for dz in (-0.20, 0.20):
            raster.quad(
                state,
                _ground_quad(
                    -WORLD_HALF_WIDTH - 0.5,
                    WORLD_HALF_WIDTH + 0.5,
                    lane.row + dz - 0.035,
                    lane.row + dz + 0.035,
                    y=0.016,
                ),
                RAIL_DETAIL,
            )


def _draw_world_boundary_ground(raster: _Raster, state: GameState) -> None:
    start = float(state.fly.row) - 0.22
    end = float(state.fly.row + ROWS_AHEAD) + 0.5
    raster.quad(
        state,
        _ground_quad(-14.0, -WORLD_HALF_WIDTH - 0.5, start, end),
        OUTSIDE_GROUND_COLOR,
    )
    raster.quad(
        state,
        _ground_quad(WORLD_HALF_WIDTH + 0.5, 14.0, start, end),
        OUTSIDE_GROUND_COLOR,
    )


def _draw_scenery(raster: _Raster, state: GameState, lane: Lane) -> None:
    if lane.kind != "grass":
        return

    for column in _scenery_obstacle_columns(state.seed, lane.row, lane.kind):
        raster.box(
            state,
            center_x=float(column),
            center_row=float(lane.row),
            width=0.72,
            depth=0.72,
            height=0.92,
            color=SCENERY_BLOCKER,
        )


def _draw_hazards(raster: _Raster, state: GameState, lane: Lane) -> None:
    for hazard in lane.hazards:
        center_x = float(hazard_position_at(lane, hazard, state.time))

        if hazard.kind == "log":
            height = 0.28
            depth = 0.42
        elif hazard.kind == "train":
            height = 0.82
            depth = 0.72
        else:
            height = 0.48 if hazard.kind == "car" else 0.62
            depth = 0.56 if hazard.kind == "car" else 0.66

        raster.box(
            state,
            center_x=center_x,
            center_row=float(lane.row),
            width=float(hazard.size),
            depth=depth,
            height=height,
            color=HAZARD_COLORS[hazard.kind],
        )

        if hazard.kind == "log":
            # Small bright end caps give a normal visual cue to log extent.
            cap = min(0.16, hazard.size * 0.08)
            for side in (-1.0, 1.0):
                end_x = center_x + side * (hazard.size / 2.0 - cap / 2.0)
                raster.box(
                    state,
                    center_x=end_x,
                    center_row=float(lane.row),
                    width=cap,
                    depth=depth * 1.02,
                    height=height * 1.02,
                    color=LOG_END,
                )


def _draw_rail_warning(raster: _Raster, state: GameState, lane: Lane) -> None:
    if lane.kind != "rail":
        return

    active = any(
        hazard.kind == "train"
        and _hazard_sweeps_column(
            lane,
            hazard,
            0.0,
            state.time,
            state.time + TRAIN_WARNING_SECONDS,
        )
        for hazard in lane.hazards
    )
    if not active:
        return

    for x in (-WORLD_HALF_WIDTH - 0.18, WORLD_HALF_WIDTH + 0.18):
        raster.box(
            state,
            center_x=float(x),
            center_row=float(lane.row - 0.12),
            width=0.18,
            depth=0.18,
            height=0.95,
            color=(76, 76, 72),
        )
        raster.box(
            state,
            center_x=float(x),
            center_row=float(lane.row - 0.12),
            width=0.28,
            depth=0.22,
            height=1.12,
            color=WARNING_RED,
        )


def render_crossy_neural_frame(state: GameState) -> np.ndarray:
    """Render the final fixed perspective neural sensor.

    Contract:
    - 160x120 RGB;
    - 108 degree horizontal FOV;
    - camera origin = current fly position;
    - fixed 25 degree downward pitch, fixed eye height;
    - only authoritative GameState/world mechanics are consumed;
    - no ObservationV4, planner value, action recommendation, CSS or browser state.

    The result is deliberately simple but genuinely perspective: distant rows
    shrink toward the horizon, 3-D hazards occlude ground behind them, and
    moving hazards create temporal image motion for the T4/T5 front end.
    """
    raster = _Raster()
    _draw_world_boundary_ground(raster, state)

    visible_lanes = [
        lane
        for lane in state.lanes
        if state.fly.row + ROWS_BEHIND <= lane.row <= state.fly.row + ROWS_AHEAD
    ]

    # Ground first; z-buffer makes the ordering robust, but this also minimizes
    # unnecessary overdraw.
    for lane in visible_lanes:
        _draw_lane(raster, state, lane)

    for lane in visible_lanes:
        _draw_scenery(raster, state, lane)
        _draw_hazards(raster, state, lane)
        _draw_rail_warning(raster, state, lane)

    return raster.color


def frame_sha256(frame: np.ndarray) -> str:
    array = np.asarray(frame)
    if array.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3) or array.dtype != np.uint8:
        raise ValueError("Expected one 160x120 uint8 RGB neural frame.")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_bmp(path: str | Path, frame: np.ndarray) -> None:
    """Write a 24-bit BMP using only the Python standard library."""
    frame = np.asarray(frame)
    if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3) or frame.dtype != np.uint8:
        raise ValueError("Expected one 160x120 uint8 RGB neural frame.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row_bytes = FRAME_WIDTH * 3
    padding = (4 - (row_bytes % 4)) % 4
    pixel_bytes = (row_bytes + padding) * FRAME_HEIGHT
    file_size = 14 + 40 + pixel_bytes

    with path.open("wb") as stream:
        stream.write(b"BM")
        stream.write(struct.pack("<IHHI", file_size, 0, 0, 54))
        stream.write(
            struct.pack(
                "<IIIHHIIIIII",
                40,
                FRAME_WIDTH,
                FRAME_HEIGHT,
                1,
                24,
                0,
                pixel_bytes,
                2835,
                2835,
                0,
                0,
            )
        )
        pad = b"\x00" * padding
        for row in frame[::-1]:
            stream.write(row[:, ::-1].tobytes())
            stream.write(pad)
