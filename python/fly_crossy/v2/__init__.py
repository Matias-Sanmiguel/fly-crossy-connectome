"""MaleCNS Crossy V2: frozen visual-to-descending connectome pipeline."""

from .substrate import MaleCNSV2Substrate
from .core import FrozenMaleCNSCore, FrozenMaleCNSState
from .vision_geometry import VisualGeometry
from .vision_frontend import FrozenVisualFrontEnd, VisualState
from .crossy_camera import render_crossy_neural_frame

__all__ = [
    "MaleCNSV2Substrate",
    "FrozenMaleCNSCore",
    "FrozenMaleCNSState",
    "VisualGeometry",
    "FrozenVisualFrontEnd",
    "VisualState",
    "render_crossy_neural_frame",
]
