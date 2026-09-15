"""Single source of truth for neural action to physical key mapping."""

from types import MappingProxyType
from typing import Mapping

from fly_crossy.protocol import Action, KeyName

from .contact import LegName


ACTION_TARGETS: Mapping[Action, tuple[LegName, KeyName]] = MappingProxyType(
    {
        "forward": ("front_left", "W"),
        "backward": ("front_right", "S"),
        "left": ("middle_left", "A"),
        "right": ("middle_right", "D"),
    }
)


__all__ = ["ACTION_TARGETS"]
