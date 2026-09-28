"""Coordinated recovery for the authoritative biomechanical world."""

from __future__ import annotations

import logging
from typing import Protocol


logger = logging.getLogger(__name__)


class RecoverableWorld(Protocol):
    ready: bool
    requires_reset: bool
    failure_reason: str | None

    def step(self) -> tuple[object, ...]: ...

    def reset(self) -> None: ...


def recover_world(world: RecoverableWorld) -> None:
    """Return the body to neutral, using the runtime's coordinated reset fail-safe."""

    for _ in range(1500):
        if world.ready:
            return

        if world.requires_reset:
            reason = world.failure_reason or "unknown"
            logger.warning(
                "Biomechanical recovery requires coordinated reset: %s", reason
            )
            world.reset()
            if not world.ready:
                raise RuntimeError(
                    "Biomechanical reset failed to restore neutral state."
                )
            return

        world.step()

    logger.warning(
        "Biomechanical recovery exceeded server deadline; performing coordinated reset."
    )
    world.reset()
    if not world.ready:
        raise RuntimeError("Biomechanical reset failed to restore neutral state.")


__all__ = ["RecoverableWorld", "recover_world"]
