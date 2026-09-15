from __future__ import annotations

from pathlib import Path

from fly_crossy.biomechanics.world import BiomechanicalWorld
from fly_crossy.protocol import Action, Observation
from fly_crossy.server import create_app


ROOT = Path(__file__).resolve().parents[2]

DEMO_ACTIONS: tuple[Action, ...] = (
    "forward",
    "left",
    "right",
    "backward",
)


def select_demo_action(observation: Observation) -> Action:
    return DEMO_ACTIONS[
        observation.game_step % len(DEMO_ACTIONS)
    ]


def build_world() -> BiomechanicalWorld:
    return BiomechanicalWorld.load(
        ROOT / "data/manifests/flybody-v1.json",
        ROOT / "configs/biomechanics-v1.json",
        ROOT / "data/calibration/keyboard-reach-v1.json",
    )


app = create_app(
    artifact_manifest=ROOT / "public/runtime-artifacts-dev.json",
    action_selector=select_demo_action,
    world_factory=build_world,
)