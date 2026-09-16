from __future__ import annotations

from pathlib import Path

from fly_crossy.biomechanics.world import BiomechanicalWorld
from fly_crossy.protocol import Action, Observation
from fly_crossy.runtime_controller import ConnectomeActionSelector
from fly_crossy.server import create_app


ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    ROOT
    / "release"
    / "eval-v1"
    / "training"
    / "connectome"
    / "checkpoint.pt"
)


controller = ConnectomeActionSelector(
    CHECKPOINT_PATH,
)


def select_connectome_action(
    observation: Observation,
) -> Action:
    action = controller(observation)

    logits = ", ".join(
        f"{value:+.2f}"
        for value in controller.logits
    )

    print(
        f"[80n] "
        f"step={observation.game_step:04d} "
        f"action={action:8s} "
        f"logits=[{logits}]",
        flush=True,
    )

    return action


def build_world() -> BiomechanicalWorld:
    return BiomechanicalWorld.load(
        ROOT / "data/manifests/flybody-v1.json",
        ROOT / "configs/biomechanics-v1.json",
        ROOT / "data/calibration/keyboard-reach-v1.json",
    )


app = create_app(
    artifact_manifest=(
        ROOT
        / "runtime-artifacts-biomechanics.json"
    ),
    action_selector=select_connectome_action,
    world_factory=build_world,
)
