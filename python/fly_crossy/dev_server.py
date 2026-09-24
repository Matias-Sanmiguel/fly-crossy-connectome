from __future__ import annotations

from pathlib import Path

from fly_crossy.biomechanics.world import BiomechanicalWorld
from fly_crossy.protocol import Action, Observation
from fly_crossy.runtime_controller import ConnectomeActionSelector
from fly_crossy.server import create_app


ROOT = Path(__file__).resolve().parents[2]

LEGACY_CHECKPOINT_PATH = (
    ROOT
    / "release"
    / "eval-v1"
    / "training"
    / "connectome"
    / "checkpoint.pt"
)

# Environment v10 is the current training candidate. Historical v6/v7/v8
# controllers remain immutable evidence; do not run them against the v9
# observation boundary. This path stays fail-closed until v9 is released.
CHECKPOINT_PATH = (
    ROOT
    / "release"
    / "eval-v10"
    / "training"
    / "connectome"
    / "checkpoint.pt"
)


controller: ConnectomeActionSelector | None = None


def _get_controller() -> ConnectomeActionSelector:
    global controller

    if controller is not None:
        return controller

    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError(
            "Environment v10 controller is not released yet."
        )

    controller = ConnectomeActionSelector(CHECKPOINT_PATH)
    return controller


def select_connectome_action(
    observation: Observation,
) -> Action:
    active_controller = _get_controller()
    action = active_controller(observation)

    logits = ", ".join(
        f"{value:+.2f}"
        for value in active_controller.logits
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
