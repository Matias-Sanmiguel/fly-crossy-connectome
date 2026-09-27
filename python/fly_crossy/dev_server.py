from __future__ import annotations

from pathlib import Path

from fly_crossy.biomechanics.world import BiomechanicalWorld
from fly_crossy.protocol import Action, Observation
from fly_crossy.runtime_controller import ConnectomeActionSelector
from fly_crossy.server import SessionController, create_app


ROOT = Path(__file__).resolve().parents[2]

LEGACY_CHECKPOINT_PATH = (
    ROOT
    / "release"
    / "eval-v1"
    / "training"
    / "connectome"
    / "checkpoint.pt"
)

# Environment v11 is the current training candidate. Until its release exists,
# the runtime uses the verified v6 controller through the explicit
# ObservationV4 -> ObservationV1 adapter in ``ConnectomeActionSelector``.
CHECKPOINT_PATH = (
    ROOT
    / "release"
    / "eval-v6"
    / "training"
    / "connectome"
    / "checkpoint.pt"
)


def build_controller_runtime() -> SessionController:
    """Build a fresh recurrent controller for one WebSocket session."""
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError(
            "Released v6 compatibility controller is unavailable."
        )

    controller = ConnectomeActionSelector(
        CHECKPOINT_PATH,
        expected_environment_version=6,
    )
    def select_action(observation: Observation) -> Action:
        action = controller(observation)
        logits = ", ".join(f"{value:+.2f}" for value in controller.logits)
        print(
            f"[80n] "
            f"step={observation.game_step:04d} "
            f"action={action:8s} "
            f"logits=[{logits}]",
            flush=True,
        )
        return action

    def neural_activity() -> tuple[tuple[int, float], ...]:
        return tuple(
            zip(
                controller.body_ids,
                controller.activity,
                strict=True,
            )
        )

    return SessionController(
        select_action=select_action,
        neural_activity=neural_activity,
        reset=controller.reset,
    )


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
    controller_factory=build_controller_runtime,
    world_factory=build_world,
)
