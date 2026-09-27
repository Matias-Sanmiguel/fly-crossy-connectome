from pathlib import Path

import pytest

from fly_crossy.protocol import Observation
from fly_crossy.runtime_controller import (
    ConnectomeActionSelector,
)


REPOSITORY_ROOT = (
    Path(__file__).resolve().parents[2]
)

CHECKPOINT_PATH = (
    REPOSITORY_ROOT
    / "release"
    / "eval-v1"
    / "training"
    / "connectome"
    / "checkpoint.pt"
)


def observation(
    episode_id: str,
    game_step: int,
    *,
    input_size: int = 370,
) -> Observation:
    return Observation.model_construct(
        type="observation",
        version=2,
        session_id="s-runtime01",
        episode_id=episode_id,
        sequence=game_step + 1,
        simulation_time=float(game_step),
        game_step=game_step,
        observation=[0.0] * input_size,
        reward=0.0,
    )


def test_real_80_neuron_checkpoint_runs_deterministically() -> None:
    controller = ConnectomeActionSelector(
        CHECKPOINT_PATH,
        expected_environment_version=3,
    )

    assert controller.node_count == 80
    assert len(controller.body_ids) == 80
    assert len(set(controller.body_ids)) == 80

    first_action = controller(
        observation(
            "e-episode01",
            0,
        )
    )

    first_activity = controller.activity

    assert first_action in {
        "forward",
        "backward",
        "left",
        "right",
        "wait",
    }

    assert len(first_activity) == 80

    assert all(
        0.0 <= value <= 1.0
        for value in first_activity
    )

    # Advance recurrent state.
    controller(
        observation(
            "e-episode01",
            1,
        )
    )

    # New episode must reset recurrent state.
    reset_action = controller(
        observation(
            "e-episode02",
            0,
        )
    )

    reset_activity = controller.activity

    assert reset_action == first_action

    assert reset_activity == pytest.approx(
        first_activity,
        abs=1e-7,
    )


def test_released_v6_controller_accepts_current_v11_observations_via_adapter() -> None:
    checkpoint = (
        REPOSITORY_ROOT
        / "release"
        / "eval-v6"
        / "training"
        / "connectome"
        / "checkpoint.pt"
    )
    controller = ConnectomeActionSelector(
        checkpoint,
        expected_environment_version=6,
    )

    first = controller(observation("e-current01", 0, input_size=517))
    first_activity = controller.activity
    reset = controller(observation("e-current02", 0, input_size=517))

    assert first in {"forward", "backward", "left", "right", "wait"}
    assert reset == first
    assert controller.activity == pytest.approx(first_activity, abs=1e-7)
