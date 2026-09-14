from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from fly_crossy.env import FlyCrossyEnv, generate_rows, hash_seed
from fly_crossy.schema import OBSERVATION_INPUT_SIZE, Action


FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "episode-v1.json"


def test_hash_and_row_generation_match_browser_contract() -> None:
    assert hash_seed("") == 2_166_136_261
    assert hash_seed("episode-v1-safe-opening") == 4_109_462_559
    assert hash_seed("parity-🪰") == 3_375_820_797

    assert [asdict(lane) for lane in generate_rows("parity-seed", 3, 2)] == [
        {
            "row": 3,
            "kind": "river",
            "hazards": [
                {"kind": "log", "position": -9, "size": 3},
                {"kind": "log", "position": 6, "size": 2},
            ],
            "direction": 1,
            "speed": 2,
            "phase": 0.53,
        },
        {
            "row": 4,
            "kind": "river",
            "hazards": [
                {"kind": "log", "position": -2, "size": 2},
                {"kind": "log", "position": -2, "size": 3},
                {"kind": "log", "position": 8, "size": 1},
                {"kind": "log", "position": -5, "size": 1},
            ],
            "direction": -1,
            "speed": 2,
            "phase": 0.718,
        },
    ]


def test_row_generation_rejects_booleans_as_browser_non_numbers() -> None:
    with pytest.raises(ValueError, match="integer starting row"):
        generate_rows("parity-seed", True, 1)
    with pytest.raises(ValueError, match="nonnegative row count"):
        generate_rows("parity-seed", 0, False)


def test_fixture_episode_matches_browser_contract() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    env = FlyCrossyEnv()

    observation, info = env.reset(seed=fixture["seed"])

    assert fixture["version"] == 1
    assert observation.shape == (OBSERVATION_INPUT_SIZE,)
    assert observation.dtype == np.float32
    assert info == {"score": 0, "seed": fixture["seed"]}

    rewards: list[float] = []
    for expected, action in zip(fixture["steps"], fixture["actions"], strict=True):
        observation, reward, terminated, truncated, info = env.step(Action(action))
        rewards.append(reward)
        assert observation.shape == (OBSERVATION_INPUT_SIZE,)
        assert reward == pytest.approx(expected["reward"])
        assert info["score"] == expected["score"]
        assert info["terminalReason"] == expected["terminalReason"]
        assert terminated is (expected["terminalReason"] is not None)
        assert truncated is False

    assert rewards == pytest.approx(fixture["rewards"])
    assert info["score"] == fixture["expectedScore"]
    assert info["terminalReason"] == fixture["terminalReason"]


def test_reset_is_deterministic_and_terminal_states_do_not_advance() -> None:
    env = FlyCrossyEnv()
    first, _ = env.reset(seed="episode-v1-safe-opening")
    env.step(Action.LEFT)
    second, _ = env.reset(seed="episode-v1-safe-opening")

    np.testing.assert_array_equal(second, first)

    env.step(Action.FORWARD)
    env.step(Action.FORWARD)
    _, reward, terminated, truncated, info = env.step(Action.FORWARD)
    terminal_step = env.state.step
    _, repeated_reward, repeated_terminated, repeated_truncated, repeated_info = env.step(Action.WAIT)

    assert reward == pytest.approx(-9.01)
    assert terminated is True
    assert truncated is False
    assert info == {"score": 3, "terminalReason": "water"}
    assert env.state.step == terminal_step
    assert repeated_reward == 0.0
    assert repeated_terminated is True
    assert repeated_truncated is False
    assert repeated_info == info
