from __future__ import annotations

import pytest

from fly_crossy.protocol import Configure, Observation, Pause, Reset, Resume, parse_client_message
from fly_crossy.session import SessionFault, SessionPhase, SimulationSession


SESSION_ID = "s-00000001"
EPISODE_ID = "e-00000001"


def client_message(value: dict[str, object]) -> Configure | Observation | Pause | Reset | Resume:
    message = parse_client_message(value)
    assert isinstance(message, (Configure, Observation, Pause, Reset, Resume))
    return message


def configure_message(sequence: int = 1) -> Configure:
    message = client_message({
        "type": "configure",
        "version": 2,
        "sessionId": SESSION_ID,
        "episodeId": EPISODE_ID,
        "sequence": sequence,
        "simulationTime": float(sequence),
        "population": 80,
        "backend": "cpu",
        "seed": 7,
        "speed": 1.0,
    })
    assert isinstance(message, Configure)
    return message


def reset_message(sequence: int, episode_id: str = EPISODE_ID) -> Reset:
    message = client_message({
        "type": "reset",
        "version": 2,
        "sessionId": SESSION_ID,
        "episodeId": episode_id,
        "sequence": sequence,
        "simulationTime": float(sequence),
    })
    assert isinstance(message, Reset)
    return message


def observation_message(
    sequence: int,
    step: int,
    episode_id: str = EPISODE_ID,
    session_id: str = SESSION_ID,
) -> Observation:
    message = client_message({
        "type": "observation",
        "version": 2,
        "sessionId": session_id,
        "episodeId": episode_id,
        "sequence": sequence,
        "simulationTime": float(sequence),
        "gameStep": step,
        "observation": [0.0] * 370,
        "reward": 0.0,
    })
    assert isinstance(message, Observation)
    return message


def lifecycle_message(message_type: str, sequence: int) -> Pause | Resume:
    message = client_message({
        "type": message_type,
        "version": 2,
        "sessionId": SESSION_ID,
        "episodeId": EPISODE_ID,
        "sequence": sequence,
        "simulationTime": float(sequence),
    })
    assert isinstance(message, (Pause, Resume))
    return message


def configured_session() -> SimulationSession:
    session = SimulationSession(SESSION_ID, EPISODE_ID)
    session.configure(configure_message())
    return session


def test_session_accepts_only_one_intention() -> None:
    session = configured_session()

    first = session.accept_observation(observation_message(sequence=2, step=0))

    assert first.intention_id == "i-00000001"
    with pytest.raises(SessionFault, match="in flight"):
        session.accept_observation(observation_message(sequence=3, step=0))


def test_reset_invalidates_old_action_result() -> None:
    session = configured_session()
    intention = session.accept_observation(observation_message(sequence=2, step=0))

    session.reset(reset_message(sequence=3, episode_id="e-new00001"))

    assert session.finish_action(intention.intention_id, "confirmed") is None


def test_non_monotonic_sequences_pause_the_session() -> None:
    session = configured_session()

    with pytest.raises(SessionFault, match="sequence"):
        session.accept_observation(observation_message(sequence=1, step=0))

    assert session.phase is SessionPhase.ERROR
    with pytest.raises(SessionFault, match="cannot resume"):
        session.resume(lifecycle_message("resume", sequence=2))


def test_mismatched_session_or_episode_pauses_the_session() -> None:
    session = configured_session()

    with pytest.raises(SessionFault, match="session"):
        session.accept_observation(observation_message(sequence=2, step=0, session_id="s-other0001"))

    assert session.phase is SessionPhase.ERROR


def test_mismatched_episode_pauses_the_session() -> None:
    session = configured_session()

    with pytest.raises(SessionFault, match="episode"):
        session.accept_observation(observation_message(sequence=2, step=0, episode_id="e-other0001"))

    assert session.phase is SessionPhase.ERROR


def test_duplicate_observation_is_rejected_after_action_completes() -> None:
    session = configured_session()
    intention = session.accept_observation(observation_message(sequence=2, step=0))
    assert session.finish_action(intention.intention_id, "waited") is not None

    with pytest.raises(SessionFault, match="(?i)duplicate observation"):
        session.accept_observation(observation_message(sequence=3, step=0))


def test_duplicate_completed_intention_is_an_idempotent_noop() -> None:
    session = configured_session()
    intention = session.accept_observation(observation_message(sequence=2, step=0))

    result = session.finish_action(intention.intention_id, "confirmed")

    assert result is not None
    assert session.finish_action(intention.intention_id, "confirmed") is None
    assert session.phase is SessionPhase.READY


def test_only_the_last_256_completed_intentions_are_idempotent() -> None:
    session = configured_session()
    for step in range(257):
        intention = session.accept_observation(observation_message(sequence=step + 2, step=step))
        assert session.finish_action(intention.intention_id, "waited") is not None

    assert session.finish_action("i-00000002", "waited") is None
    with pytest.raises(SessionFault, match="unknown intention"):
        session.finish_action("i-00000001", "waited")


def test_unknown_action_result_pauses_the_session() -> None:
    session = configured_session()

    with pytest.raises(SessionFault, match="unknown intention"):
        session.finish_action("i-unknown01", "failed")

    assert session.phase is SessionPhase.ERROR


def test_pause_and_resume_preserve_the_ready_lifecycle() -> None:
    session = configured_session()

    session.pause(lifecycle_message("pause", sequence=2))
    session.resume(lifecycle_message("resume", sequence=3))

    assert session.phase is SessionPhase.READY
