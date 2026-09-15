from __future__ import annotations

import pytest
from pydantic import ValidationError

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
    simulation_time: float | None = None,
) -> Observation:
    message = client_message({
        "type": "observation",
        "version": 2,
        "sessionId": session_id,
        "episodeId": episode_id,
        "sequence": sequence,
        "simulationTime": float(sequence) if simulation_time is None else simulation_time,
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


def test_server_frames_have_strictly_increasing_outbound_sequences() -> None:
    session = configured_session()
    intention = session.accept_observation(observation_message(sequence=2, step=0))

    result = session.finish_action(intention.intention_id, "waited", completion_time=2.5)

    assert result is not None
    assert result.sequence > intention.sequence


def test_terminal_result_uses_authoritative_completion_time() -> None:
    session = configured_session()
    intention = session.accept_observation(
        observation_message(sequence=47, step=0, simulation_time=12.25)
    )

    result = session.finish_action(intention.intention_id, "waited", completion_time=12.75)

    assert result is not None
    assert result.simulation_time == 12.75


def test_invalid_terminal_completion_does_not_mutate_or_consume_sequence() -> None:
    session = configured_session()
    intention = session.accept_observation(observation_message(sequence=2, step=0))

    with pytest.raises(ValidationError, match="simulationTime"):
        session.finish_action(
            intention.intention_id,
            "waited",
            completion_time=86_400.5,
        )

    assert session.pending is not None
    assert session.pending.id == intention.intention_id
    assert session.phase is SessionPhase.ACTING
    result = session.finish_action(
        intention.intention_id,
        "waited",
        completion_time=2.5,
    )
    assert result is not None
    assert result.sequence == intention.sequence + 1


def test_wait_intention_uses_a_neutral_motor_phase() -> None:
    session = configured_session()

    intention = session.accept_observation(observation_message(sequence=2, step=0))

    assert intention.action == "wait"
    assert intention.motor_phase == "neutral"


def test_reset_invalidates_old_action_result() -> None:
    session = configured_session()
    intention = session.accept_observation(observation_message(sequence=2, step=0))

    session.reset(reset_message(sequence=3, episode_id="e-new00001"))

    assert session.finish_action(intention.intention_id, "confirmed", completion_time=3.5) is None


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
    assert session.finish_action(intention.intention_id, "waited", completion_time=2.5) is not None

    with pytest.raises(SessionFault, match="(?i)duplicate observation"):
        session.accept_observation(observation_message(sequence=3, step=0))


def test_duplicate_completed_intention_is_an_idempotent_noop() -> None:
    session = configured_session()
    intention = session.accept_observation(observation_message(sequence=2, step=0))

    result = session.finish_action(intention.intention_id, "confirmed", completion_time=2.5)

    assert result is not None
    assert session.finish_action(intention.intention_id, "confirmed", completion_time=2.5) is None
    assert session.phase is SessionPhase.READY


def test_only_the_last_256_completed_intentions_are_idempotent() -> None:
    session = configured_session()
    for step in range(257):
        intention = session.accept_observation(observation_message(sequence=step + 2, step=step))
        assert session.finish_action(intention.intention_id, "waited", completion_time=step + 2.5) is not None

    assert session.finish_action("i-00000002", "waited", completion_time=259.5) is None
    with pytest.raises(SessionFault, match="unknown intention"):
        session.finish_action("i-00000001", "waited", completion_time=259.5)


def test_unknown_action_result_pauses_the_session() -> None:
    session = configured_session()

    with pytest.raises(SessionFault, match="unknown intention"):
        session.finish_action("i-unknown01", "failed", completion_time=2.5)

    assert session.phase is SessionPhase.ERROR


def test_pause_and_resume_preserve_the_ready_lifecycle() -> None:
    session = configured_session()

    session.pause(lifecycle_message("pause", sequence=2))
    session.resume(lifecycle_message("resume", sequence=3))

    assert session.phase is SessionPhase.READY

def test_observation_can_bind_to_a_real_requested_motor_action() -> None:
    session = configured_session()

    intention = session.accept_observation(
        observation_message(sequence=2, step=0),
        action="forward",
        motor_phase="targeting",
    )

    assert intention.action == "forward"
    assert intention.motor_phase == "targeting"
    assert session.pending is not None
    assert session.pending.requested_action == "forward"
    assert session.phase is SessionPhase.ACTING


def test_physical_recovery_blocks_the_next_observation_until_neutral() -> None:
    session = configured_session()

    intention = session.accept_observation(
        observation_message(sequence=2, step=0),
        action="forward",
        motor_phase="targeting",
    )

    result = session.finish_action(
        intention.intention_id,
        "confirmed",
        completion_time=2.5,
        recovery_pending=True,
    )

    assert result is not None
    assert session.phase is SessionPhase.RECOVERING

    with pytest.raises(SessionFault, match="recovery"):
        session.accept_observation(
            observation_message(sequence=3, step=1),
            action="left",
            motor_phase="targeting",
        )

    assert session.phase is SessionPhase.RECOVERING

    session.finish_recovery()

    assert session.phase is SessionPhase.READY

    next_intention = session.accept_observation(
        observation_message(sequence=4, step=1),
        action="left",
        motor_phase="targeting",
    )

    assert next_intention.action == "left"