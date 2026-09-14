"""Transport-independent authority for one simulation session."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from fly_crossy.protocol import Action, ActionResult, Configure, Intention, Observation, Pause, Reset, Resume


class SessionPhase(StrEnum):
    CONNECTING = "connecting"
    READY = "ready"
    ACTING = "acting"
    PAUSED = "paused"
    ERROR = "error"


class SessionFault(Exception):
    """A stable code and safe explanation for a rejected session transition."""

    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(public_message)


@dataclass(slots=True)
class PendingIntention:
    id: str
    episode_id: str
    game_step: int
    requested_action: Action


class SimulationSession:
    """Keep client inputs and physical action completion causally single-flight."""

    _COMPLETION_HISTORY = 256

    def __init__(self, session_id: str, episode_id: str) -> None:
        self.session_id = session_id
        self.episode_id = episode_id
        self.phase = SessionPhase.CONNECTING
        self.pending: PendingIntention | None = None
        self._configuration: Configure | None = None
        self._last_sequence = -1
        self._last_game_step: int | None = None
        self._intention_number = 0
        self._completed_ids: deque[str] = deque()
        self._completed_id_set: set[str] = set()

    def configure(self, message: Configure) -> None:
        self._validate_envelope(message)
        if self.phase is not SessionPhase.CONNECTING:
            self._fault("INVALID_PHASE", "Session is already configured.")
        self._configuration = message
        self.phase = SessionPhase.READY

    def reset(self, message: Reset) -> None:
        self._validate_envelope(message, permit_episode_change=True)
        if self.phase is SessionPhase.CONNECTING:
            self._fault("NOT_CONFIGURED", "Session must be configured before it can reset.")
        if self.phase is SessionPhase.ERROR:
            self._fault("SESSION_IN_ERROR", "Session is in an error state and cannot reset.")

        self._complete_pending()
        self.episode_id = message.episode_id
        self._last_game_step = None
        self.phase = SessionPhase.READY

    def accept_observation(self, message: Observation) -> Intention:
        self._validate_envelope(message)
        if self.phase is SessionPhase.CONNECTING:
            self._fault("NOT_CONFIGURED", "Session must be configured before observations are accepted.")
        if self.phase is SessionPhase.ERROR:
            self._fault("SESSION_IN_ERROR", "Session is in an error state.")
        if self.phase is SessionPhase.PAUSED:
            self._fault("SESSION_PAUSED", "Session is paused.")
        if self.pending is not None:
            self._fault("INTENTION_IN_FLIGHT", "An intention is already in flight.")
        if self._last_game_step is not None and message.game_step == self._last_game_step:
            self._fault("DUPLICATE_OBSERVATION", "Duplicate observation for this game step.")
        if self._last_game_step is not None and message.game_step < self._last_game_step:
            self._fault("GAME_STEP_VIOLATION", "Game steps must increase monotonically.")

        self._intention_number += 1
        intention_id = f"i-{self._intention_number:08d}"
        action: Action = "wait"
        self.pending = PendingIntention(intention_id, self.episode_id, message.game_step, action)
        self._last_game_step = message.game_step
        self.phase = SessionPhase.ACTING
        return Intention.model_validate({
            "type": "intention",
            "version": 2,
            "sessionId": self.session_id,
            "episodeId": self.episode_id,
            "sequence": message.sequence,
            "simulationTime": message.simulation_time,
            "intentionId": intention_id,
            "action": action,
            "motorPhase": "targeting",
        })

    def finish_action(
        self,
        intention_id: str,
        result: Literal["confirmed", "waited", "failed"],
    ) -> ActionResult | None:
        if intention_id in self._completed_id_set:
            return None
        if self.phase is SessionPhase.ERROR:
            self._fault("SESSION_IN_ERROR", "Session is in an error state.")
        if self.pending is None or self.pending.id != intention_id:
            self._fault("UNKNOWN_INTENTION", "Action result references an unknown intention.")

        pending = self.pending
        self._complete_pending()
        self.phase = SessionPhase.READY
        return ActionResult.model_validate({
            "type": "action_result",
            "version": 2,
            "sessionId": self.session_id,
            "episodeId": pending.episode_id,
            "sequence": self._last_sequence,
            "simulationTime": float(self._last_sequence),
            "intentionId": pending.id,
            "result": result,
        })

    def pause(self, message: Pause) -> None:
        self._validate_envelope(message)
        if self.phase is SessionPhase.CONNECTING:
            self._fault("NOT_CONFIGURED", "Session must be configured before it can pause.")
        if self.phase is SessionPhase.ERROR:
            self._fault("SESSION_IN_ERROR", "Session is in an error state and cannot pause.")
        self.phase = SessionPhase.PAUSED

    def resume(self, message: Resume) -> None:
        self._validate_envelope(message)
        if self.phase is SessionPhase.ERROR:
            self._fault("SESSION_IN_ERROR", "Session is in an error state and cannot resume.")
        if self.phase is not SessionPhase.PAUSED:
            self._fault("INVALID_PHASE", "Session is not paused.")
        self.phase = SessionPhase.ACTING if self.pending is not None else SessionPhase.READY

    def _validate_envelope(self, message: Configure | Observation | Pause | Reset | Resume, *, permit_episode_change: bool = False) -> None:
        if message.session_id != self.session_id:
            self._fault("SESSION_MISMATCH", "Message session does not match this session.")
        if not permit_episode_change and message.episode_id != self.episode_id:
            self._fault("EPISODE_MISMATCH", "Message episode does not match this session.")
        if message.sequence <= self._last_sequence:
            self._fault("SEQUENCE_VIOLATION", "Message sequence must increase monotonically.")
        self._last_sequence = message.sequence

    def _complete_pending(self) -> None:
        if self.pending is None:
            return
        self._remember_completed(self.pending.id)
        self.pending = None

    def _remember_completed(self, intention_id: str) -> None:
        if len(self._completed_ids) == self._COMPLETION_HISTORY:
            self._completed_id_set.remove(self._completed_ids.popleft())
        self._completed_ids.append(intention_id)
        self._completed_id_set.add(intention_id)

    def _fault(self, code: str, public_message: str) -> None:
        self.phase = SessionPhase.ERROR
        raise SessionFault(code, public_message)
