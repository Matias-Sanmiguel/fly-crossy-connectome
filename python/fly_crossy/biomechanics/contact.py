"""Browser-independent physical contact confirmation and one-shot latching."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias

from fly_crossy.protocol import KeyName

from .config import KEY_NAMES, KeyboardConfig


LegName: TypeAlias = Literal[
    "front_left",
    "front_right",
    "middle_left",
    "middle_right",
    "hind_left",
    "hind_right",
]
ContactKind: TypeAlias = Literal[
    "pending",
    "confirmed",
    "wrong-key",
    "ambiguous",
    "unsafe-force",
    "timed-out",
    "cancelled",
]

LEG_NAMES: tuple[LegName, ...] = (
    "front_left",
    "front_right",
    "middle_left",
    "middle_right",
    "hind_left",
    "hind_right",
)
DIRECTIONAL_KEYS: frozenset[KeyName] = frozenset(("W", "A", "S", "D"))
KEY_TO_LEG: Mapping[KeyName, LegName] = MappingProxyType(
    {
        "W": "front_left",
        "A": "middle_left",
        "S": "front_right",
        "D": "middle_right",
        "SPACE_LEFT": "hind_left",
        "SPACE_RIGHT": "hind_right",
    }
)


def _finite_nonnegative(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field} must be finite; expected a finite non-negative SI value"
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(
            f"{field} must be finite; expected a finite non-negative SI value"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class KeyContact:
    """Complete physical state of one key for one simulation timestep.

    Travel is expressed in metres and normal force in newtons.  A key is
    considered physically touched only when ``touching_tarsi`` is non-empty;
    force and travel alone are deliberately insufficient contact evidence.
    """

    key: KeyName
    travel: float
    normal_force: float
    touching_tarsi: tuple[LegName, ...]

    def __post_init__(self) -> None:
        if self.key not in KEY_NAMES:
            raise ValueError(f"unknown key {self.key!r}")
        object.__setattr__(
            self,
            "travel",
            _finite_nonnegative(self.travel, "travel"),
        )
        object.__setattr__(
            self,
            "normal_force",
            _finite_nonnegative(self.normal_force, "normal_force"),
        )
        tarsi = tuple(self.touching_tarsi)
        if len(set(tarsi)) != len(tarsi) or any(
            tarsus not in LEG_NAMES for tarsus in tarsi
        ):
            raise ValueError("touching_tarsi must contain unique known leg names")
        ordered_tarsi = tuple(sorted(tarsi, key=LEG_NAMES.index))
        object.__setattr__(self, "touching_tarsi", ordered_tarsi)


@dataclass(frozen=True, slots=True)
class ContactSample:
    """Atomic six-key contact frame at one simulation timestamp.

    Requiring the whole keyboard in one frame prevents simultaneous contacts
    from becoming order-dependent one-at-a-time calls.
    """

    time: float
    contacts: tuple[KeyContact, ...]

    def __post_init__(self) -> None:
        timestamp = _finite_nonnegative(self.time, "time")
        contacts = tuple(self.contacts)
        keys = tuple(contact.key for contact in contacts)
        if len(contacts) != len(KEY_NAMES) or set(keys) != set(KEY_NAMES):
            raise ValueError(
                "contact frame must contain exactly one state for every key"
            )
        object.__setattr__(self, "time", timestamp)
        object.__setattr__(self, "contacts", contacts)


@dataclass(frozen=True, slots=True)
class ContactOutcome:
    """One observable contact-gate result.

    Terminal results are emitted only once.  ``None`` from a later
    :meth:`ContactGate.sample` means no event should be emitted.
    """

    kind: ContactKind
    intention_id: str
    key: KeyName | None

    @property
    def is_directional_confirmation(self) -> bool:
        """Whether this outcome can represent a confirmed directional key."""
        return self.kind == "confirmed" and self.key in DIRECTIONAL_KEYS


class ContactGateBusy(RuntimeError):
    """Raised when a second intention is started before the first terminates."""


@dataclass(slots=True)
class _ActiveIntention:
    intention_id: str
    intended_key: KeyName
    started_at: float | None
    debounce_started_at: float | None = None


class ContactGate:
    """Confirm physical key presses using config-owned SI thresholds.

    A travel reading beyond the configured mechanical limit is treated as
    invalid evidence: it resets continuous debounce and can only end through
    later valid evidence or timeout.  The intention clock starts with its
    first sampled physics frame, never with an earlier idle frame.
    """

    def __init__(self, config: KeyboardConfig) -> None:
        if not isinstance(config, KeyboardConfig):
            raise TypeError("config must be a validated KeyboardConfig")
        self._config = config
        self._active: _ActiveIntention | None = None
        self._completed_ids: set[str] = set()
        self._blocked_keys: set[KeyName] = set()
        self._last_time: float | None = None
        self._last_states: dict[KeyName, KeyContact] | None = None

    def begin(self, intention_id: str, intended_key: KeyName) -> None:
        """Begin one intention without bypassing an unreleased physical key."""
        if not isinstance(intention_id, str) or not intention_id:
            raise ValueError("intention_id must be a non-empty string")
        if intended_key not in KEY_NAMES:
            raise ValueError(f"unknown intended key {intended_key!r}")
        if self._active is not None:
            raise ContactGateBusy(
                f"contact gate is already processing {self._active.intention_id}"
            )
        if intention_id in self._completed_ids:
            raise ValueError(f"intention {intention_id} is already completed")
        self._active = _ActiveIntention(
            intention_id=intention_id,
            intended_key=intended_key,
            started_at=None,
        )

    def cancel(self) -> ContactOutcome | None:
        """Cancel the live intention once; repeated cancellation is silent."""
        if self._active is None:
            return None
        if self._last_states is not None:
            self._block_pressed_keys(self._last_states)
        return self._finish("cancelled", None)

    def sample(self, sample: ContactSample) -> ContactOutcome | None:
        """Evaluate one atomic keyboard frame and emit at most one terminal."""
        if not isinstance(sample, ContactSample):
            raise TypeError("sample must be a ContactSample")
        if self._last_time is not None and sample.time <= self._last_time:
            raise ValueError(
                "contact sample timestamps must be strictly increasing "
                "(strictly monotonic)"
            )
        self._last_time = sample.time
        states = {contact.key: contact for contact in sample.contacts}
        self._last_states = states
        self._observe_releases(states)

        active = self._active
        if active is None:
            return None
        if active.started_at is None:
            active.started_at = sample.time

        elapsed = sample.time - active.started_at
        if elapsed >= self._config.decision_timeout_seconds:
            self._block_pressed_keys(states)
            return self._finish("timed-out", None)

        touched_keys = tuple(
            key for key in KEY_NAMES if states[key].touching_tarsi
        )
        if len(touched_keys) > 1:
            self._block_pressed_keys(states)
            return self._finish("ambiguous", None)

        unsafe_keys = tuple(
            key
            for key in KEY_NAMES
            if states[key].normal_force > self._config.maximum_force_newtons
        )
        if unsafe_keys:
            self._block_pressed_keys(states)
            unsafe_key = unsafe_keys[0] if len(unsafe_keys) == 1 else None
            return self._finish("unsafe-force", unsafe_key)

        if touched_keys and touched_keys[0] != active.intended_key:
            self._block_pressed_keys(states)
            return self._finish("wrong-key", touched_keys[0])

        intended_state = states[active.intended_key]
        expected_tarsus = KEY_TO_LEG[active.intended_key]
        has_valid_evidence = (
            active.intended_key not in self._blocked_keys
            and intended_state.touching_tarsi == (expected_tarsus,)
            and self._config.minimum_travel_meters
            <= intended_state.travel
            <= self._config.key_travel_meters
            and self._config.minimum_force_newtons
            <= intended_state.normal_force
            <= self._config.maximum_force_newtons
        )
        if not has_valid_evidence:
            active.debounce_started_at = None
            return self._pending()

        if active.debounce_started_at is None:
            active.debounce_started_at = sample.time
            return self._pending()
        if sample.time - active.debounce_started_at < self._config.debounce_seconds:
            return self._pending()

        self._blocked_keys.add(active.intended_key)
        return self._finish("confirmed", active.intended_key)

    def _observe_releases(self, states: Mapping[KeyName, KeyContact]) -> None:
        for key in tuple(self._blocked_keys):
            if states[key].travel < self._config.release_travel_meters:
                self._blocked_keys.remove(key)

    def _block_pressed_keys(self, states: Mapping[KeyName, KeyContact]) -> None:
        self._blocked_keys.update(
            key
            for key in KEY_NAMES
            if states[key].travel >= self._config.release_travel_meters
        )

    def _pending(self) -> ContactOutcome:
        active = self._active
        if active is None:  # pragma: no cover - private invariant
            raise RuntimeError("contact gate has no active intention")
        return ContactOutcome("pending", active.intention_id, None)

    def _finish(
        self, kind: ContactKind, key: KeyName | None
    ) -> ContactOutcome:
        active = self._active
        if active is None:  # pragma: no cover - private invariant
            raise RuntimeError("contact gate has no active intention")
        outcome = ContactOutcome(kind, active.intention_id, key)
        self._completed_ids.add(active.intention_id)
        self._active = None
        return outcome


__all__ = [
    "ContactGate",
    "ContactGateBusy",
    "ContactKind",
    "ContactOutcome",
    "ContactSample",
    "DIRECTIONAL_KEYS",
    "KEY_TO_LEG",
    "KeyContact",
    "LEG_NAMES",
    "LegName",
]
