"""
tests.fixtures.fake_clock — Deterministic ``clock`` callable for e2e tests.

The Phase-0 DI seam at ``src/vehicle_registry/vehicle_registry.py:53`` accepts
an optional ``clock: Callable[[], datetime]`` and is consulted everywhere the
registry used to call ``datetime.now()`` (e.g. ``vehicle_registry_core.py:228``,
``vehicle_registry_identity.py:343, 992, 1085``). ``FakeClock`` is a drop-in
replacement that lets tests advance time explicitly and assert state at known
moments without sleeping.

Default starting time: ``2026-05-13T12:00:00Z`` (matches the current date in
the e2e test fixtures).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Optional


# 2026-05-13T12:00:00Z. Naive datetimes are used to mirror the registry's
# historical ``datetime.now()`` call (no tz). Tests that want tz-aware
# behaviour can construct a FakeClock with their own initial_time.
_DEFAULT_NOW = datetime(2026, 5, 13, 12, 0, 0)


class FakeClock:
    """Manually-controlled monotonic clock for matching e2e tests.

    Usage::

        clock = FakeClock()
        registry = VehicleRegistry(clock=clock, db_checker=FakeDBChecker())
        registry.register_anpr_event("ABC-123", "entry")
        clock.advance(45)  # simulate 45 seconds of wall time
        registry.confirm_at_b1_entrance(...)

    Parameters
    ----------
    initial_time:
        The starting datetime. Defaults to ``2026-05-13T12:00:00`` (naive).
    """

    def __init__(self, initial_time: Optional[datetime] = None):
        self._lock = RLock()
        self._now = initial_time if initial_time is not None else _DEFAULT_NOW
        self._frozen = False

    # ------------------------------------------------------------------ #
    # Callable protocol — what VehicleRegistry actually consumes.
    # ------------------------------------------------------------------ #

    def __call__(self) -> datetime:
        """Return the current fixture time. Identical to :meth:`now`."""
        return self.now()

    def now(self) -> datetime:
        """Return the current fixture time (does not advance the clock)."""
        with self._lock:
            return self._now

    # ------------------------------------------------------------------ #
    # Explicit time control
    # ------------------------------------------------------------------ #

    def advance(self, seconds: float) -> datetime:
        """Move the clock forward by ``seconds`` and return the new time.

        Has no effect when the clock is frozen — use :meth:`unfreeze` first.
        """
        with self._lock:
            if not self._frozen:
                self._now = self._now + timedelta(seconds=float(seconds))
            return self._now

    def set_now(self, when: datetime) -> None:
        """Jump the clock to an explicit datetime. Bypasses frozen state."""
        with self._lock:
            self._now = when

    def freeze(self) -> None:
        """Subsequent :meth:`advance` calls are ignored. Useful for asserting
        invariants at a fixed instant."""
        with self._lock:
            self._frozen = True

    def unfreeze(self) -> None:
        """Re-enable :meth:`advance`."""
        with self._lock:
            self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"FakeClock(now={self._now.isoformat()}, frozen={self._frozen})"
