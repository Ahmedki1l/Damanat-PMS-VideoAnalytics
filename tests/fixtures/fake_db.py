"""
tests.fixtures.fake_db — Toggleable fake for the ``db_checker`` DI seam.

The Phase-0 DI seam at ``src/vehicle_registry/vehicle_registry.py:52`` accepts
an optional ``db_checker: Callable[[Optional[str]], bool]`` and consults it
from ``is_plate_inside`` (see ``src/vehicle_registry/vehicle_registry_identity.py:39-49``).
``FakeDBChecker`` provides four behaviours plus call accounting so e2e tests
can deterministically exercise the matching feature's interaction with the
parking-sessions probe.

Modes
-----

* ``"always_true"`` (default) — plate is "inside". Mirrors the legacy
  fail-open behaviour of ``is_plate_inside`` when the DB is unreachable.
* ``"always_false"`` — plate is "outside". Used to test "car already exited"
  guards at ``vehicle_registry_identity.py:574, 1154``.
* ``"raises"`` — raises a ``DBProbeError`` on every call. The registry's
  injected-checker branch wraps this in a ``try/except`` and returns ``True``
  (fail-open), so test code can assert the slot-bind still proceeds.
* ``"intermittent"`` — alternates between successes (returning the configured
  ``intermittent_value``) and failures (raising ``DBProbeError``) according
  to ``success_count`` and ``failure_count``.

Every call (whether it returns or raises) is recorded in ``self.calls`` so
tests can assert call frequency and ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


class DBProbeError(RuntimeError):
    """Raised by ``FakeDBChecker`` to simulate a database-side failure."""


@dataclass
class FakeDBChecker:
    """Drop-in replacement for the production ``is_plate_inside`` SQL probe.

    Parameters
    ----------
    mode:
        One of ``"always_true"`` / ``"always_false"`` / ``"raises"`` /
        ``"intermittent"``. Defaults to ``"always_true"``.
    success_count:
        For ``"intermittent"`` mode, the number of consecutive successful
        calls before a failure cycle. Defaults to ``2``.
    failure_count:
        For ``"intermittent"`` mode, the number of consecutive failing calls
        before flipping back to successes. Defaults to ``1``.
    intermittent_value:
        The boolean returned by ``"intermittent"`` mode while the success
        window is active. Defaults to ``True``.

    Attributes
    ----------
    calls:
        Ordered list of every plate this checker was invoked with.
    """

    mode: str = "always_true"
    success_count: int = 2
    failure_count: int = 1
    intermittent_value: bool = True

    calls: List[Optional[str]] = field(default_factory=list)
    _intermittent_step: int = 0

    _SUPPORTED_MODES = ("always_true", "always_false", "raises", "intermittent")

    def __post_init__(self) -> None:
        if self.mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported FakeDBChecker mode: {self.mode!r}. "
                f"Expected one of {self._SUPPORTED_MODES}."
            )
        if self.success_count < 0:
            raise ValueError("success_count must be >= 0")
        if self.failure_count < 0:
            raise ValueError("failure_count must be >= 0")

    # ------------------------------------------------------------------ #
    # Callable protocol — required by the registry DI seam.
    # ------------------------------------------------------------------ #

    def __call__(self, plate: Optional[str]) -> bool:
        """Probe the fake DB. Recording the call happens before any branch
        so ``self.calls`` always reflects every invocation, even the raising
        ones."""
        self.calls.append(plate)

        if self.mode == "always_true":
            return True
        if self.mode == "always_false":
            return False
        if self.mode == "raises":
            raise DBProbeError(f"simulated DB outage for plate={plate!r}")
        if self.mode == "intermittent":
            return self._intermittent_decision(plate)

        # Should be unreachable thanks to __post_init__ guards.
        raise RuntimeError(f"unhandled FakeDBChecker mode: {self.mode}")  # pragma: no cover

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _intermittent_decision(self, plate: Optional[str]) -> bool:
        """Cycle: ``success_count`` returns of ``intermittent_value`` followed
        by ``failure_count`` raises. Repeats forever."""
        cycle_len = self.success_count + self.failure_count
        if cycle_len == 0:
            # Pathological config — degenerate to always_true.
            return True
        position = self._intermittent_step % cycle_len
        self._intermittent_step += 1

        if position < self.success_count:
            return self.intermittent_value
        raise DBProbeError(
            f"simulated intermittent DB outage for plate={plate!r}"
        )

    @property
    def call_count(self) -> int:
        """Number of times this checker has been invoked."""
        return len(self.calls)

    def reset(self) -> None:
        """Clear call history and (for intermittent mode) the cycle pointer."""
        self.calls.clear()
        self._intermittent_step = 0
