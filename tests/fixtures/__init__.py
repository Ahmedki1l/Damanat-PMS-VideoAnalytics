"""
tests.fixtures — Fault-injection fixtures for the matching cascade.

This package provides deterministic, GPU-free, network-free test scaffolding so
the failure modes of the matching feature can be exercised in CI without a real
GPU, database, ANPR server, or camera. Components:

  * ``fake_db.FakeDBChecker`` — drop-in replacement for the ``db_checker``
    callable accepted by ``VehicleRegistry.__init__`` (the Phase-0 DI seam at
    src/vehicle_registry/vehicle_registry.py:52). Supports always_true,
    always_false, raises, and intermittent failure modes plus call accounting.

  * ``fake_clock.FakeClock`` — drop-in replacement for the ``clock`` callable
    accepted by ``VehicleRegistry.__init__`` (DI seam at vehicle_registry.py:53).
    Exposes ``advance(seconds)`` / ``freeze()`` / ``unfreeze()`` for explicit
    deterministic time control.

  * ``match_fixtures`` — factory helpers that build ``PendingANPREvent``,
    ``ParkEntryCandidate``, ``VehicleSession`` objects and a fully wired
    ``VehicleRegistry`` test instance (``make_test_registry``) with stub
    ReID / image matchers so no real model is loaded.

These fixtures are consumed by ``tests/test_matching_e2e.py``.
"""

from tests.fixtures.fake_clock import FakeClock
from tests.fixtures.fake_db import FakeDBChecker
from tests.fixtures.match_fixtures import (
    StubImageMatcher,
    StubReIDMatcher,
    make_color_crop,
    make_park_entry_candidate,
    make_pending_anpr_event,
    make_test_registry,
    make_vehicle_session,
)

__all__ = [
    "FakeClock",
    "FakeDBChecker",
    "StubImageMatcher",
    "StubReIDMatcher",
    "make_color_crop",
    "make_park_entry_candidate",
    "make_pending_anpr_event",
    "make_test_registry",
    "make_vehicle_session",
]
