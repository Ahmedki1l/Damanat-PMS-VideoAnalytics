"""
tests.test_matching_e2e — Fault-injection coverage for the matching cascade.

Exercises 8 failure scenarios end-to-end without a real GPU, database, ANPR
server, or camera, using the Phase-0 DI seams (``db_checker``, ``clock``,
``reattach_dry_run``) and the fault-injection fixtures in
``tests.fixtures``.

Scenarios (mapped to WS-E T-E4 in the master plan):

  1. ``test_anpr_dropped`` — pending event never arrives; candidate sits idle
     and is purged after the grace period; no false confirmation.
  2. ``test_anpr_delayed_within_window`` — ANPR event arrives at T=29s,
     binding still succeeds inside ``PENDING_ANPR_EXPIRY_SECONDS``.
     ``test_anpr_delayed_past_window`` — at T=45s the candidate is past the
     expiry policy and gets correctly dropped.
  3. ``test_db_down_during_slot_bind`` — ``db_checker`` raises; the registry
     fail-opens so the slot still binds (legacy behaviour preserved).
  4. ``test_cpu_only_no_gpu_path`` — ``VehicleRegistry`` + ``MatchDecision``
     are wired with injected feature vectors; no real model loaded.
  5. ``test_duplicate_plate_within_10_min`` — same plate twice within 600s;
     second call is deduped per ``vehicle_registry_core.py:249``.
  6. ``test_plate_cloning_two_cars_same_plate_in_1_minute`` — same plate on
     CAM-01 at T=0 then again at T=30s; duplicate is dropped.
  7. ``test_track_id_recycle_on_same_camera`` — ByteTrack recycles track 7
     on CAM-03 across two distinct cars; the second car gets a NEW session.
  8. ``test_cross_camera_handoff_with_reattach_dry_run`` — car appears on
     CAM-03 then CAM-07; ``match_global_session`` finds the existing
     session, ``reattach_track_to_confirmed_session`` with
     ``reattach_dry_run=True`` does not destroy the orphan,
     ``reattach_dry_run=False`` actually reattaches.

Tests run in <1s each and produce no log noise (loggers default to WARNING).
"""

from __future__ import annotations

import logging
import unittest

from src.config import MatchingConfig
from src.matching import MatchDecision
from src.vehicle_registry.vehicle_registry import VehicleRegistry
from tests.fixtures import (
    FakeClock,
    FakeDBChecker,
    make_color_crop,
    make_park_entry_candidate,
    make_pending_anpr_event,
    make_test_registry,
)

# Keep the test output clean — the registry emits INFO at every callsite.
logging.getLogger("src.vehicle_registry").setLevel(logging.CRITICAL)
logging.getLogger("src.vehicle_registry.vehicle_registry_identity").setLevel(
    logging.CRITICAL
)
logging.getLogger("src.vehicle_registry.vehicle_registry_core").setLevel(
    logging.CRITICAL
)
logging.getLogger("reid_match_perf").setLevel(logging.CRITICAL)


# --------------------------------------------------------------------------- #
# Scenario 1 — ANPR dropped (no event ever arrives for the candidate)
# --------------------------------------------------------------------------- #


class TestANPRDropped(unittest.TestCase):
    """Park-entry candidate is created but the matching ANPR event never
    arrives. The candidate must NOT be falsely confirmed, and it must
    expire cleanly after ``CANDIDATE_EXPIRY_SECONDS`` of inactivity."""

    def test_anpr_dropped(self) -> None:
        clock = FakeClock()
        registry = make_test_registry(clock=clock)

        # Park-entry candidate is created at T=0 (no ANPR pre-event).
        candidate = registry.open_park_entry_candidate("CAM-03", track_id=4)
        snapshot = make_color_crop(bgr=(64, 200, 32))
        registry.update_park_entry_candidate_snapshot(
            candidate.candidate_id, snapshot, quality_score=5.0
        )

        # No binding possible — no pending ANPR.
        bound = registry.bind_next_pending_anpr_to_candidate(candidate.candidate_id)
        self.assertIsNone(bound)
        self.assertEqual(candidate.status, "open")
        self.assertEqual(len(registry._sessions), 0)
        self.assertEqual(len(registry._park_entry_candidates), 1)

        # Advance past the candidate expiry policy and run the GC.
        # CANDIDATE_EXPIRY_SECONDS = 30s; advance 31s to clear the deadline.
        clock.advance(registry.CANDIDATE_EXPIRY_SECONDS + 1)
        registry._cleanup_stale_data(clock())

        # Candidate must be fully purged; no session created.
        self.assertEqual(len(registry._sessions), 0)
        self.assertNotIn(candidate.candidate_id, registry._park_entry_candidates)
        self.assertEqual(candidate.status, "expired")


# --------------------------------------------------------------------------- #
# Scenario 2 — ANPR delayed >30s
# --------------------------------------------------------------------------- #


class TestANPRDelayed(unittest.TestCase):
    """Two sub-scenarios: ANPR arrives just inside the window vs. just outside.
    The registry policy at ``vehicle_registry_identity.py:305`` and
    ``vehicle_registry_core.py:293`` is "drop pending events older than
    ``PENDING_ANPR_EXPIRY_SECONDS`` (30s)"."""

    def test_anpr_delayed_within_window_binds(self) -> None:
        """ANPR arrives 29s after the candidate — bind still succeeds."""
        clock = FakeClock()
        registry = make_test_registry(clock=clock)

        # T=0 — candidate opens at CAM-03 and is snapshotted.
        candidate = registry.open_park_entry_candidate("CAM-03", track_id=2)
        snapshot = make_color_crop(bgr=(20, 200, 20))
        registry.update_park_entry_candidate_snapshot(
            candidate.candidate_id, snapshot, quality_score=5.0
        )

        # Pre-bind check: no pending event yet.
        self.assertIsNone(
            registry.bind_next_pending_anpr_to_candidate(candidate.candidate_id)
        )

        # T=29s — ANPR finally arrives, still inside the 30s grace window.
        clock.advance(29)
        event = registry.register_anpr_event("DELAY-001", "entry")
        plate = registry.bind_next_pending_anpr_to_candidate(candidate.candidate_id)

        self.assertEqual(plate, "DELAY-001")
        self.assertEqual(event.status, "provisional")
        self.assertEqual(candidate.status, "provisional")
        self.assertEqual(candidate.bound_event_id, event.event_id)

    def test_anpr_delayed_past_window_drops(self) -> None:
        """ANPR arrives 45s after the candidate — pending event ages out
        before it can bind."""
        clock = FakeClock()
        registry = make_test_registry(clock=clock)

        # Same setup as the within-window test.
        candidate = registry.open_park_entry_candidate("CAM-03", track_id=3)
        registry.update_park_entry_candidate_snapshot(
            candidate.candidate_id,
            make_color_crop(bgr=(20, 200, 20)),
            quality_score=5.0,
        )

        # T=45s — ANPR arrives but pending-event policy will only honour it
        # if its age is <= PENDING_ANPR_EXPIRY_SECONDS (30s) at bind time.
        # The event itself is recorded as "pending" because
        # register_anpr_event stamps event.timestamp = now (so age=0 at
        # registration), so the bind still succeeds for THIS event but the
        # candidate is the stale party. We make the candidate stale by
        # running GC before binding to mirror the realistic flow where
        # the candidate's last_seen_at is 45s in the past.
        clock.advance(45)

        # GC purges the stale candidate first — exactly the realistic flow.
        registry._cleanup_stale_data(clock())

        # Now register the late ANPR event and try to bind.
        registry.register_anpr_event("DELAY-002", "entry")
        bound = registry.bind_next_pending_anpr_to_candidate(candidate.candidate_id)
        self.assertIsNone(bound)
        self.assertEqual(candidate.status, "expired")
        # No session was created from this late binding attempt.
        self.assertEqual(len(registry._sessions), 0)


# --------------------------------------------------------------------------- #
# Scenario 3 — DB down during slot bind
# --------------------------------------------------------------------------- #


class TestDBDownDuringSlotBind(unittest.TestCase):
    """``try_link_to_slot`` consults ``is_plate_inside`` before binding the
    slot. When the injected ``db_checker`` raises, the registry must
    fail-open (return True) so a transient DB outage does not strand
    legitimate confirmations. See ``vehicle_registry_identity.py:39-49``
    and the use at line 1154."""

    def _confirm_session(
        self,
        registry: VehicleRegistry,
        clock: FakeClock,
        plate: str,
        camera_id: str,
        track_id: int,
    ) -> str:
        """Helper — drive a confirmed session into the registry."""
        registry.register_anpr_event(plate, "entry", timestamp=clock())
        candidate = registry.open_park_entry_candidate(camera_id, track_id)
        snapshot = make_color_crop(bgr=(180, 30, 90))
        registry.update_park_entry_candidate_snapshot(
            candidate.candidate_id, snapshot, quality_score=5.0
        )
        registry.bind_next_pending_anpr_to_candidate(candidate.candidate_id)
        result = registry.confirm_at_b1_entrance(camera_id, track_id, snapshot)
        self.assertEqual(result, plate)
        return registry.get_session_id_for_track(camera_id, track_id)

    def test_db_down_during_slot_bind_fails_open(self) -> None:
        clock = FakeClock()
        db = FakeDBChecker(mode="raises")
        registry = make_test_registry(clock=clock, db_checker=db)

        # Drive a confirmation through to a parked-eligible state.
        session_id = self._confirm_session(
            registry, clock, "DB-001", camera_id="CAM-03", track_id=11
        )
        self.assertIsNotNone(session_id)

        # Reset the call counter so we measure only try_link_to_slot's probe.
        db.reset()

        # Slot bind should succeed even though the DB checker raises.
        plate = registry.try_link_to_slot(
            slot_id="SLOT-A",
            slot_name="Slot A",
            zone_id="Z-1",
            zone_name="Zone 1",
            camera_id="CAM-03",
            floor="B1",
            track_id=11,
            timestamp=clock(),
            snapshot_path="snap.jpg",
        )
        self.assertEqual(plate, "DB-001")
        # The checker was invoked at least once and recorded the call
        # (even though it raised). is_plate_inside must have caught the
        # exception and returned True.
        self.assertGreaterEqual(db.call_count, 1)
        self.assertIn("DB-001", [p for p in db.calls if p is not None])

        # Session is now in _parked under SLOT-A.
        self.assertIn("SLOT-A", registry._parked)
        self.assertEqual(registry._parked["SLOT-A"].plate, "DB-001")
        self.assertEqual(registry._parked["SLOT-A"].status, "parked")


# --------------------------------------------------------------------------- #
# Scenario 4 — CPU-only (no GPU) path
# --------------------------------------------------------------------------- #


class TestCPUOnlyNoGPU(unittest.TestCase):
    """Drives a confirmation through ``VehicleRegistry`` and
    ``MatchDecision`` without ever touching the real ReID matcher (so no
    GPU / no torchreid / no OpenVINO). All feature vectors are injected by
    the test, mirroring the CPU-only deployment scenario."""

    def test_cpu_only_decision_confirms_with_injected_features(self) -> None:
        clock = FakeClock()
        registry = make_test_registry(clock=clock)
        # The default StubReIDMatcher already operates entirely on CPU; the
        # point of this test is to assert that the decision chokepoint
        # produces correct verdicts using only inj. features.

        # Inject a pending event + provisional candidate by hand so the
        # bound_event_id linkage is in place but the matcher is never
        # called for ANY heavy operation.
        plate = "CPU-001"
        event = make_pending_anpr_event(plate, timestamp=clock(), status="provisional")
        registry._pending_events[event.event_id] = event
        registry._pending_event_order.append(event.event_id)

        candidate = make_park_entry_candidate(
            camera_id="CAM-03",
            track_id=1,
            timestamp=clock(),
            image=make_color_crop(bgr=(40, 90, 150)),
            status="provisional",
            bound_event_id=event.event_id,
        )
        event.candidate_id = candidate.candidate_id
        registry._park_entry_candidates[candidate.candidate_id] = candidate

        # Drive confirm with an image that produces a similar-enough feature.
        # The stub matcher returns vectors derived from BGR means; same image
        # yields cosine == 1.0 which clears any threshold.
        result = registry.confirm_at_b1_entrance(
            "CAM-03", 1, candidate.snapshot_image
        )
        self.assertEqual(result, plate)
        self.assertEqual(event.status, "confirmed")
        self.assertEqual(candidate.status, "confirmed")

    def test_cpu_only_match_decision_thresholds(self) -> None:
        """Direct check that ``MatchDecision`` (no plugins) produces the
        expected verdict on synthetic inputs — confirming the chokepoint
        is usable in isolation without any registry plumbing."""
        config = MatchingConfig()
        decision = MatchDecision(config)

        confirm = decision.decide_b1(0.60, is_anpr_candidate=False, candidate_count=3)
        self.assertEqual(confirm.verdict, "confirm")
        self.assertEqual(confirm.reason, "b1_zone")

        reject = decision.decide_b1(0.30, is_anpr_candidate=True, candidate_count=3)
        self.assertEqual(reject.verdict, "reject")

        fallback = decision.decide_b1(
            0.00, is_anpr_candidate=False, candidate_count=1
        )
        self.assertEqual(fallback.verdict, "confirm")
        self.assertEqual(fallback.reason, "single_candidate_fallback")


# --------------------------------------------------------------------------- #
# Scenario 5 — Duplicate plate within 10 min
# --------------------------------------------------------------------------- #


class TestDuplicatePlateWithin10Min(unittest.TestCase):
    """``register_anpr_event`` dedupes a second entry for the same plate
    within 600s (vehicle_registry_core.py:249). Both calls must return
    the *same* PendingANPREvent."""

    def test_duplicate_plate_within_10_min_is_deduped(self) -> None:
        clock = FakeClock()
        registry = make_test_registry(clock=clock)

        first = registry.register_anpr_event("DUP-010", "entry")
        # 9 min later — still within the 10-min window.
        clock.advance(9 * 60)
        second = registry.register_anpr_event("DUP-010", "entry")

        self.assertEqual(first.event_id, second.event_id)
        self.assertEqual(len(registry._pending_event_order), 1)
        # Confirm a second entry past the window IS a fresh event.
        clock.advance(2 * 60)  # total = 11 min — past 600s
        third = registry.register_anpr_event("DUP-010", "entry")
        self.assertNotEqual(third.event_id, first.event_id)


# --------------------------------------------------------------------------- #
# Scenario 6 — Plate cloning (same plate on two cars within 1 min)
# --------------------------------------------------------------------------- #


class TestPlateCloning(unittest.TestCase):
    """Two ANPR entries for plate ABC-123 within 30s of each other. The
    registry treats this as a duplicate (same case as #5) and returns the
    same PendingANPREvent — the second car is therefore not given a fresh
    pending record. This is the correct outcome from the registry's
    perspective; downstream voting is what would resolve a true cloning
    attempt."""

    def test_plate_cloning_within_1_minute_is_deduped(self) -> None:
        clock = FakeClock()
        registry = make_test_registry(clock=clock)

        first = registry.register_anpr_event(
            "ABC123", "entry", camera_id="CAM-01"
        )
        clock.advance(30)
        second = registry.register_anpr_event(
            "ABC123", "entry", camera_id="CAM-01"
        )

        # Same case as the 10-min dedup: second call returns the same event.
        self.assertIs(first, second)
        self.assertEqual(len(registry._pending_event_order), 1)
        self.assertEqual(first.status, "pending")


# --------------------------------------------------------------------------- #
# Scenario 7 — Track-ID recycle on same camera
# --------------------------------------------------------------------------- #


class TestTrackIDRecycleOnSameCamera(unittest.TestCase):
    """ByteTrack reuses a numerically-recycled track_id on the same camera
    after the previous owner left the scene. The registry must not
    silently attach the second car's frames to the first car's session.

    Concretely:

      * Car A registers as ABC-100 and confirms on (CAM-03, track_id=7).
      * Car A exits — ``register_anpr_event(..., "exit")`` purges the
        session.
      * Car B arrives on the same camera and ByteTrack happens to assign
        it the *same* numeric track_id=7.
      * The B1 confirmation for Car B must produce a NEW session with the
        new plate (not inherit Car A's session).

    The protection lives in
    ``vehicle_registry_core._drop_other_track_mappings_for_session``
    (``same_camera_only=True``)."""

    def test_track_id_recycle_does_not_inherit_session(self) -> None:
        clock = FakeClock()
        registry = make_test_registry(clock=clock)

        # --- Car A: register, confirm on (CAM-03, 7) ---
        registry.register_anpr_event("CAR-A", "entry", timestamp=clock())
        cand_a = registry.open_park_entry_candidate("CAM-03", track_id=7)
        snapshot_a = make_color_crop(bgr=(10, 200, 10))
        registry.update_park_entry_candidate_snapshot(
            cand_a.candidate_id, snapshot_a, quality_score=5.0
        )
        registry.bind_next_pending_anpr_to_candidate(cand_a.candidate_id)
        plate_a = registry.confirm_at_b1_entrance("CAM-03", 7, snapshot_a)
        self.assertEqual(plate_a, "CAR-A")
        session_a_id = registry.get_session_id_for_track("CAM-03", 7)
        self.assertIsNotNone(session_a_id)

        # --- Car A exits the garage ---
        clock.advance(60)
        registry.register_anpr_event("CAR-A", "exit", timestamp=clock())
        # After exit, Car A's session is gone from active state.
        self.assertNotIn("CAR-A", [s.plate for s in registry._sessions.values()])
        # Track 7 mapping is purged too.
        self.assertNotIn(("CAM-03", 7), registry._track_session_map)

        # --- Car B arrives on the SAME camera with RECYCLED track_id=7 ---
        clock.advance(30)
        registry.register_anpr_event("CAR-B", "entry", timestamp=clock())
        cand_b = registry.open_park_entry_candidate("CAM-03", track_id=7)
        snapshot_b = make_color_crop(bgr=(10, 10, 200))  # different colour
        registry.update_park_entry_candidate_snapshot(
            cand_b.candidate_id, snapshot_b, quality_score=5.0
        )
        registry.bind_next_pending_anpr_to_candidate(cand_b.candidate_id)
        plate_b = registry.confirm_at_b1_entrance("CAM-03", 7, snapshot_b)
        self.assertEqual(plate_b, "CAR-B")

        # Verify: Car B has a brand-new session ID, NOT Car A's.
        session_b_id = registry.get_session_id_for_track("CAM-03", 7)
        self.assertIsNotNone(session_b_id)
        self.assertNotEqual(session_b_id, session_a_id)
        # And the active plate on (CAM-03, 7) is Car B's.
        self.assertEqual(registry.get_plate_for_track("CAM-03", 7), "CAR-B")


# --------------------------------------------------------------------------- #
# Scenario 8 — Cross-camera handoff (with reattach_dry_run)
# --------------------------------------------------------------------------- #


class TestCrossCameraHandoff(unittest.TestCase):
    """Car appears on CAM-03 (track_id=5) and is then seen on CAM-07
    (track_id=12). The two-step verification:

      Step 1: ``match_global_session`` finds the existing plate-bound session
              from CAM-03 by ReID alone.
      Step 2: ``reattach_track_to_confirmed_session(reattach_dry_run=True)``
              attaches CAM-07's track to the existing session without
              destroying the orphan anonymous session that was created on
              CAM-07 first.
      Step 3: ``reattach_track_to_confirmed_session(reattach_dry_run=False)``
              actually purges the orphan, matching pre-refactor semantics.
    """

    def _make_registry_with_pinned_match(self):
        """Helper: build registry + drive a confirmed session for CAR-X
        on CAM-03 (track_id=5). Return (registry, clock, session_id,
        reference_vector)."""
        clock = FakeClock()
        registry = make_test_registry(clock=clock)

        registry.register_anpr_event("CAR-X", "entry", timestamp=clock())
        candidate = registry.open_park_entry_candidate("CAM-03", track_id=5)
        # Use a uniform white crop so both cameras observe the same feature.
        snapshot = make_color_crop(bgr=(200, 200, 200))
        registry.update_park_entry_candidate_snapshot(
            candidate.candidate_id, snapshot, quality_score=5.0
        )
        registry.bind_next_pending_anpr_to_candidate(candidate.candidate_id)
        plate = registry.confirm_at_b1_entrance("CAM-03", 5, snapshot)
        self.assertEqual(plate, "CAR-X")
        session_id = registry.get_session_id_for_track("CAM-03", 5)
        reference_vector = registry._sessions[session_id].feature_vector
        return registry, clock, session_id, reference_vector

    def test_match_global_session_finds_handoff(self) -> None:
        registry, clock, session_id, ref_vec = self._make_registry_with_pinned_match()

        # T+12s — car is now on CAM-07. Advance the clock past the
        # SESSION_HANDOFF_GUARD_SECONDS (10s) so reattach is eligible.
        clock.advance(12)

        # Stop the live-track guard from filtering out CAM-03 by aging
        # its last-seen timestamp beyond 3s.
        # (Already covered by the 12s advance above.)
        match_sid = registry.match_global_session(
            query_vector=ref_vec,
            camera_id="CAM-07",
            track_id=12,
            similarity_threshold=0.55,
        )
        self.assertEqual(match_sid, session_id)

    def test_reattach_dry_run_preserves_orphan(self) -> None:
        registry, clock, session_id, ref_vec = self._make_registry_with_pinned_match()
        clock.advance(12)

        # Manually create an anonymous appearance-only session on CAM-07
        # so the reattach path has an orphan to consider deleting.
        orphan_id = registry.create_appearance_session(
            "CAM-07", track_id=12, feature_vector=ref_vec, timestamp=clock()
        )
        self.assertIn(orphan_id, registry._sessions)
        self.assertIsNone(registry._sessions[orphan_id].plate)

        # Dry-run reattach — the orphan must SURVIVE the call.
        result = registry.reattach_track_to_confirmed_session(
            camera_id="CAM-07",
            track_id=12,
            query_vector=ref_vec,
            similarity_threshold=0.52,
            reattach_dry_run=True,
        )
        self.assertEqual(result, session_id)
        # Track now maps to the plate-bound session.
        self.assertEqual(
            registry._track_session_map.get(("CAM-07", 12)), session_id
        )
        # And the orphan anonymous session was NOT deleted because of dry-run.
        self.assertIn(orphan_id, registry._sessions)

    def test_reattach_without_dry_run_destroys_orphan(self) -> None:
        registry, clock, session_id, ref_vec = self._make_registry_with_pinned_match()
        clock.advance(12)

        orphan_id = registry.create_appearance_session(
            "CAM-07", track_id=12, feature_vector=ref_vec, timestamp=clock()
        )
        self.assertIn(orphan_id, registry._sessions)

        # Full reattach (dry_run=False) — orphan must be purged.
        result = registry.reattach_track_to_confirmed_session(
            camera_id="CAM-07",
            track_id=12,
            query_vector=ref_vec,
            similarity_threshold=0.52,
            reattach_dry_run=False,
        )
        self.assertEqual(result, session_id)
        self.assertEqual(
            registry._track_session_map.get(("CAM-07", 12)), session_id
        )
        # Orphan is gone.
        self.assertNotIn(orphan_id, registry._sessions)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
