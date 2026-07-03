"""Tests for the plate-lock feature — freeze a confirmed plate onto a parked
slot so it can't drift/blank/move until the slot goes VACANT.

Three layers, all torch-free (they bypass the ReID matcher exactly like
test_plate_keyed_guard_smoke.py):

  * State machine  — lock/confidence fields + clear reset (pure unit).
  * DB schema      — the four new parking_slots columns land on SQLite via
                     both create_all and the legacy back-fill.
  * Registry       — lock set, freeze guards (a locked slot is never
                     relocated/released), confidence read, restart restore.
"""
import os
import sys
import unittest
import uuid
from datetime import datetime, timedelta

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

sys.path.append(os.path.abspath("."))

from src.database import Base
import src.model.parkingslot  # noqa: F401 — register ParkingSlot on Base
from src.models.state_machine import SlotState, SlotStateMachine
from src.vehicle_registry.vehicle_registry import VehicleRegistry
from src.vehicle_registry.vehicle_registry_models import VehicleSession

_NEW_COLS = {"current_plate", "plate_confidence", "plate_locked", "plate_locked_at"}


# --------------------------------------------------------------------------- #
# 1. State machine
# --------------------------------------------------------------------------- #
class TestStateMachineLock(unittest.TestCase):
    def _sm(self):
        return SlotStateMachine(slot_id="S1")

    def test_defaults_unlocked(self):
        sm = self._sm()
        self.assertFalse(sm.plate_locked)
        self.assertFalse(sm.is_plate_locked())
        self.assertEqual(sm.plate_confidence, 0.0)

    def test_bind_provisional_sets_confidence_not_lock(self):
        sm = self._sm()
        sm.bind_identity("ABC123", "url", confidence=0.55)
        self.assertEqual(sm.plate_number, "ABC123")
        self.assertEqual(sm.plate_confidence, 0.55)
        self.assertFalse(sm.is_plate_locked())

    def test_bind_lock_freezes(self):
        sm = self._sm()
        sm.bind_identity("ABC123", "url", confidence=0.82, lock=True)
        self.assertTrue(sm.is_plate_locked())
        self.assertEqual(sm.plate_confidence, 0.82)

    def test_confidence_none_leaves_stored_value(self):
        sm = self._sm()
        sm.bind_identity("ABC123", confidence=0.6)
        sm.bind_identity("ABC123")  # refresh without a new score
        self.assertEqual(sm.plate_confidence, 0.6)

    def test_clear_identity_resets_lock_and_confidence(self):
        sm = self._sm()
        sm.bind_identity("ABC123", "url", confidence=0.9, lock=True)
        sm.clear_identity()
        self.assertEqual(sm.plate_number, "")
        self.assertFalse(sm.plate_locked)
        self.assertEqual(sm.plate_confidence, 0.0)

    def test_slot_vacant_transition_clears_lock(self):
        # A LEAVING slot that confirms VACANT must drop the lock (clear_identity
        # runs inside the state machine's LEAVING->VACANT branch).
        sm = SlotStateMachine(slot_id="S1", confirm_leave_frames=2,
                              initial_state=SlotState.OCCUPIED)
        sm.bind_identity("ABC123", confidence=0.9, lock=True)
        sm.update(vehicle_present=False)          # OCCUPIED -> LEAVING
        events = []
        events += sm.update(vehicle_present=False)  # LEAVING -> VACANT (2 frames)
        self.assertEqual(sm.state, SlotState.VACANT)
        self.assertFalse(sm.plate_locked)
        self.assertEqual(sm.plate_number, "")


# --------------------------------------------------------------------------- #
# 2. DB schema (SQLite is the test dialect; MSSQL is prod)
# --------------------------------------------------------------------------- #
class TestPlateLockSchema(unittest.TestCase):
    def test_fresh_create_all_has_new_columns(self):
        from src.database import DatabaseManager

        db = DatabaseManager("sqlite:///:memory:")
        db.create_tables()
        cols = {c["name"] for c in inspect(db.engine).get_columns("parking_slots")}
        self.assertTrue(_NEW_COLS <= cols, f"missing: {_NEW_COLS - cols}")

    def test_backfill_adds_columns_to_legacy_table(self):
        from src.database import DatabaseManager

        db = DatabaseManager("sqlite:///:memory:")
        with db.engine.begin() as c:
            c.execute(text(
                "CREATE TABLE parking_slots "
                "(slot_id VARCHAR(50) PRIMARY KEY, slot_name VARCHAR(100))"
            ))
        db._ensure_schema_updates()
        cols = {c["name"] for c in inspect(db.engine).get_columns("parking_slots")}
        self.assertTrue(_NEW_COLS <= cols, f"missing: {_NEW_COLS - cols}")


# --------------------------------------------------------------------------- #
# 3. Registry lock / freeze / restore
# --------------------------------------------------------------------------- #
class TestRegistryLock(unittest.TestCase):
    PLATE = "94-LNV"
    CAM = "CAM_03"  # not identity-disabled

    def setUp(self):
        self.registry = VehicleRegistry(image_dir="tests/test_images")

    def _session(self, *, parked_slot=None, score=0.0, age_minutes=0):
        sid = f"sess-{uuid.uuid4().hex[:8]}"
        now = datetime.now()
        s = VehicleSession(
            session_id=sid,
            plate=self.PLATE,
            first_seen_at=now - timedelta(minutes=age_minutes),
            last_seen_at=now - timedelta(minutes=age_minutes),
            last_seen_camera=self.CAM,
            status="parked" if parked_slot else "confirmed",
            linked_slot=parked_slot,
            linked_slot_name=parked_slot,
            linked_camera=self.CAM if parked_slot else None,
            linked_floor="B1" if parked_slot else None,
            linked_at=now if parked_slot else None,
            new_pipeline_score=score,
        )
        self.registry._sessions[sid] = s
        if parked_slot:
            self.registry._parked[parked_slot] = s
        return s

    def test_lock_slot_and_query(self):
        self._session(parked_slot="Slot_A", score=0.83)
        self.assertFalse(self.registry.is_slot_locked("Slot_A"))
        self.registry.lock_slot("Slot_A")
        self.assertTrue(self.registry.is_slot_locked("Slot_A"))
        self.assertAlmostEqual(
            self.registry.get_slot_binding_confidence("Slot_A"), 0.83, places=5
        )

    def test_lock_slot_ignores_unparked(self):
        self.registry.lock_slot("Ghost")
        self.assertFalse(self.registry.is_slot_locked("Ghost"))

    def test_locked_slot_not_relocated_by_same_session(self):
        # Same session frozen into Slot_A must not move to Slot_B.
        s = self._session(parked_slot="Slot_A", score=0.9)
        self.registry.lock_slot("Slot_A")
        tid = 8888
        self.registry._track_session_map[(self.CAM, tid)] = s.session_id

        result = self.registry.try_link_to_slot(
            slot_id="Slot_B", slot_name="Slot B", zone_id="Z", zone_name="Z",
            camera_id=self.CAM, floor="B1", track_id=tid, timestamp=datetime.now(),
        )
        self.assertEqual(result, self.PLATE)
        self.assertIs(self.registry._parked.get("Slot_A"), s)
        self.assertNotIn("Slot_B", self.registry._parked)
        self.assertTrue(self.registry.is_slot_locked("Slot_A"))

    def test_locked_stale_slot_not_released_by_reid_slot_link(self):
        # ReID-side protection intact: a DIFFERENT session for the same plate
        # must NOT release a locked slot via try_link_to_slot (the in-frame path
        # keeps its "never steal a frozen identity" refusal). Only a genuine ANPR
        # gate read may evict a locked stale session — see
        # test_locked_stale_slot_evicted_by_anpr_reentry.
        stale = self._session(parked_slot="Slot_A", score=0.9, age_minutes=10)
        self.registry.lock_slot("Slot_A")
        fresh = self._session(age_minutes=0)
        tid = 9999
        self.registry._track_session_map[(self.CAM, tid)] = fresh.session_id

        self.registry.try_link_to_slot(
            slot_id="Slot_B", slot_name="Slot B", zone_id="Z", zone_name="Z",
            camera_id=self.CAM, floor="B1", track_id=tid, timestamp=datetime.now(),
        )
        # Locked Slot_A survives; the frozen plate stays put.
        self.assertIs(self.registry._parked.get("Slot_A"), stale)
        self.assertTrue(self.registry.is_slot_locked("Slot_A"))

    def test_unlink_slot_drops_lock(self):
        self._session(parked_slot="Slot_A", score=0.9)
        self.registry.lock_slot("Slot_A")
        self.registry.unlink_slot("Slot_A")
        self.assertFalse(self.registry.is_slot_locked("Slot_A"))
        self.assertNotIn("Slot_A", self.registry._parked)

    def test_restore_parked_binding_locked(self):
        self.registry.restore_parked_binding(
            slot_id="Slot_A", slot_name="Slot A", plate=self.PLATE,
            confidence=0.77, camera_id=self.CAM, floor="B1", locked=True,
            timestamp=datetime.now(),
        )
        self.assertIn("Slot_A", self.registry._parked)
        self.assertTrue(self.registry.is_slot_locked("Slot_A"))
        self.assertAlmostEqual(
            self.registry.get_slot_binding_confidence("Slot_A"), 0.77, places=5
        )

    def test_restore_parked_binding_provisional(self):
        self.registry.restore_parked_binding(
            slot_id="Slot_A", slot_name="Slot A", plate=self.PLATE,
            confidence=0.4, camera_id=self.CAM, floor="B1", locked=False,
            timestamp=datetime.now(),
        )
        self.assertIn("Slot_A", self.registry._parked)
        self.assertFalse(self.registry.is_slot_locked("Slot_A"))

    # ----------------------------------------------------------------------- #
    # ANPR chokepoint — _claim_plate_globally (one plate = one active session)
    # ----------------------------------------------------------------------- #
    def test_locked_stale_slot_evicted_by_anpr_reentry(self):
        # A genuine ANPR gate read IS authoritative: it evicts even a locked
        # stale session for the same plate (a likely missed exit).
        stale = self._session(parked_slot="Slot_A", score=0.9, age_minutes=10)
        self.registry.lock_slot("Slot_A")
        self.assertTrue(self.registry.is_slot_locked("Slot_A"))

        released = self.registry._claim_plate_globally(
            self.PLATE, keep_session_id=None, timestamp=datetime.now()
        )

        self.assertEqual(released, ["Slot_A"])
        self.assertEqual(stale.status, "exited")
        self.assertEqual(stale.exit_reason, "plate_reclaimed_elsewhere")
        self.assertNotIn(stale.session_id, self.registry._sessions)
        self.assertNotIn("Slot_A", self.registry._parked)
        self.assertFalse(self.registry.is_slot_locked("Slot_A"))

    def test_claim_plate_no_match_is_noop(self):
        s = self._session(parked_slot="Slot_A", score=0.9)
        released = self.registry._claim_plate_globally(
            "SOME-OTHER-PLATE", keep_session_id=None
        )
        self.assertEqual(released, [])
        self.assertIs(self.registry._parked.get("Slot_A"), s)
        self.assertEqual(s.status, "parked")

    def test_claim_plate_closes_slotless_session_no_slot_released(self):
        s = self._session(score=0.5)  # confirmed, not parked
        released = self.registry._claim_plate_globally(self.PLATE, keep_session_id=None)
        self.assertEqual(released, [])
        self.assertEqual(s.status, "exited")
        self.assertNotIn(s.session_id, self.registry._sessions)

    def test_claim_plate_excludes_keep_session(self):
        keep = self._session(parked_slot="Slot_A", score=0.9)
        other = self._session(parked_slot="Slot_B", score=0.8, age_minutes=5)
        released = self.registry._claim_plate_globally(
            self.PLATE, keep_session_id=keep.session_id
        )
        self.assertEqual(released, ["Slot_B"])
        self.assertIs(self.registry._parked.get("Slot_A"), keep)
        self.assertEqual(keep.status, "parked")
        self.assertEqual(other.status, "exited")
        self.assertNotIn("Slot_B", self.registry._parked)

    def test_claim_plate_ignores_already_exited(self):
        s = self._session(score=0.5)
        s.status = "exited"  # not an active victim
        released = self.registry._claim_plate_globally(self.PLATE, keep_session_id=None)
        self.assertEqual(released, [])
        self.assertIn(s.session_id, self.registry._sessions)  # left untouched

    def test_clear_slot_db_binding_invokes_primitive_off_lock(self):
        # _clear_slot_db_binding forwards to the DB primitive with the slot_id
        # and must run with the registry lock released (no DB I/O under lock).
        import threading
        import src.services.slot_status_service as svc

        calls = []
        lock_free = {"value": None}

        class _StubSession:
            def close(self):
                pass

        class _StubDB:
            def SessionLocal(self):
                return _StubSession()

        def _fake_clear(db, slot_id):
            calls.append(slot_id)
            acquired = {"ok": False}

            def _try():
                got = self.registry._lock.acquire(timeout=0.5)
                acquired["ok"] = got
                if got:
                    self.registry._lock.release()

            t = threading.Thread(target=_try)
            t.start()
            t.join()
            lock_free["value"] = acquired["ok"]
            return True

        orig = svc.clear_slot_plate_binding
        svc.clear_slot_plate_binding = _fake_clear
        try:
            self.registry.db_manager = _StubDB()
            self.registry._clear_slot_db_binding("Slot_A")
        finally:
            svc.clear_slot_plate_binding = orig

        self.assertEqual(calls, ["Slot_A"])
        self.assertTrue(lock_free["value"], "DB clear must run outside the registry lock")

    # ----------------------------------------------------------------------- #
    # Boot-restore dedup — one plate never restores onto two slots
    # ----------------------------------------------------------------------- #
    def _restore(self, slot, ts, locked=False):
        self.registry.restore_parked_binding(
            slot_id=slot, slot_name=slot, plate=self.PLATE,
            confidence=0.9, camera_id=self.CAM, floor="B1", locked=locked,
            timestamp=ts,
        )

    def test_restore_dedup_later_survives_earlier_evicted(self):
        t0 = datetime.now()
        self._restore("Slot_A", t0)                       # older
        self._restore("Slot_B", t0 + timedelta(minutes=5))  # newer
        self.assertNotIn("Slot_A", self.registry._parked)
        self.assertIn("Slot_B", self.registry._parked)
        self.assertEqual(self.registry.get_slot_plate("Slot_B"), self.PLATE)

    def test_restore_dedup_order_independent(self):
        # Restore the NEWER binding first, then the older: the older must be
        # skipped (not restored), not the other way round.
        t0 = datetime.now()
        self._restore("Slot_B", t0 + timedelta(minutes=5))  # newer, first
        self._restore("Slot_A", t0)                          # older, second
        self.assertIn("Slot_B", self.registry._parked)
        self.assertNotIn("Slot_A", self.registry._parked)

    def test_restore_dedup_locked_older_still_evicted(self):
        t0 = datetime.now()
        self._restore("Slot_A", t0, locked=True)
        self.assertTrue(self.registry.is_slot_locked("Slot_A"))
        self._restore("Slot_B", t0 + timedelta(minutes=5), locked=True)
        self.assertNotIn("Slot_A", self.registry._parked)
        self.assertFalse(self.registry.is_slot_locked("Slot_A"))
        self.assertIn("Slot_B", self.registry._parked)
        self.assertTrue(self.registry.is_slot_locked("Slot_B"))


if __name__ == "__main__":
    unittest.main()
