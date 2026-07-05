"""
tests.test_reentry_grace — the re-entry DB-grace override on ``is_plate_inside``.

A car can exit (``parking_sessions`` row -> ``closed``) and RE-ENTER before
PMS-AI inserts the new ``open`` row. During that window ``is_plate_inside`` would
wrongly report the plate as "already exited" (the freshest DB row is still the
exit's ``closed`` row), which blocks the slot-bind, the B1 confirmation, and the
per-plate gallery folder seed, and lets the exit-janitor re-purge the fresh
session.

The fix records the latest ANPR entry/exit VA itself observes per plate, and
``is_plate_inside`` treats the plate as inside when VA saw a genuine
exit -> NEWER entry within ``REENTRY_DB_GRACE_SECONDS``. These tests pin that
behaviour AND the anti-ghost invariant it must not break (a plain missed-exit,
i.e. an exit with no later entry, is still reported outside).

``FakeDBChecker`` runs in the (previously unused) ``"always_false"`` mode so the
DB always reports "outside" — any ``True`` result therefore proves the
VA-internal override fired, and ``plate in db.calls`` proves the override is
layered on top of the DB probe, not a bypass of it.

The engine exit-janitor (``_exit_janitor_tick``) consults the same public
``has_recent_reentry`` predicate before purging, so the ``has_recent_reentry``
assertions below fully cover the janitor's skip/purge decision — no heavyweight
``_exit_janitor_tick`` integration test (which would need a DB + engine) is
needed.
"""

from tests.fixtures import (
    FakeClock,
    FakeDBChecker,
    make_color_crop,
    make_test_registry,
)

PLATE = "REENTRY-1"


def _registry_always_false():
    """A wired test registry whose DB probe always says 'outside', so any
    ``is_plate_inside`` True is attributable solely to the re-entry override."""
    clock = FakeClock()
    db = FakeDBChecker(mode="always_false")
    reg = make_test_registry(clock=clock, db_checker=db)
    return reg, clock, db


def test_reentry_within_grace_is_inside():
    """(a) exit -> newer entry within grace => inside, despite the DB (which
    still only has the old closed row) reporting the plate outside."""
    reg, clock, db = _registry_always_false()

    reg.register_anpr_event(PLATE, "entry")   # original visit
    clock.advance(10)
    reg.register_anpr_event(PLATE, "exit")    # car leaves; DB row -> closed
    clock.advance(30)
    reg.register_anpr_event(PLATE, "entry")   # genuine re-entry

    assert reg.has_recent_reentry(PLATE) is True
    assert reg.is_plate_inside(PLATE) is True
    # The DB probe was still consulted (override is layered on top, not a bypass).
    assert PLATE in db.calls


def test_entry_only_no_exit_is_not_inside():
    """(b1) an entry with no prior exit must NOT satisfy the override — with the
    DB reporting outside, the plate is outside (anti-ghost: no exit observed)."""
    reg, clock, db = _registry_always_false()

    reg.register_anpr_event(PLATE, "entry")

    assert reg.has_recent_reentry(PLATE) is False
    assert reg.is_plate_inside(PLATE) is False


def test_exit_after_entry_no_reentry_is_not_inside():
    """(b2) entry -> exit with NO re-entry: entry_at <= exit_at, so the override
    stays False. This is the exact missed-exit ghost the guard must keep
    evicting — a stale session here must not be resurrected."""
    reg, clock, db = _registry_always_false()

    reg.register_anpr_event(PLATE, "entry")
    clock.advance(10)
    reg.register_anpr_event(PLATE, "exit")

    assert reg.has_recent_reentry(PLATE) is False
    assert reg.is_plate_inside(PLATE) is False


def test_override_expires_after_grace():
    """(c) once more than REENTRY_DB_GRACE_SECONDS elapse since the re-entry the
    override lapses and the DB 'outside' verdict wins again (bounds the
    worst-case double-fault ghost to the grace window)."""
    reg, clock, db = _registry_always_false()

    reg.register_anpr_event(PLATE, "entry")
    clock.advance(10)
    reg.register_anpr_event(PLATE, "exit")
    clock.advance(5)
    reg.register_anpr_event(PLATE, "entry")   # re-entry

    assert reg.is_plate_inside(PLATE) is True
    clock.advance(reg.REENTRY_DB_GRACE_SECONDS + 1)
    assert reg.has_recent_reentry(PLATE) is False
    assert reg.is_plate_inside(PLATE) is False


def test_duplicate_reentry_refreshes_entry_time():
    """(d) a duplicate/coalesced re-entry read still refreshes the entry stamp,
    so the override tracks the latest read. Pins the stamp placement AHEAD of
    the coalesce / duplicate-ignore early-return paths in register_anpr_event."""
    reg, clock, db = _registry_always_false()

    reg.register_anpr_event(PLATE, "entry")
    clock.advance(10)
    reg.register_anpr_event(PLATE, "exit")
    clock.advance(5)
    reg.register_anpr_event(PLATE, "entry")   # first re-entry read
    clock.advance(2)
    reg.register_anpr_event(PLATE, "entry")   # duplicate-ignored read, same plate

    assert reg._last_anpr_entry_at[PLATE] == clock.now()
    assert reg.has_recent_reentry(PLATE) is True


def test_grace_boundary_is_inclusive():
    """(e) the window is inclusive: True at exactly GRACE seconds after the
    re-entry, False one second past it (guards against off-by-one if the
    constant is retuned)."""
    reg, clock, db = _registry_always_false()

    reg.register_anpr_event(PLATE, "entry")
    clock.advance(10)
    reg.register_anpr_event(PLATE, "exit")
    clock.advance(5)
    reg.register_anpr_event(PLATE, "entry")   # re-entry at t0

    clock.advance(reg.REENTRY_DB_GRACE_SECONDS)
    assert reg.has_recent_reentry(PLATE) is True   # exactly at the boundary
    clock.advance(1)
    assert reg.has_recent_reentry(PLATE) is False


def test_reentry_of_one_plate_does_not_leak_to_another():
    """A re-entry override is per-plate: an unrelated plate that only exited (no
    re-entry) is still reported outside."""
    reg, clock, db = _registry_always_false()
    other = "OTHER-2"

    reg.register_anpr_event(PLATE, "entry")
    clock.advance(10)
    reg.register_anpr_event(PLATE, "exit")
    reg.register_anpr_event(other, "entry")
    clock.advance(5)
    reg.register_anpr_event(other, "exit")     # other only exits, never returns
    clock.advance(5)
    reg.register_anpr_event(PLATE, "entry")    # PLATE re-enters

    assert reg.has_recent_reentry(PLATE) is True
    assert reg.has_recent_reentry(other) is False


def test_reentry_unblocks_confirm_and_slot_bind_while_db_still_closed():
    """End-to-end of the reported symptom. With the DB still reporting the plate
    outside (its freshest row is the old exit), a genuine re-entry now lets BOTH
    the B1 confirmation (``confirm_at_b1_entrance``) and the slot bind
    (``try_link_to_slot``) proceed — instead of the
    ``[try_link_to_slot] refused ... already exited ... skipping bind`` path.
    Mirrors the exit -> entry -> B1 -> slot sequence from the production log."""
    reg, clock, db = _registry_always_false()
    plate, cam, track = "REBIND-9", "CAM-03", 77

    # Original visit, then exit: the DB's freshest row is now 'closed'
    # (FakeDBChecker always_false stands in for that).
    reg.register_anpr_event(plate, "entry", timestamp=clock())
    clock.advance(10)
    reg.register_anpr_event(plate, "exit", timestamp=clock())

    # Genuine re-entry.
    clock.advance(20)
    reg.register_anpr_event(plate, "entry", timestamp=clock())

    # Drive the B1 confirmation for the re-entered car. Previously refused at
    # is_plate_inside ('already exited'); the re-entry override lets it through.
    candidate = reg.open_park_entry_candidate(cam, track)
    snapshot = make_color_crop(bgr=(180, 30, 90))
    reg.update_park_entry_candidate_snapshot(
        candidate.candidate_id, snapshot, quality_score=5.0
    )
    reg.bind_next_pending_anpr_to_candidate(candidate.candidate_id)
    confirmed = reg.confirm_at_b1_entrance(cam, track, snapshot, timestamp=clock())
    assert confirmed == plate

    # Slot bind now proceeds instead of the "skipping bind" refusal.
    bound = reg.try_link_to_slot(
        slot_id="B25",
        slot_name="B25",
        zone_id="Z-B2",
        zone_name="B2",
        camera_id=cam,
        floor="B2",
        track_id=track,
        timestamp=clock(),
        snapshot_path="snap.jpg",
    )
    assert bound == plate
    assert "B25" in reg._parked
    assert reg._parked["B25"].plate == plate
