"""
sync_geometry.py — move authored zoning geometry between databases.

The geometry you draw with ``draw_slots.py`` (slot polygons, area-crossing
boundaries) and the areas you define all live in three tables:

    parking_areas   the area defs (capacity, adjacency)
    parking_slots   the slot polygons you drew
    boundaries      the area-to-area crossing polygons you drew

This tool dumps those rows to a portable JSON file on the machine where you
authored them (the testing PC), and re-applies them on another machine (the
production server). The camera roster is NOT here — it comes from the Gateway
/ config.yaml, so it is left untouched.

Typical workflow
----------------
    # 1) On the TESTING PC (its config.yaml points at the test DB):
    python tools/sync_geometry.py export --out b1_geometry.json --floor B1

    # 2) Copy b1_geometry.json to the production server.

    # 3) On PRODUCTION (its config.yaml points at the prod DB):
    python tools/sync_geometry.py seed --in geometry.json
    #    add --dry-run first to preview, or --mode replace to wipe+load the floor

Which DB it talks to is decided by the config.yaml you point at (``--config``),
or override directly with ``--db-url``. Seeding is an idempotent UPSERT by
primary key, so re-running is safe. Runtime-only slot fields (``is_available``,
``last_snapshot_path``) are reset on seed — prod keeps its own live state and
paths, not the testing PC's.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from sqlalchemy import select

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config
from src.database import init_db
from src.model.parking_area import ParkingArea
from src.model.parkingslot import ParkingSlot
from src.model.boundary import Boundary
from src.model.slots_status import SlotStatus
from src.model.alert import Alert

# Export order matters for seeding: areas first (slots/boundaries reference
# area ids), then the geometry that points at them.
TABLES = [
    ("parking_areas", ParkingArea, "floor"),
    ("parking_slots", ParkingSlot, "floor"),
    ("boundaries", Boundary, "floor"),
]

# Slot columns that are live runtime state, not authored geometry. Reset on
# seed so we never carry the testing PC's occupancy / snapshot paths into prod.
# The plate-lock binding (current_plate/plate_confidence/plate_locked/
# plate_locked_at) is likewise live state, not geometry — carrying it over both
# leaks the testing PC's occupancy AND breaks the seed, because json.dump turns
# plate_locked_at into a microsecond string the SQL Server DateTime column can't
# convert back on merge.
SLOT_RUNTIME_RESET = {
    "is_available": True,
    "last_snapshot_path": None,
    "current_plate": None,
    "plate_confidence": 0.0,
    "plate_locked": False,
    "plate_locked_at": None,
}

SCHEMA_VERSION = 1


def _row_to_dict(row) -> dict:
    """Serialise a mapped row to a plain dict (JSON columns come back as
    Python list/dict already, so they dump cleanly)."""
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


def _open_session(config_path: str, db_url: str | None):
    config = load_config(config_path)
    url = db_url or config.database.url
    if not url:
        raise SystemExit(
            "[ERROR] No database URL. Set database.DATABASE_URL in the config "
            "or pass --db-url."
        )
    db = init_db(url)
    db.create_tables()  # ensure the three tables exist on the target
    return config, db.SessionLocal(), url


def _safe_db_label(url: str) -> str:
    """A host/db label with the password stripped, for logging."""
    try:
        from sqlalchemy.engine.url import make_url
        u = make_url(url)
        return f"{u.host or '?'}/{u.database or '?'}"
    except Exception:
        return "?"


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def cmd_export(session, url, args):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_db": _safe_db_label(url),
        "floor_filter": args.floor,
        "tables": {},
    }
    for name, model, floor_col in TABLES:
        q = session.query(model)
        if args.floor and floor_col:
            q = q.filter(getattr(model, floor_col) == args.floor)
        rows = [_row_to_dict(r) for r in q.all()]
        payload["tables"][name] = rows
        print(f"  {name:<16} {len(rows):>5} row(s)")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    print(f"[OK] exported -> {args.out}  (source: {payload['source_db']}"
          f"{', floor=' + args.floor if args.floor else ''})")


# --------------------------------------------------------------------------- #
# seed
# --------------------------------------------------------------------------- #
def _filter_cols(model, data: dict) -> dict:
    """Keep only keys that are real columns on the model (tolerates schema
    drift between the export and the target)."""
    cols = {c.name for c in model.__table__.columns}
    return {k: v for k, v in data.items() if k in cols}


def _clear_slot_dependents(session, floor: str | None):
    """Before a ``replace`` wipe of parking_slots, detach the child rows that
    FK-reference it, or SQL Server rejects the DELETE (FK constraint 547).

    - slot_status: ephemeral per-slot status log — deleted outright.
    - alerts: historical records we keep; the FK is nullable and each alert
      carries its own ``slot_number`` string, so we NULL slot_id (detach) rather
      than destroy the alert.

    When ``floor`` is given, only slots on that floor are affected.
    """
    slot_scope = select(ParkingSlot.slot_id)
    if floor:
        slot_scope = slot_scope.where(ParkingSlot.floor == floor)

    ss = session.query(SlotStatus).filter(SlotStatus.slot_id.in_(slot_scope))
    ss_deleted = ss.delete(synchronize_session=False)

    al = session.query(Alert).filter(Alert.slot_id.in_(slot_scope))
    al_detached = al.update({Alert.slot_id: None}, synchronize_session=False)

    if ss_deleted or al_detached:
        print(f"  {'slot_status':<16} -{ss_deleted} deleted, "
              f"{al_detached} alert(s) detached")


def cmd_seed(session, url, args):
    with open(args.infile, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if payload.get("schema_version") != SCHEMA_VERSION:
        print(f"[WARN] export schema_version={payload.get('schema_version')} "
              f"!= tool {SCHEMA_VERSION}; proceeding best-effort.")

    print(f"[seed] target: {_safe_db_label(url)}   source dump: "
          f"{payload.get('source_db', '?')}  ({payload.get('exported_at', '?')})")
    if args.mode == "replace":
        print("[seed] mode=replace — existing rows"
              f"{' for floor ' + args.floor if args.floor else ''} will be DELETED first.")
    if args.dry_run:
        print("[seed] DRY RUN — no changes will be committed.")

    totals = {}
    for name, model, floor_col in TABLES:
        rows = payload["tables"].get(name, [])
        if args.floor:
            rows = [r for r in rows if r.get(floor_col) == args.floor]

        # First-run guard (for container start-up): skip any table that already
        # has rows, so a restart/redeploy never overwrites live data.
        if getattr(args, "if_empty", False):
            existing = session.query(model).count()
            if existing:
                print(f"  {name:<16} skipped ({existing} existing row(s))")
                totals[name] = 0
                continue

        if args.mode == "replace":
            # parking_slots is FK-referenced by slot_status / alerts; clear those
            # child rows first or the bulk DELETE hits a constraint violation.
            if model is ParkingSlot:
                _clear_slot_dependents(session, args.floor)
            dq = session.query(model)
            if args.floor and floor_col:
                dq = dq.filter(getattr(model, floor_col) == args.floor)
            deleted = dq.delete(synchronize_session=False)
            print(f"  {name:<16} -{deleted} deleted")

        for raw in rows:
            data = _filter_cols(model, raw)
            if model is ParkingSlot:
                data.update(SLOT_RUNTIME_RESET)
            session.merge(model(**data))  # insert-or-update by primary key

        totals[name] = len(rows)
        print(f"  {name:<16} {len(rows):>5} upserted")

    if args.dry_run:
        session.rollback()
        print("[OK] dry run complete — rolled back, nothing written.")
    else:
        session.commit()
        print(f"[OK] seeded {sum(totals.values())} row(s) into "
              f"{_safe_db_label(url)}.")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Export/seed authored zoning geometry (areas, slots, boundaries).")
    p.add_argument("--config", default="config.yaml",
                   help="Config file picking the DB (default: config.yaml)")
    p.add_argument("--db-url", default=None,
                   help="Override the DB URL from --config (talk to either DB directly)")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("export", help="Dump geometry from this DB to a JSON file")
    e.add_argument("--out", required=True, help="Output JSON path")
    e.add_argument("--floor", default=None, help="Only this floor, e.g. B1")

    s = sub.add_parser("seed", help="Load a JSON dump into this DB (upsert)")
    s.add_argument("--in", dest="infile", required=True, help="Input JSON path")
    s.add_argument("--floor", default=None, help="Only seed this floor, e.g. B1")
    s.add_argument("--mode", choices=["upsert", "replace"], default="upsert",
                   help="upsert (default) merges by PK; replace wipes the floor first")
    s.add_argument("--dry-run", action="store_true",
                   help="Preview counts, then roll back without writing")
    s.add_argument("--if-empty", action="store_true",
                   help="Seed only tables that are currently empty (first-run "
                        "guard for container startup; never clobbers live data)")

    args = p.parse_args()
    config, session, url = _open_session(args.config, args.db_url)
    try:
        if args.command == "export":
            cmd_export(session, url, args)
        elif args.command == "seed":
            cmd_seed(session, url, args)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
