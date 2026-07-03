"""
manage_areas.py — edit the zoning tables (`parking_areas`, `boundaries`) in the
live database.

`config.yaml` only *seeds* `parking_areas` on the first run; after that the
database is the source of truth. Use this tool to change areas afterwards.

Usage:
    # Show what is currently in the database
    python tools/manage_areas.py list
    python tools/manage_areas.py boundaries

    # Apply config.yaml's areas: block to the DB (upsert every YAML area).
    # This is how you push edits you made in config.yaml AFTER the first seed.
    python tools/manage_areas.py push

    # Add or update a single area (creates it if the id is new)
    python tools/manage_areas.py set --id B1-E --name "B1 Center" --floor B1 \
        --capacity 25 --adjacency "B1-F:15,B1-D:15"

    # Update just one field (others left unchanged)
    python tools/manage_areas.py set --id B1-E --capacity 28

    # Remove an area / a boundary
    python tools/manage_areas.py delete --id B1-E
    python tools/manage_areas.py delete-boundary --id ramp_C_to_RAMP

    # Relabel an existing boundary's area_from/area_to (e.g. after renaming or
    # splitting an area) without touching its camera_id or polygon
    python tools/manage_areas.py set-boundary --id "Cam-7 Bound2" --area-to RAMP-UP

    # Wipe parking_areas and re-seed it from config.yaml's areas: block
    python tools/manage_areas.py reseed --yes

Pass --config to point at a non-default config file (default: config.yaml).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config
from src.database import init_db
from src.model.parking_area import ParkingArea
from src.model.boundary import Boundary
from src.services.config_service import ensure_areas_initialized


def _parse_adjacency(text: str) -> dict:
    """Parse "B1-F:15,B1-D:15" -> {"B1-F": 15.0, "B1-D": 15.0}."""
    out = {}
    for pair in (text or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise SystemExit(f"[ERROR] bad adjacency item '{pair}' (want AREA:SECONDS)")
        area, secs = pair.rsplit(":", 1)
        out[area.strip()] = float(secs.strip())
    return out


def _open_session(config_path: str):
    config = load_config(config_path)
    if not config.database.url:
        raise SystemExit("[ERROR] No database URL in config. Set database.DATABASE_URL.")
    db = init_db(config.database.url)
    db.create_tables()  # ensure parking_areas / boundaries exist
    return config, db.SessionLocal()


def cmd_list(session, config):
    rows = session.query(ParkingArea).order_by(ParkingArea.floor, ParkingArea.area_id).all()
    if not rows:
        print("(no areas in the database — run 'push' or start the engine once to seed)")
        return
    print(f"{'area_id':<10} {'floor':<6} {'cap':>4} {'cams':>5}  adjacency")
    print("-" * 64)
    for r in rows:
        cams = len(config.cameras_in_area(r.area_id))
        adj = ", ".join(f"{k}:{v:g}" for k, v in (r.adjacency_json or {}).items())
        active = "" if r.is_active else "  [inactive]"
        print(f"{r.area_id:<10} {r.floor or '':<6} {r.capacity or 0:>4} {cams:>5}  {adj}{active}")


def cmd_boundaries(session, config):
    rows = session.query(Boundary).order_by(Boundary.camera_id, Boundary.boundary_id).all()
    if not rows:
        print("(no boundaries — draw them with draw_slots.py 'b' mode)")
        return
    print(f"{'boundary_id':<24} {'camera':<10} {'from':<8} -> {'to':<8}")
    print("-" * 60)
    for r in rows:
        print(f"{r.boundary_id:<24} {r.camera_id or '':<10} {r.area_from or '?':<8} -> {r.area_to or '?':<8}")


def cmd_push(session, config):
    """Upsert every area from config.yaml into parking_areas."""
    if not config.areas:
        print("[WARN] config.yaml has no areas: block — nothing to push.")
        return
    by_id = {r.area_id: r for r in session.query(ParkingArea).all()}
    added = updated = 0
    for area in config.areas:
        row = by_id.get(area.area_id)
        if row:
            row.name, row.floor, row.capacity = area.name, area.floor, area.capacity
            row.adjacency_json = dict(area.adjacency)
            row.is_active = True
            updated += 1
        else:
            session.add(ParkingArea(
                area_id=area.area_id, name=area.name, floor=area.floor,
                capacity=area.capacity, adjacency_json=dict(area.adjacency),
                is_active=True,
            ))
            added += 1
    session.commit()
    print(f"[OK] pushed config.yaml areas -> DB ({added} added, {updated} updated).")


def cmd_set(session, config, args):
    row = session.query(ParkingArea).filter(ParkingArea.area_id == args.id).first()
    created = row is None
    if created:
        row = ParkingArea(area_id=args.id, is_active=True, capacity=0, adjacency_json={})
        session.add(row)
    if args.name is not None:
        row.name = args.name
    if args.floor is not None:
        row.floor = args.floor
    if args.capacity is not None:
        row.capacity = args.capacity
    if args.adjacency is not None:
        row.adjacency_json = _parse_adjacency(args.adjacency)
    if args.active is not None:
        row.is_active = args.active == "true"
    session.commit()
    print(f"[OK] {'created' if created else 'updated'} area '{args.id}'.")


def cmd_delete(session, config, args):
    row = session.query(ParkingArea).filter(ParkingArea.area_id == args.id).first()
    if not row:
        print(f"[WARN] area '{args.id}' not found.")
        return
    session.delete(row)
    session.commit()
    print(f"[OK] deleted area '{args.id}'.")


def cmd_delete_boundary(session, config, args):
    row = session.query(Boundary).filter(Boundary.boundary_id == args.id).first()
    if not row:
        print(f"[WARN] boundary '{args.id}' not found.")
        return
    session.delete(row)
    session.commit()
    print(f"[OK] deleted boundary '{args.id}'.")


def cmd_set_boundary(session, config, args):
    """Relabel an existing boundary's area_from/area_to (e.g. after renaming
    an area) without touching its camera_id or polygon."""
    row = session.query(Boundary).filter(Boundary.boundary_id == args.id).first()
    if not row:
        print(f"[WARN] boundary '{args.id}' not found.")
        return
    if args.area_from is not None:
        row.area_from = args.area_from
    if args.area_to is not None:
        row.area_to = args.area_to
    session.commit()
    print(f"[OK] boundary '{args.id}' now {row.area_from} -> {row.area_to}.")


def cmd_reseed(session, config, args):
    if not args.yes:
        print("[ABORT] reseed wipes parking_areas. Re-run with --yes to confirm.")
        return
    n = session.query(ParkingArea).delete()
    session.commit()
    ensure_areas_initialized(session, config)  # re-seeds from config.yaml
    print(f"[OK] wiped {n} area(s) and re-seeded from config.yaml.")


def main():
    p = argparse.ArgumentParser(description="Edit zoning tables (parking_areas, boundaries).")
    p.add_argument("--config", default="config.yaml", help="Config file (default: config.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List areas in the database")
    sub.add_parser("boundaries", help="List boundaries in the database")
    sub.add_parser("push", help="Upsert config.yaml areas into the database")

    s = sub.add_parser("set", help="Add or update one area")
    s.add_argument("--id", required=True)
    s.add_argument("--name")
    s.add_argument("--floor")
    s.add_argument("--capacity", type=int)
    s.add_argument("--adjacency", help='e.g. "B1-F:15,B1-D:15"')
    s.add_argument("--active", choices=["true", "false"])

    d = sub.add_parser("delete", help="Delete one area")
    d.add_argument("--id", required=True)

    db = sub.add_parser("delete-boundary", help="Delete one boundary")
    db.add_argument("--id", required=True)

    sb = sub.add_parser("set-boundary", help="Relabel an existing boundary's area_from/area_to")
    sb.add_argument("--id", required=True, help="boundary_id")
    sb.add_argument("--area-from")
    sb.add_argument("--area-to")

    r = sub.add_parser("reseed", help="Wipe parking_areas and re-seed from config.yaml")
    r.add_argument("--yes", action="store_true", help="Confirm the wipe")

    args = p.parse_args()
    config, session = _open_session(args.config)
    try:
        if args.command == "list":
            cmd_list(session, config)
        elif args.command == "boundaries":
            cmd_boundaries(session, config)
        elif args.command == "push":
            cmd_push(session, config)
        elif args.command == "set":
            cmd_set(session, config, args)
        elif args.command == "delete":
            cmd_delete(session, config, args)
        elif args.command == "delete-boundary":
            cmd_delete_boundary(session, config, args)
        elif args.command == "set-boundary":
            cmd_set_boundary(session, config, args)
        elif args.command == "reseed":
            cmd_reseed(session, config, args)
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
