"""Show occupied parking slots and their plate numbers — live.

Read-only companion to the running engine. It queries the SAME database
`main.py` writes to (`parking_slots.is_available` / `current_plate`), so it
reflects live occupancy while the engine is running without touching it.

Usage (from the project root, venv active):

    # One-shot snapshot of every occupied slot + its plate
    python tools/show_occupied_slots.py

    # Live view, refreshing every 2s (Ctrl-C to stop)
    python tools/show_occupied_slots.py --watch 2

    # Filter by floor / camera
    python tools/show_occupied_slots.py --floor B1
    python tools/show_occupied_slots.py --camera CAM-04

    # Machine-readable output (one JSON array), e.g. to pipe elsewhere
    python tools/show_occupied_slots.py --json

    # Only slots that actually have a resolved plate
    python tools/show_occupied_slots.py --plated-only
"""

import argparse
import json
import os
import sys
import time

# Allow running as `python tools/show_occupied_slots.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import load_config
from src.model import ParkingSlot, SlotStatus


def _fmt_time(dt):
    if not dt:
        return "-"
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(dt)


def collect_occupied(session, floor=None, camera=None, plated_only=False):
    """Return a list of dicts describing every occupied slot + its plate."""
    query = session.query(ParkingSlot).filter(ParkingSlot.is_available == False)  # noqa: E712
    if floor:
        query = query.filter(ParkingSlot.floor == floor)
    if camera:
        query = query.filter(ParkingSlot.camera_id == camera)

    slots = query.all()

    rows = []
    for slot in slots:
        plate = slot.current_plate
        plate_source = "locked" if plate else None

        # Occupied but no frozen plate yet (identity not resolved). Fall back to
        # the most recent slot_status row, which may carry a provisional plate.
        if not plate:
            latest = (
                session.query(SlotStatus)
                .filter(SlotStatus.slot_id == slot.slot_id)
                .order_by(SlotStatus.time.desc())
                .first()
            )
            if latest and latest.plate_number:
                plate = latest.plate_number
                plate_source = "status"

        if plated_only and not plate:
            continue

        rows.append(
            {
                "floor": slot.floor or "-",
                "camera": slot.camera_id or "-",
                "slot": slot.slot_name or slot.slot_id,
                "slot_id": slot.slot_id,
                "plate": plate or None,
                "plate_source": plate_source,
                "locked": bool(getattr(slot, "plate_locked", False)),
                "confidence": float(getattr(slot, "plate_confidence", 0.0) or 0.0),
                "since": getattr(slot, "plate_locked_at", None),
            }
        )

    rows.sort(key=lambda r: (r["floor"], r["camera"], r["slot"]))
    return rows


def render_table(rows):
    if not rows:
        return "No occupied slots."

    headers = ["Floor", "Camera", "Slot", "Plate", "Lock", "Conf", "Since"]

    def to_cells(r):
        plate = r["plate"] or "(pending)"
        return [
            r["floor"],
            r["camera"],
            r["slot"],
            plate,
            "yes" if r["locked"] else "",
            f"{r['confidence']:.2f}" if r["confidence"] else "",
            _fmt_time(r["since"]),
        ]

    table = [headers] + [to_cells(r) for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]

    def fmt_row(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines += [fmt_row(c) for c in [to_cells(r) for r in rows]]

    plated = sum(1 for r in rows if r["plate"])
    lines.append("")
    lines.append(f"{len(rows)} occupied slot(s), {plated} with a plate.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Show occupied parking slots and their plate numbers (live)."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--floor", help="Filter by floor (e.g. B1)")
    parser.add_argument("--camera", help="Filter by camera id (e.g. CAM-04)")
    parser.add_argument(
        "--plated-only",
        action="store_true",
        help="Only show occupied slots that have a resolved plate",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    parser.add_argument(
        "--watch",
        nargs="?",
        type=float,
        const=2.0,
        default=None,
        metavar="SECONDS",
        help="Refresh continuously every SECONDS (default 2). Ctrl-C to stop.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    db_url = config.database.url
    if not db_url:
        print("[ERROR] No database URL in config (database.DATABASE_URL).", file=sys.stderr)
        sys.exit(1)

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    def one_pass():
        session = Session()
        try:
            rows = collect_occupied(
                session,
                floor=args.floor,
                camera=args.camera,
                plated_only=args.plated_only,
            )
        finally:
            session.close()

        if args.json:
            payload = [
                {**r, "since": _fmt_time(r["since"]) if r["since"] else None} for r in rows
            ]
            print(json.dumps(payload, indent=2))
        else:
            print(render_table(rows))

    if args.watch is None:
        one_pass()
        return

    interval = max(0.5, args.watch)
    try:
        while True:
            # Clear screen (ANSI) then reprint.
            print("\033[2J\033[H", end="")
            print(f"Occupied slots — refreshing every {interval:g}s (Ctrl-C to stop)")
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            print()
            one_pass()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
