"""
One-time migration: back-fill reservation_type / reserved_for on named slots.

Run once after deploying the reservation_type schema change:
    python migrate_parking_categories.py

The script is idempotent — it skips if any slot already has a non-general
reservation_type (meaning the migration already ran).

reserved_for must match the exact string stored in vehicles.title so that
_is_named_slot_vehicle_allowed() can authorise the slot owner.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# pyrefly: ignore [missing-import]
from sqlalchemy import create_engine, text

# ── DB URL ────────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    try:
        from src.config import load_config
        cfg = load_config()
        DATABASE_URL = cfg.database.url
    except Exception as e:
        print(f"[ERROR] Could not determine DATABASE_URL: {e}")
        print("Set the DATABASE_URL environment variable and retry.")
        sys.exit(1)

# ── Slot data ─────────────────────────────────────────────────────────────────
EMPLOYEE_SLOTS = [
    # (slot_id, reserved_for)   reserved_for must match vehicles.title exactly
    ("B10_CTO",     "CTO"),
    ("B1_CRO",      "CRO"),
    ("B3_CEO",      "CEO"),
    ("B5_GMIA",     "GMIA"),
    ("B6_Reserved", "Reserved"),
    ("B11_CFO",     "CFO"),
    ("B13_COO",     "COO"),
    ("B8_BDM",      "BDM"),
]

SPECIAL_SLOTS = [
    # (slot_id, new_slot_name)
    ("G1", "Special Needs"),
]

# ── Run ───────────────────────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    already = conn.execute(
        text("SELECT COUNT(*) FROM parking_slots WHERE reservation_type != 'GENERAL'")
    ).scalar()
    if already > 0:
        print(f"[SKIP] {already} slot(s) already have a non-GENERAL reservation_type — migration already applied.")
        sys.exit(0)

    for slot_id, reserved_for in EMPLOYEE_SLOTS:
        rows = conn.execute(
            text(
                "UPDATE parking_slots "
                "SET reservation_type = 'EMPLOYEE', reserved_for = :rf, is_violation_zone = 0 "
                "WHERE slot_id = :sid"
            ),
            {"rf": reserved_for, "sid": slot_id},
        ).rowcount
        status = "OK" if rows else "NOT FOUND"
        print(f"[{status}] {slot_id} → EMPLOYEE, reserved_for={reserved_for}")

    for slot_id, slot_name in SPECIAL_SLOTS:
        rows = conn.execute(
            text(
                "UPDATE parking_slots "
                "SET reservation_type = 'SPECIAL', slot_name = :sn, is_violation_zone = 0 "
                "WHERE slot_id = :sid"
            ),
            {"sn": slot_name, "sid": slot_id},
        ).rowcount
        status = "OK" if rows else "NOT FOUND"
        print(f"[{status}] {slot_id} → SPECIAL, slot_name='{slot_name}'")

print("[DONE] Migration complete.")
