"""
Migrate parking slot storage into the existing database.

What this script does:
  1. Ensures the `parking_slots` table has the new columns:
     - camera_id
     - slot_type
     - last_snapshot_path
  2. Optionally imports legacy JSON slot files from config.yaml into the DB.

Examples:
  python migrate_parking_slots_to_db.py
  python migrate_parking_slots_to_db.py --import-json
  python migrate_parking_slots_to_db.py --import-json --camera CAM_03
  python migrate_parking_slots_to_db.py --import-json --force
"""

import argparse
import os
import sys

from src.config import load_config
from src.database import init_db
from src.repositories import ParkingSlotRepository
from src.services.parking_service import bootstrap_camera_slots_from_json


def migrate_camera_slots(session, camera, force: bool = False) -> None:
    existing_rows = ParkingSlotRepository.filter_camera_slots(session, camera.id)
    if existing_rows and not force:
        print(
            f"[SKIP] {camera.id}: already has {len(existing_rows)} row(s) in parking_slots. "
            "Use --force to re-import from JSON."
        )
        return

    if not camera.slots_file:
        print(f"[SKIP] {camera.id}: no slots_file configured.")
        return

    if not os.path.exists(camera.slots_file):
        print(f"[SKIP] {camera.id}: JSON file not found -> {camera.slots_file}")
        return

    if force and existing_rows:
        print(f"[INFO] {camera.id}: re-importing from {camera.slots_file}")
    else:
        print(f"[INFO] {camera.id}: importing from {camera.slots_file}")

    migrated = bootstrap_camera_slots_from_json(
        session,
        camera_id=camera.id,
        floor=camera.floor,
        slots_file=camera.slots_file,
        default_zone_id=camera.name,
        default_zone_name=camera.name,
    )
    if migrated:
        fresh_rows = ParkingSlotRepository.filter_camera_slots(session, camera.id)
        print(f"[OK] {camera.id}: {len(fresh_rows)} row(s) now stored in parking_slots")
    else:
        print(f"[WARN] {camera.id}: nothing was imported")


def main():
    parser = argparse.ArgumentParser(description="Migrate parking slot storage into the database.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml).",
    )
    parser.add_argument(
        "--import-json",
        action="store_true",
        help="Import legacy slots/*.json files into parking_slots.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Only import one camera from config, e.g. CAM_03.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-import a camera even if it already has rows in parking_slots.",
    )
    args = parser.parse_args()

    print("[INFO] Loading configuration...")
    config = load_config(args.config)
    if not config.database.url:
        print("[ERROR] DATABASE_URL not found in config.")
        sys.exit(1)

    print("[INFO] Initializing database...")
    db_manager = init_db(config.database.url)

    try:
        db_manager.create_tables()
        print("[OK] Database schema is up to date.")
    except Exception as exc:
        print(f"[ERROR] Failed to update schema: {exc}")
        sys.exit(1)

    if not args.import_json:
        print("[INFO] Schema migration only. Use --import-json to import legacy slot JSON files.")
        return

    cameras = config.cameras
    if args.camera:
        cameras = [camera for camera in cameras if camera.id == args.camera]
        if not cameras:
            print(f"[ERROR] Camera '{args.camera}' not found in config.")
            sys.exit(1)

    session = db_manager.SessionLocal()
    try:
        for camera in cameras:
            migrate_camera_slots(session, camera, force=args.force)
    except Exception as exc:
        session.rollback()
        print(f"[ERROR] Migration failed: {exc}")
        sys.exit(1)
    finally:
        session.close()

    print("[SUCCESS] Parking slot migration completed.")


if __name__ == "__main__":
    main()
