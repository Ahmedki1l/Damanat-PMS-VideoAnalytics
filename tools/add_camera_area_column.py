"""
add_camera_area_column.py — one-off script to add the `area` column to the
cameras table (zoning) and optionally populate it from config.yaml.

The cameras table is owned by the API Gateway; this script just adds the
nullable `area` column to the shared database so cameras can carry their
parking area. Idempotent: safe to re-run.

Usage:
    python tools/add_camera_area_column.py             # add the column only
    python tools/add_camera_area_column.py --populate  # also set each camera's
                                                       # area from config.yaml
    python tools/add_camera_area_column.py --config config.yaml
"""

import argparse
import os
import sys

from sqlalchemy import inspect, text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config
from src.database import init_db


def main():
    parser = argparse.ArgumentParser(
        description="Add (and optionally populate) the cameras.area column."
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path.")
    parser.add_argument(
        "--populate",
        action="store_true",
        help="Set each camera's area from config.yaml after adding the column.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not config.database.url:
        print("[ERROR] No database URL in config (database.DATABASE_URL).")
        sys.exit(1)

    db = init_db(config.database.url)
    engine = db.engine
    inspector = inspect(engine)

    if "cameras" not in inspector.get_table_names():
        print("[ERROR] Table 'cameras' not found in the database.")
        sys.exit(1)

    columns = {col["name"] for col in inspector.get_columns("cameras")}
    if "area" in columns:
        print("[OK] Column cameras.area already exists — nothing to add.")
    else:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE cameras ADD area VARCHAR(50) NULL"))
        print("[OK] Added column cameras.area (VARCHAR(50) NULL).")

    if args.populate:
        zoned = [(c.id, c.area) for c in config.cameras if c.area]
        if not zoned:
            print("[WARN] No cameras have an 'area' in config.yaml — nothing to populate.")
        else:
            # camera_id format differs between config.yaml (e.g. CAM-03) and the
            # cameras table (e.g. Cam_03), so match on a normalised id.
            def norm(x):
                return (x or "").upper().replace("-", "").replace("_", "")

            with engine.connect() as conn:
                db_ids = [r[0] for r in conn.execute(text("SELECT camera_id FROM cameras")).fetchall()]
            db_by_norm = {norm(cid): cid for cid in db_ids}

            updated, missing = 0, []
            with engine.begin() as conn:
                for cfg_id, area in zoned:
                    actual = db_by_norm.get(norm(cfg_id))
                    if actual is None:
                        missing.append(cfg_id)
                        continue
                    conn.execute(
                        text("UPDATE cameras SET area = :area WHERE camera_id = :cid"),
                        {"area": area, "cid": actual},
                    )
                    updated += 1
            print(f"[OK] Populated area on {updated} camera(s) from config.yaml.")
            if missing:
                print(f"[WARN] No DB camera matched: {', '.join(missing)}")

    # Show the result
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT camera_id, floor, area FROM cameras ORDER BY area, camera_id")
        ).fetchall()
    print("\ncamera_id        floor    area")
    print("-" * 40)
    for camera_id, floor, area in rows:
        print(f"{(camera_id or ''):<16} {(floor or ''):<8} {area or ''}")


if __name__ == "__main__":
    main()
