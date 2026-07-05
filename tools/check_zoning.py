"""Diagnose zoning coverage — the cross-floor ReID leak check.

Loads the roster exactly as the engine does (DB-first via config_service, YAML
fallback), builds the AreaRegistry, and flags every camera whose resolved area
is blank or points at an area_id that isn't defined. A blank area on a
slot-hosting camera means its match_global_session runs UNBOUNDED (legacy
all-sessions pool) → it can ReID-match a car on another floor → the B1<->B2
plate duplication.

Also lists the on-disk gallery so you can confirm folders are being seeded.

Run on the RUN MACHINE, from the project root with the venv active:

    .venv/Scripts/python.exe tools/check_zoning.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.zoning.area_registry import AreaRegistry


def main() -> None:
    cfg = load_config("config.yaml")

    # Mirror the engine's DB-first roster load so we see the SAME areas it uses.
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from src.services.config_service import (
            load_cameras_from_db,
            sync_areas_from_db,
        )

        eng = create_engine(cfg.database.url)
        db = sessionmaker(bind=eng)()
        try:
            sync_areas_from_db(db, cfg)      # areas: DB is authoritative
            load_cameras_from_db(db, cfg)    # cameras + their area column
        finally:
            db.close()
        print("[roster] loaded DB-first (same path as the engine).")
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[roster] DB merge unavailable ({exc!r}) — using YAML roster only.")

    areas = AreaRegistry(cfg)
    defined = set(areas.all_area_ids())
    print(f"[areas] zoning enabled={areas.enabled} | defined area_ids={sorted(defined)}\n")

    print(f"{'camera':<10} {'floor':<8} {'area':<12} verdict")
    print("-" * 60)
    leaks = []
    for cam in cfg.cameras:
        area = areas.area_for_camera(cam.id)
        floor = (cam.floor or "").strip()
        is_gf = floor.lower().startswith("ground")
        if not area:
            verdict = "OK (ground/un-zoned)" if is_gf else ">>> UNMAPPED — UNBOUNDED MATCH (LEAK)"
            if not is_gf:
                leaks.append(cam.id)
        elif area not in defined:
            verdict = f">>> area '{area}' NOT in areas: — matches NOTHING"
            leaks.append(cam.id)
        else:
            verdict = f"OK (floor={areas.floor(area)})"
        print(f"{cam.id:<10} {floor:<8} {area:<12} {verdict}")

    print()
    if leaks:
        print(f"[!] {len(leaks)} camera(s) misconfigured: {leaks}")
        print("    Blank area on a non-ground camera = cross-floor ReID leak.")
    else:
        print("[OK] every non-ground camera maps to a defined area.")

    # Gallery presence
    base = getattr(cfg.output, "snapshot_base_dir", "vehicle_images") or "vehicle_images"
    gdir = os.path.join(base, "gallery")
    print(f"\n[gallery] {gdir}: ", end="")
    if os.path.isdir(gdir):
        folders = [e.name for e in os.scandir(gdir) if e.is_dir()]
        print(f"{len(folders)} plate folder(s): {folders[:20]}")
    else:
        print("DOES NOT EXIST (no folder ever seeded, or wrong base dir)")


if __name__ == "__main__":
    main()
