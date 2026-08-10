"""
calibrate_coverage_threshold.py — pick ``assigner.coverage_threshold`` from the
slot polygons you actually drew, before pointing a camera at it.

WHY THIS EXISTS
---------------
``assignment_mode: coverage`` decides slot membership by asking what fraction of
a vehicle's *axis-aligned detection box* lies inside the slot polygon. That
number has a ceiling the operator never sees: the polygons are perspective
trapezoids, and a detection box is an axis-aligned rectangle. A car that fills
its slot perfectly still produces a rectangle whose corners hang outside the
trapezoid, so its coverage tops out at roughly

    polygon_area / area_of_the_polygon's_own_bounding_box

For a sharply-angled slot that ceiling is well under 0.5. Set
``coverage_threshold`` above a slot's ceiling and that slot can never be
occupied by anything — it goes quietly, permanently VACANT. No error, no log.

This tool measures the ceiling for every slot in the database and reports how
many slots each candidate threshold would strand.

WHAT IT SIMULATES
-----------------
For occupancy factor ``s``, it shrinks each slot polygon toward its centroid by
``s`` and takes the axis-aligned bounds of the result — a stand-in for the
detection box of a car occupying that fraction of the slot. ``s=1.0`` is a
vehicle filling the slot kerb-to-kerb (the worst case for coverage, since the
box corners overhang most); smaller ``s`` is a compact car with room around it,
whose box sits further inside the polygon and scores higher.

Real detection boxes are noisier than this — the model over- and under-shoots,
and cars park off-centre. Treat the output as the ceiling to stay below, not as
the operating point. Confirm on footage with tools/show_occupied_slots.py.

USAGE
-----
    python tools/calibrate_coverage_threshold.py
    python tools/calibrate_coverage_threshold.py --config config.yaml
    python tools/calibrate_coverage_threshold.py --slots parking_slots.json
    python tools/calibrate_coverage_threshold.py --verbose   # per-slot table
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shapely.affinity import scale as shapely_scale
from shapely.geometry import Polygon, box as shapely_box

# Occupancy factors simulated, and the thresholds reported against them.
OCCUPANCY_FACTORS = (1.0, 0.9, 0.8, 0.7)
CANDIDATE_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


def _load_from_db(config_path):
    """Read (slot_id, polygon) from the DB named by ``config_path``."""
    import yaml
    from sqlalchemy import create_engine, text

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    url = (raw.get("database") or {}).get("DATABASE_URL")
    if not url:
        raise SystemExit(f"No database.DATABASE_URL in {config_path}")

    engine = create_engine(url)
    with engine.connect() as conn:
        rows = conn.execute(text("select slot_id, polygon from parking_slots")).fetchall()
    return [(r[0], r[1]) for r in rows]


def _load_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return [
        (e["id"], e.get("polygon"))
        for e in entries
        if e.get("type") != "virtual_line"
    ]


def _to_polygon(raw):
    """Parse a stored polygon into a valid Shapely Polygon, or None."""
    points = json.loads(raw) if isinstance(raw, str) else raw
    if not points or len(points) < 3:
        return None
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # self-intersecting hand-drawn polygons
    if polygon.is_empty or polygon.area <= 0:
        return None
    return polygon


def _coverage_ceiling(polygon, occupancy):
    """Coverage a car occupying ``occupancy`` of this slot would score."""
    vehicle = polygon if occupancy >= 1.0 else shapely_scale(
        polygon, xfact=occupancy, yfact=occupancy, origin="centroid"
    )
    x1, y1, x2, y2 = vehicle.bounds
    det_box = shapely_box(x1, y1, x2, y2)
    if det_box.area <= 0:
        return 0.0
    return det_box.intersection(polygon).area / det_box.area


def _percentile(sorted_values, fraction):
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(fraction * len(sorted_values)))
    return sorted_values[idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="config.yaml naming the DB")
    parser.add_argument("--slots", help="read polygons from a slots JSON instead of the DB")
    parser.add_argument("--verbose", action="store_true", help="print the per-slot table")
    args = parser.parse_args()

    raw_rows = _load_from_json(args.slots) if args.slots else _load_from_db(args.config)

    slots = []
    skipped = []
    for slot_id, raw in raw_rows:
        if str(slot_id).lower().endswith("roi") or str(slot_id).lower() == "roi":
            continue  # ROI polygons are not slots
        try:
            polygon = _to_polygon(raw)
        except Exception as exc:
            skipped.append((slot_id, str(exc)))
            continue
        if polygon is None:
            skipped.append((slot_id, "degenerate or <3 points"))
            continue
        slots.append((slot_id, polygon))

    if not slots:
        raise SystemExit("No usable slot polygons found.")

    print(f"Measured {len(slots)} slot polygons"
          + (f" ({len(skipped)} skipped)" if skipped else ""))
    for slot_id, why in skipped:
        print(f"  [SKIP] {slot_id}: {why}")

    ceilings = {}
    for occupancy in OCCUPANCY_FACTORS:
        ceilings[occupancy] = sorted(
            (_coverage_ceiling(p, occupancy), sid) for sid, p in slots
        )

    print("\nBest coverage a correctly-parked car can score, by how much of the")
    print("slot it fills (1.0 = fills it kerb-to-kerb, the worst case):\n")
    print(f"  {'fills':>6}  {'min':>6}  {'p10':>6}  {'median':>6}  {'worst slot':<20}")
    for occupancy in OCCUPANCY_FACTORS:
        values = [c for c, _ in ceilings[occupancy]]
        worst_id = ceilings[occupancy][0][1]
        print(f"  {occupancy:>6.1f}  {values[0]:>6.3f}  "
              f"{_percentile(values, 0.10):>6.3f}  "
              f"{_percentile(values, 0.50):>6.3f}  {worst_id:<20}")

    print("\nSlots that could NEVER reach a given coverage_threshold")
    print("(a stranded slot reads permanently VACANT, silently):\n")
    header = "  threshold  " + "  ".join(f"fills={o:.1f}" for o in OCCUPANCY_FACTORS)
    print(header)
    for threshold in CANDIDATE_THRESHOLDS:
        cells = []
        for occupancy in OCCUPANCY_FACTORS:
            stranded = sum(1 for c, _ in ceilings[occupancy] if c < threshold)
            cells.append(f"{stranded:>4}/{len(slots):<4}")
        print(f"  {threshold:>9.2f}  " + "  ".join(cells))

    worst_case = ceilings[1.0]
    safe = worst_case[0][0]
    print(f"\nHighest threshold that strands NO slot even when cars fill them: {safe:.3f}")
    print(f"Limiting slot: {worst_case[0][1]}")
    print("Leave headroom for detector jitter and off-centre parking: start")
    print(f"around {max(0.20, safe * 0.75):.2f} and confirm on footage.")

    if args.verbose:
        print("\nPer-slot ceiling (cars filling the slot), worst first:")
        for ceiling, slot_id in worst_case:
            print(f"  {ceiling:>6.3f}  {slot_id}")


if __name__ == "__main__":
    main()
