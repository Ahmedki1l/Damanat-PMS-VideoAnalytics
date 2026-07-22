import json
import os
import re
from typing import Iterable, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.model import ParkingSlot
from src.models.slot import ParkingSlot as RuntimeParkingSlot
from src.models.slot import build_polygon
from src.repositories import ParkingSlotRepository
from src.schemas import ParkingSlotUpdate

SLOT_TYPE_PARKING = "parking"
SLOT_TYPE_SPECIAL_ZONE = "special_zone"
SLOT_TYPE_ROI = "roi"
SLOT_TYPE_BOUNDARY = "boundary"  # zoning: area-to-area crossing polygon


def get_slot_or_404(db: Session, slot_id: str):
    slot = ParkingSlotRepository.get_by_id(db, slot_id)
    if not slot:
        raise HTTPException(status_code=404, detail=f"Slot {slot_id} not found")
    return slot


def classify_slot_type(slot_id: str) -> str:
    normalized = (slot_id or "").strip()
    lowered = normalized.lower()
    if lowered == "roi" or lowered.endswith("__roi"):
        return SLOT_TYPE_ROI
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if any(
        marker in compact
        for marker in ("parkentry", "b1entry", "entrance", "entrence")
    ):
        return SLOT_TYPE_SPECIAL_ZONE
    return SLOT_TYPE_PARKING


def _entry_slot_type(entry) -> str:
    """Resolve a slot entry's type, honouring an explicit ``type`` field
    (set by the boundary drawing mode) before falling back to id-based
    classification."""
    explicit = None
    if hasattr(entry, "get"):
        explicit = entry.get("type")
    if explicit == SLOT_TYPE_BOUNDARY:
        return SLOT_TYPE_BOUNDARY
    return classify_slot_type(_entry_slot_id(entry))


def _entry_area_from(entry) -> Optional[str]:
    if hasattr(entry, "get"):
        return entry.get("area_from") or None
    return getattr(entry, "area_from", None) or None


def _entry_area_to(entry) -> Optional[str]:
    if hasattr(entry, "get"):
        return entry.get("area_to") or None
    return getattr(entry, "area_to", None) or None


def _normalize_polygon_points(points: Iterable[Iterable[float]]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in points]


def _resolve_db_slot_id(camera_id: str, slot_id: str, slot_type: str) -> str:
    if slot_type == SLOT_TYPE_ROI:
        return f"{camera_id}__roi"
    return slot_id


def _entry_polygon(entry) -> list[list[float]]:
    if hasattr(entry, "polygon"):
        return _normalize_polygon_points(list(entry.polygon.exterior.coords)[:-1])
    return _normalize_polygon_points(entry.get("polygon", []))


def _entry_slot_id(entry) -> str:
    if hasattr(entry, "id"):
        return entry.id
    return entry.get("id", "")


def _entry_slot_name(entry) -> str:
    if hasattr(entry, "label"):
        return entry.label or entry.id
    return entry.get("label") or entry.get("id", "")


def _entry_zone_id(entry, default_zone_id: str) -> Optional[str]:
    if hasattr(entry, "zone_id"):
        return entry.zone_id or default_zone_id or None
    return entry.get("zone_id") or default_zone_id or None


def _entry_zone_name(entry, default_zone_name: str, zone_id: Optional[str]) -> str:
    if hasattr(entry, "zone_name"):
        return entry.zone_name or default_zone_name or zone_id or entry.label or entry.id
    return (
        entry.get("zone_name")
        or default_zone_name
        or zone_id
        or entry.get("label")
        or entry.get("id", "")
    )


def _db_row_to_runtime_slot(
    row: ParkingSlot,
    ref_resolution=None,
    actual_resolution=None,
) -> RuntimeParkingSlot:
    polygon = build_polygon(
        row.polygon or [],
        ref_resolution=ref_resolution,
        actual_resolution=actual_resolution,
    )
    return RuntimeParkingSlot(
        id=row.slot_id,
        polygon=polygon,
        label=row.slot_name or row.slot_id,
        zone_id=row.zone_id or "",
        zone_name=row.zone_name or row.slot_name or row.slot_id,
    )


def load_camera_slots(
    db: Session,
    camera_id: str,
    ref_resolution=None,
    actual_resolution=None,
):
    rows = ParkingSlotRepository.filter_camera_slots(db, camera_id)
    parking_slots = []
    special_zones = []
    roi_polygon = None

    for row in rows:
        if not row.polygon or len(row.polygon) < 3:
            continue

        slot_type = row.slot_type or classify_slot_type(row.slot_id)
        polygon = build_polygon(
            row.polygon,
            ref_resolution=ref_resolution,
            actual_resolution=actual_resolution,
        )
        if slot_type == SLOT_TYPE_ROI:
            roi_polygon = polygon
            continue

        runtime_slot = RuntimeParkingSlot(
            id=row.slot_id,
            polygon=polygon,
            label=row.slot_name or row.slot_id,
            zone_id=row.zone_id or "",
            zone_name=row.zone_name or row.slot_name or row.slot_id,
        )
        if slot_type == SLOT_TYPE_SPECIAL_ZONE:
            special_zones.append(runtime_slot)
        else:
            parking_slots.append(runtime_slot)

    # Boundaries live in their own table (zoning), not in parking_slots.
    boundaries = load_camera_boundaries(
        db, camera_id, ref_resolution=ref_resolution, actual_resolution=actual_resolution
    )

    return parking_slots, special_zones, roi_polygon, boundaries


def load_camera_boundaries(
    db: Session,
    camera_id: str,
    ref_resolution=None,
    actual_resolution=None,
):
    """Load a camera's area-to-area boundary polygons from the ``boundaries``
    table as :class:`BoundaryZone` instances (empty list when none)."""
    from src.model.boundary import Boundary
    from src.zoning.boundary_detector import BoundaryZone

    rows = db.query(Boundary).filter(Boundary.camera_id == camera_id).all()
    result = []
    for row in rows:
        if not row.polygon or len(row.polygon) < 3:
            continue
        polygon = build_polygon(
            row.polygon,
            ref_resolution=ref_resolution,
            actual_resolution=actual_resolution,
        )
        result.append(
            BoundaryZone(
                id=row.boundary_id,
                polygon=polygon,
                area_from=row.area_from or "",
                area_to=row.area_to or "",
            )
        )
    return result


def sync_camera_boundaries(
    db: Session,
    camera_id: str,
    floor: str,
    boundary_entries: list,
):
    """Upsert a camera's boundary polygons into the ``boundaries`` table.

    Mirrors :func:`sync_camera_slot_definitions` but for boundaries. Entries are
    editor dicts carrying ``type="boundary"`` + ``area_from``/``area_to``;
    non-boundary entries are ignored. Boundaries no longer present for the
    camera are deleted."""
    from src.model.boundary import Boundary

    desired: dict[str, dict] = {}
    for entry in boundary_entries:
        if _entry_slot_type(entry) != SLOT_TYPE_BOUNDARY:
            continue
        polygon_json = _entry_polygon(entry)
        if len(polygon_json) < 3:
            continue
        bid = _entry_slot_id(entry)
        desired[bid] = {
            "boundary_id": bid,
            "camera_id": camera_id,
            "floor": floor,
            "polygon": polygon_json,
            "area_from": _entry_area_from(entry),
            "area_to": _entry_area_to(entry),
        }

    existing_rows = db.query(Boundary).filter(Boundary.camera_id == camera_id).all()
    existing_by_id = {row.boundary_id: row for row in existing_rows}

    for bid, payload in desired.items():
        existing = existing_by_id.get(bid)
        if existing:
            existing.camera_id = payload["camera_id"]
            existing.floor = payload["floor"]
            existing.polygon = payload["polygon"]
            existing.area_from = payload["area_from"]
            existing.area_to = payload["area_to"]
        else:
            db.add(Boundary(**payload))

    desired_ids = set(desired.keys())
    for existing in existing_rows:
        if existing.boundary_id not in desired_ids:
            db.delete(existing)

    db.commit()


def sync_camera_slot_definitions(
    db: Session,
    camera_id: str,
    floor: str,
    slot_entries: list,
    managed_slot_types: tuple[str, ...],
    default_zone_id: str = "",
    default_zone_name: str = "",
):
    desired_rows: dict[str, dict] = {}

    for entry in slot_entries:
        slot_id = _entry_slot_id(entry)
        slot_type = _entry_slot_type(entry)
        if slot_type not in managed_slot_types:
            continue

        polygon_json = _entry_polygon(entry)
        if len(polygon_json) < 3:
            continue

        zone_id = _entry_zone_id(entry, default_zone_id)
        zone_name = _entry_zone_name(entry, default_zone_name, zone_id)
        db_slot_id = _resolve_db_slot_id(camera_id, slot_id, slot_type)
        desired_rows[db_slot_id] = {
            "slot_id": db_slot_id,
            "slot_name": _entry_slot_name(entry),
            "camera_id": camera_id,
            "floor": floor,
            "zone_id": zone_id,
            "zone_name": zone_name,
            "slot_type": slot_type,
            "polygon": polygon_json,
            "is_violation_zone": "violation" in slot_id.lower(),
        }

    existing_rows = ParkingSlotRepository.filter_camera_slots_by_types(
        db,
        camera_id,
        managed_slot_types,
    )
    existing_by_id = {row.slot_id: row for row in existing_rows}

    for slot_id, payload in desired_rows.items():
        existing = existing_by_id.get(slot_id) or ParkingSlotRepository.get_by_id(db, slot_id)
        if existing:
            existing.slot_name = payload["slot_name"]
            existing.camera_id = payload["camera_id"]
            existing.floor = payload["floor"]
            existing.zone_id = payload["zone_id"]
            existing.zone_name = payload["zone_name"]
            existing.slot_type = payload["slot_type"]
            existing.polygon = payload["polygon"]
            # Only set is_violation_zone when the slot is actually a violation zone.
            # Employee/special slots must NOT be flagged as violation zones — their
            # reservation_type is the source of truth for restriction behaviour.
            if existing.reservation_type == "GENERAL":
                existing.is_violation_zone = bool(
                    existing.is_violation_zone or payload["is_violation_zone"]
                )
            # reservation_type and reserved_for are managed via API — never overwrite on sync
        else:
            db.add(
                ParkingSlot(
                    slot_id=payload["slot_id"],
                    slot_name=payload["slot_name"],
                    camera_id=payload["camera_id"],
                    floor=payload["floor"],
                    zone_id=payload["zone_id"],
                    zone_name=payload["zone_name"],
                    slot_type=payload["slot_type"],
                    polygon=payload["polygon"],
                    is_available=True,
                    is_violation_zone=payload["is_violation_zone"],
                )
            )

    desired_ids = set(desired_rows.keys())
    for existing in existing_rows:
        if existing.slot_id not in desired_ids:
            db.delete(existing)

    db.commit()


def save_camera_roi(
    db: Session,
    camera_id: str,
    floor: str,
    polygon_points: list[list[float]],
):
    sync_camera_slot_definitions(
        db=db,
        camera_id=camera_id,
        floor=floor,
        slot_entries=[
            {
                "id": "roi",
                "label": "Global ROI Mask",
                "polygon": polygon_points,
                "zone_id": camera_id,
                "zone_name": "Global ROI Mask",
            }
        ],
        managed_slot_types=(SLOT_TYPE_ROI,),
        default_zone_id=camera_id,
        default_zone_name="Global ROI Mask",
    )


def bootstrap_camera_slots_from_json(
    db: Session,
    camera_id: str,
    floor: str,
    slots_file: str,
    default_zone_id: str = "",
    default_zone_name: str = "",
) -> bool:
    if not slots_file or not os.path.exists(slots_file):
        return False

    with open(slots_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    sync_camera_slot_definitions(
        db=db,
        camera_id=camera_id,
        floor=floor,
        slot_entries=raw_data,
        managed_slot_types=(SLOT_TYPE_PARKING, SLOT_TYPE_SPECIAL_ZONE, SLOT_TYPE_ROI),
        default_zone_id=default_zone_id,
        default_zone_name=default_zone_name,
    )
    return True


def update_slot_partial(db: Session, slot_id: str, data: ParkingSlotUpdate):
    existing = get_slot_or_404(db, slot_id)
    for key, value in data.dict(exclude_unset=True).items():
        setattr(existing, key, value)
    ParkingSlotRepository.update(db)
    return existing
