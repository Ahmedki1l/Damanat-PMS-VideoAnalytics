import json
import os
import sys
from tempfile import NamedTemporaryFile

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append(os.path.abspath("."))

from src.database import Base
from src.repositories import ParkingSlotRepository
from src.services.parking_service import (
    SLOT_TYPE_PARKING,
    SLOT_TYPE_SPECIAL_ZONE,
    bootstrap_camera_slots_from_json,
    load_camera_slots,
    save_camera_roi,
    sync_camera_slot_definitions,
)


def make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


def test_camera_slot_storage_loads_by_type_and_camera():
    session = make_session()
    try:
        sync_camera_slot_definitions(
            db=session,
            camera_id="CAM_03",
            floor="B1",
            slot_entries=[
                {"id": "B8_BDM", "label": "B8_BDM", "polygon": [[10, 10], [20, 10], [20, 20], [10, 20]]},
                {"id": "B1_Entrence", "label": "B1_Entrence", "polygon": [[30, 30], [40, 30], [40, 40], [30, 40]]},
            ],
            managed_slot_types=(SLOT_TYPE_PARKING, SLOT_TYPE_SPECIAL_ZONE),
            default_zone_id="CAM_03",
            default_zone_name="CAM_03",
        )
        save_camera_roi(
            session,
            camera_id="CAM_03",
            floor="B1",
            polygon_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
        )

        parking_slots, special_zones, roi_polygon, boundaries = load_camera_slots(
            session,
            camera_id="CAM_03",
            ref_resolution=(640, 360),
            actual_resolution=(1280, 720),
        )
        camera_rows = ParkingSlotRepository.filter_camera_slots(session, "CAM_03")

        assert len(camera_rows) == 3
        assert len(parking_slots) == 1
        assert parking_slots[0].id == "B8_BDM"
        assert list(parking_slots[0].polygon.exterior.coords)[0] == (20.0, 20.0)
        assert len(special_zones) == 1
        assert special_zones[0].id == "B1_Entrence"
        assert roi_polygon is not None
    finally:
        session.close()


def test_bootstrap_from_legacy_json_imports_all_definition_types():
    session = make_session()
    try:
        with NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(
                [
                    {"id": "B8_BDM", "polygon": [[1, 1], [2, 1], [2, 2], [1, 2]], "label": "B8_BDM"},
                    {"id": "B1_Entrence", "polygon": [[3, 3], [4, 3], [4, 4], [3, 4]], "label": "B1_Entrence"},
                    {"id": "roi", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "label": "Global ROI Mask"},
                ],
                handle,
            )
            json_path = handle.name

        migrated = bootstrap_camera_slots_from_json(
            session,
            camera_id="CAM_03",
            floor="B1",
            slots_file=json_path,
            default_zone_id="CAM_03",
            default_zone_name="CAM_03",
        )
        parking_slots, special_zones, roi_polygon, boundaries = load_camera_slots(session, "CAM_03")

        assert migrated is True
        assert len(parking_slots) == 1
        assert len(special_zones) == 1
        assert roi_polygon is not None
    finally:
        if "json_path" in locals() and os.path.exists(json_path):
            os.unlink(json_path)
        session.close()
