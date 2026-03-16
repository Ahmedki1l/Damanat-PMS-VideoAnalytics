"""
api.py — FastAPI server for ANPR integration and slot status queries.

Endpoints:
  POST /api/anpr/event               — Receive plate + image from ANPR server
  GET  /api/slots                     — All slot statuses across all cameras
  GET  /api/slots/{floor}             — Slot statuses for a specific floor
  GET  /api/vehicle/{plate}           — Find where a plate is parked
  GET  /api/vehicles                  — All currently parked vehicles
  GET  /api/vehicles/pending          — Vehicles that entered but not yet parked
  GET  /api/stats                     — Summary statistics

Usage:
    # Standalone API server:
    python -m src.api

    # Or start with main.py:
    python main.py --api
"""

import os
import base64
from datetime import datetime
import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.vehicle_registry import VehicleRegistry


# --- Pydantic Models ---

class ANPREventRequest(BaseModel):
    """ANPR event payload (JSON body with optional base64 image)."""
    plate: str
    direction: str = "entry"  # "entry" or "exit"
    image_base64: Optional[str] = None  # Base64-encoded JPEG image
    camera_id: Optional[str] = None     # Which ANPR camera sent this
    timestamp: Optional[str] = None     # ISO timestamp (defaults to now)


class ANPREventResponse(BaseModel):
    """Response after processing an ANPR event."""
    status: str
    plate: str
    direction: str
    image_saved: bool
    timestamp: str


class SlotStatus(BaseModel):
    slot_id: str
    state: str
    occupied: bool
    assigned_track_id: Optional[int] = None
    camera_id: Optional[str] = None
    floor: Optional[str] = None
    plate: Optional[str] = None


class VehicleLocation(BaseModel):
    plate: str
    slot_id: str
    camera_id: Optional[str] = None
    floor: Optional[str] = None
    parked_at: Optional[str] = None
    entry_time: str


class StatsResponse(BaseModel):
    total_slots: int
    occupied_slots: int
    vacant_slots: int
    parked_vehicles: int
    pending_entries: int
    total_visits: int


# --- App Factory ---

def create_app(
    vehicle_registry: Optional[VehicleRegistry] = None,
    get_slot_statuses=None,
) -> FastAPI:
    """
    Create the FastAPI application.

    Args:
        vehicle_registry: Shared VehicleRegistry instance.
        get_slot_statuses: Callback function that returns current slot statuses
                          as a list of dicts. Provided by the engine.
    """
    app = FastAPI(
        title="Damanat PMS Video Analytics API",
        description="Parking management system with ANPR integration and slot monitoring.",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Use provided or create new registry
    registry = vehicle_registry or VehicleRegistry()

    # ── ANPR Endpoints ──────────────────────────────────────

    @app.post("/api/anpr/event", response_model=ANPREventResponse)
    async def anpr_event_json(event: ANPREventRequest):
        """
        Receive an ANPR event via JSON body.

        Accepts plate number and optionally a base64-encoded vehicle image.
        """
        img_status = "no"
        if event.image_base64 is None:
            img_status = "no (field is None)"
        elif event.image_base64 == "":
            img_status = "no (empty string)"
        else:
            img_status = f"yes ({len(event.image_base64)} base64 chars ≈ {len(event.image_base64)*3//4//1024}KB)"

        print(f"\n{'='*60}")
        print(f"[API] ANPR EVENT RECEIVED (JSON)")
        print(f"[API]   Plate     : {event.plate}")
        print(f"[API]   Direction : {event.direction}")
        print(f"[API]   Camera    : {event.camera_id or 'N/A'}")
        print(f"[API]   Image     : {img_status}")
        print(f"[API]   Time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        image_bytes = None
        if event.image_base64:
            try:
                image_bytes = base64.b64decode(event.image_base64)
            except Exception:
                print(f"[API] ERROR: Invalid base64 image for plate {event.plate}")
                raise HTTPException(status_code=400, detail="Invalid base64 image data")

        record = registry.register_anpr_event(
            plate=event.plate,
            direction=event.direction,
            image_bytes=image_bytes,
        )

        print(f"[API] ✓ Plate {record.plate} registered | Image saved: {record.image_path or 'none'}")

        return ANPREventResponse(
            status="ok",
            plate=record.plate,
            direction=record.direction,
            image_saved=record.image_path is not None,
            timestamp=record.timestamp.isoformat(),
        )

    @app.post("/api/anpr/event/upload", response_model=ANPREventResponse)
    async def anpr_event_upload(
        plate: str = Form(...),
        direction: str = Form("entry"),
        image: Optional[UploadFile] = File(None),
    ):
        """
        Receive an ANPR event via multipart form upload.

        Alternative to JSON endpoint — accepts image as file upload.
        """
        print(f"\n{'='*60}")
        print(f"[API] ANPR EVENT RECEIVED (FILE UPLOAD)")
        print(f"[API]   Plate     : {plate}")
        print(f"[API]   Direction : {direction}")
        print(f"[API]   Image     : {image.filename if image else 'none'}")
        print(f"[API]   Time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        image_bytes = None
        if image:
            image_bytes = await image.read()

        record = registry.register_anpr_event(
            plate=plate,
            direction=direction,
            image_bytes=image_bytes,
        )

        print(f"[API] ✓ Plate {record.plate} registered | Image saved: {record.image_path or 'none'}")

        return ANPREventResponse(
            status="ok",
            plate=record.plate,
            direction=record.direction,
            image_saved=record.image_path is not None,
            timestamp=record.timestamp.isoformat(),
        )

    # ── Slot Status Endpoints ───────────────────────────────

    @app.get("/api/slots", response_model=List[SlotStatus])
    async def get_all_slots():
        """Get status of all parking slots across all cameras."""
        if get_slot_statuses is None:
            return []

        statuses = get_slot_statuses()
        result = []
        for s in statuses:
            slot_id = s.get("slot_id", "")
            plate = registry.get_slot_plate(slot_id)
            result.append(SlotStatus(
                slot_id=slot_id,
                state=s.get("state", "UNKNOWN"),
                occupied=s.get("occupied", False),
                assigned_track_id=s.get("assigned_track_id"),
                camera_id=s.get("camera_id"),
                floor=s.get("floor"),
                plate=plate,
            ))
        return result

    @app.get("/api/slots/{floor}", response_model=List[SlotStatus])
    async def get_floor_slots(floor: str):
        """Get slot statuses for a specific floor (B1, B2)."""
        if get_slot_statuses is None:
            return []

        statuses = get_slot_statuses()
        result = []
        for s in statuses:
            if s.get("floor", "").upper() == floor.upper():
                slot_id = s.get("slot_id", "")
                plate = registry.get_slot_plate(slot_id)
                result.append(SlotStatus(
                    slot_id=slot_id,
                    state=s.get("state", "UNKNOWN"),
                    occupied=s.get("occupied", False),
                    assigned_track_id=s.get("assigned_track_id"),
                    camera_id=s.get("camera_id"),
                    floor=s.get("floor"),
                    plate=plate,
                ))
        return result

    # ── Vehicle Endpoints ───────────────────────────────────

    @app.get("/api/vehicle/{plate}")
    async def find_vehicle(plate: str):
        """Find where a specific vehicle is parked by plate number."""
        location = registry.get_plate_location(plate)
        if location is None:
            raise HTTPException(
                status_code=404,
                detail=f"Vehicle with plate '{plate}' not found in any slot.",
            )
        return location

    @app.get("/api/vehicles", response_model=List[VehicleLocation])
    async def get_all_vehicles():
        """Get all currently parked vehicles with their slot locations."""
        return registry.get_all_parked()

    @app.get("/api/vehicles/pending")
    async def get_pending_entries():
        """Get vehicles that entered via ANPR but haven't been linked to a slot."""
        return registry.get_pending_entries()

    # ── Stats Endpoint ──────────────────────────────────────

    @app.get("/api/stats", response_model=StatsResponse)
    async def get_stats():
        """Get parking system summary statistics."""
        reg_stats = registry.get_stats()

        total = 0
        occupied = 0
        if get_slot_statuses:
            statuses = get_slot_statuses()
            total = len(statuses)
            occupied = sum(1 for s in statuses if s.get("occupied", False))

        return StatsResponse(
            total_slots=total,
            occupied_slots=occupied,
            vacant_slots=total - occupied,
            parked_vehicles=reg_stats["parked_count"],
            pending_entries=reg_stats["pending_entries"],
            total_visits=reg_stats["total_visits"],
        )

    # ── Health Check ────────────────────────────────────────

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "Damanat PMS Video Analytics"}

    return app


# --- Standalone Server ---

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
