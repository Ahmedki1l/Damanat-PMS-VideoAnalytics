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
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncIterable

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.sse import EventSourceResponse
from pydantic import BaseModel

from src.vehicle_registry import VehicleRegistry
from src.events.event_bus import EventBus
from src.models.state_machine import SlotEvent
from src.services.alert_service import get_alert_type_for_slot
from src.services.named_slot_service import get_slot_restriction_type


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
    slot_name: Optional[str] = None
    floor: Optional[str] = None
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    occupied: bool
    state: str
    plate_number: Optional[str] = None
    camera_id: Optional[str] = None
    is_restricted: bool = False
    restriction_type: Optional[str] = None
    snapshot_url: Optional[str] = None


class VehicleLocation(BaseModel):
    plate_number: Optional[str] = None
    slot_id: str
    slot_name: Optional[str] = None
    floor: Optional[str] = None
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    parked_at: Optional[str] = None
    camera_id: Optional[str] = None
    snapshot_url: Optional[str] = None
    gate_snapshot_urls: List[str] = []
    gallery_snapshot_urls: List[str] = []
    entry_time: Optional[str] = None
    
class StreamEventRequest(BaseModel):
    """Payload for triggering a raw SSE event without DB persistence."""
    event_type: str = "vehicle_parked"
    slot_id: str
    track_id: Optional[int] = None
    camera_id: str = "EXTERNAL"
    floor: str = "N/A"
    plate_number: str = ""
    is_alert: bool = False
    severity: str = "info"
    slot_name: Optional[str] = None
    zone_id: str = ""
    zone_name: str = ""
    alert_id: Optional[int] = None


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
    get_camera_frame=None,
    get_slot_snapshot_source=None,
    get_park_entry_crop=None,
    get_engine_status=None,
    event_bus: Optional[EventBus] = None,
    db_manager=None,
    snapshot_base_dir: str = "vehicle_images",
    public_base_url: str = "",
    snapshot_url_prefix: str = "/pms-video-analytics/snapshots",
    gateway_path_prefix: str = "",
) -> FastAPI:
    """
    Create the FastAPI application.

    Args:
        vehicle_registry: Shared VehicleRegistry instance.
        get_slot_statuses: Callback function that returns current slot statuses
                          as a list of dicts. Provided by the engine.
        get_slot_snapshot_source: Callback returning slot runtime metadata
                                 (camera ownership + polygon) for snapshots.
        get_engine_status: Callback function for health metrics.
        event_bus: Optional EventBus instance for real-time alerts.
        db_manager: Optional DB manager for querying slot restriction status.
    """
    app = FastAPI(
        title="Damanat PMS Video Analytics API",
        description="Parking management system with ANPR integration and slot monitoring.",
        version="1.0.0",
        root_path=gateway_path_prefix,
    )

    @app.get("/")
    async def root():
        """Root endpoint providing service overview."""
        return {
            "status": "online",
            "service": "Damanat PMS Video Analytics",
            "version": "1.0.0",
            "message": "AI-powered parking management system is running.",
            "endpoints": {
                "docs": "/docs",
                "health": "/api/health",
                "slots": "/api/slots"
            }
        }

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve vehicle/snapshot images from snapshot_base_dir via a regular
    # GET endpoint at a fixed public path. {filepath:path} is required
    # because alert snapshots are written under alerts/<file>.jpg by
    # engine_runtime._save_alert_snapshot and exposed with that nested
    # shape by vehicle_registry_queries._get_snapshot_url.
    os.makedirs(snapshot_base_dir, exist_ok=True)
    _snapshot_base_abs = os.path.abspath(snapshot_base_dir)

    @app.get(
        "/pms-video-analytics/snapshots/{filepath:path}",
        summary="Serve a saved snapshot JPEG",
        response_class=FileResponse,
        name="snapshots",
    )
    def serve_snapshot(filepath: str):
        # Resolve and require the result to live inside snapshot_base_dir.
        # Catches `..` traversal, absolute paths, and Windows drive specs
        # in one check, and still allows legitimate subfolders like alerts/.
        requested_abs = os.path.abspath(os.path.join(_snapshot_base_abs, filepath))
        if not requested_abs.startswith(_snapshot_base_abs + os.sep):
            raise HTTPException(status_code=404, detail="Snapshot not found")
        if not os.path.isfile(requested_abs):
            raise HTTPException(status_code=404, detail="Snapshot not found")
        return FileResponse(requested_abs, media_type="image/jpeg")

    # Use provided or create new registry
    registry = vehicle_registry or VehicleRegistry(
        image_dir=snapshot_base_dir,
        public_base_url=public_base_url,
        snapshot_url_prefix=snapshot_url_prefix,
        gateway_path_prefix=gateway_path_prefix,
    )

    def _build_slot_snapshot_url(slot_id: str) -> str:
        return f"/api/slots/{slot_id}/snapshot/live"

    def _resolve_slot_snapshot_source(slot_id: str) -> Dict[str, Any]:
        if get_slot_snapshot_source is None:
            raise HTTPException(
                status_code=503,
                detail="Slot snapshot source lookup is not enabled on this server.",
            )

        source = get_slot_snapshot_source(slot_id)
        if source is None:
            raise HTTPException(status_code=404, detail=f"Slot '{slot_id}' not found.")

        camera_id = source.get("camera_id")
        slot = source.get("slot")
        if not camera_id or slot is None:
            raise HTTPException(
                status_code=503,
                detail=f"Slot '{slot_id}' is missing live snapshot metadata.",
            )
        return source

    def _crop_slot_snapshot(frame, slot):
        import cv2
        import numpy as np

        if frame is None or slot is None or frame.size == 0:
            return None

        polygon_points = np.array(
            [[int(x), int(y)] for x, y in slot.polygon.exterior.coords[:-1]],
            dtype=np.int32,
        )
        if polygon_points.size == 0:
            return None

        frame_h, frame_w = frame.shape[:2]
        x, y, width, height = cv2.boundingRect(polygon_points)
        x = max(0, x)
        y = max(0, y)
        width = min(width, frame_w - x)
        height = min(height, frame_h - y)
        if width <= 0 or height <= 0:
            return None

        crop = frame[y:y + height, x:x + width].copy()
        shifted_points = polygon_points - np.array([x, y])
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [shifted_points], 255)
        masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
        return masked_crop if masked_crop.size > 0 else None

    def _crop_vehicle_bbox_snapshot(
        frame,
        bbox: Optional[tuple[float, float, float, float]] = None,
        padding_ratio: float = 0.12,
    ):
        if frame is None or bbox is None or frame.size == 0:
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox]
        frame_h, frame_w = frame.shape[:2]
        pad_x = max(12, int((x2 - x1) * padding_ratio))
        pad_y = max(12, int((y2 - y1) * padding_ratio))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame_w, x2 + pad_x)
        y2 = min(frame_h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2].copy()
        return crop if crop.size > 0 else None

    def _jpeg_response_for_crop(crop):
        import cv2

        if crop is None or crop.size == 0:
            raise HTTPException(status_code=503, detail="Unable to generate slot snapshot.")

        success, encoded = cv2.imencode(".jpg", crop)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode slot snapshot.")
        return Response(content=encoded.tobytes(), media_type="image/jpeg")

    def _build_live_slot_snapshot_response(slot_id: str):
        source = _resolve_slot_snapshot_source(slot_id)
        camera_id = source["camera_id"]
        slot = source["slot"]
        state_machine = source.get("state_machine")

        if get_camera_frame is None:
            raise HTTPException(
                status_code=503,
                detail="Live camera frame access is not enabled on this server.",
            )

        success, frame = get_camera_frame(camera_id)
        if not success or frame is None:
            raise HTTPException(
                status_code=503,
                detail=f"Live camera frame unavailable for slot '{slot_id}'.",
            )

        bbox = getattr(state_machine, "latest_detection_bbox", None) if state_machine else None
        crop = _crop_vehicle_bbox_snapshot(frame, bbox=bbox)
        if crop is None or crop.size == 0:
            crop = _crop_slot_snapshot(frame, slot)
        return _jpeg_response_for_crop(crop)

    def _capture_instant_snapshot(
        plate: str,
        direction: str,
        camera_id: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
    ) -> bool:
        if direction != "entry":
            return False

        if image_bytes:
            import cv2
            import numpy as np
            import time

            frame = cv2.imdecode(
                np.frombuffer(image_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is not None and frame.size > 0:
                fake_track_id = -int(time.time() * 1000) % 100000
                candidate = registry.open_park_entry_candidate(
                    camera_id or "ANPR",
                    fake_track_id,
                )
                registry.update_park_entry_candidate_snapshot(
                    candidate.candidate_id,
                    frame,
                    quality_score=999.0,
                )

                # Mark as ANPR-image sourced so downstream matching can prefer
                # this candidate's feature vector over zone-crop candidates.
                # The snapshot file on disk is kept intentionally as a durable
                # gate reference (gate_snapshot_paths on the session).
                with registry._lock:
                    live_candidate = registry._park_entry_candidates.get(candidate.candidate_id)
                    if live_candidate is not None:
                        live_candidate.source = "anpr_image"

                bound_plate = registry.bind_next_pending_anpr_to_candidate(
                    candidate.candidate_id
                )
                if bound_plate:
                    # Retrieve gate snapshot paths and bound event_id from the candidate
                    # so the direct session is wired to the same ANPR event record.
                    _gate_paths: list = []
                    _event_id: str = ""
                    with registry._lock:
                        live_cand = registry._park_entry_candidates.get(candidate.candidate_id)
                        if live_cand is not None:
                            _gate_paths = list(live_cand.snapshot_paths) or (
                                [live_cand.snapshot_path] if live_cand.snapshot_path else []
                            )
                            _event_id = live_cand.bound_event_id or ""

                    registry.confirm_anpr_session_directly(
                        plate=bound_plate,
                        image=frame,
                        event_id=_event_id,
                        candidate_id=candidate.candidate_id,
                        gate_snapshot_paths=_gate_paths,
                    )
                    print(
                        f"[API] ANPR-image candidate created & bound for plate {plate} "
                        f"(will be used as primary identity reference at B1)"
                    )
                    return True
                print(
                    f"[API] ANPR-image candidate created but binding failed "
                    f"for plate {plate}"
                )

        if get_park_entry_crop is not None:
            import cv2
            candidate_camera_ids = ["CAM_03"]

            for snapshot_camera_id in candidate_camera_ids:
                success, crop = get_park_entry_crop(snapshot_camera_id)
                if not success or crop is None:
                    continue

                # 1. Force open an artificial candidate so it can be matched
                # Give it an arbitrary negative ID so YOLO tracks won't conflict
                import time
                fake_track_id = -int(time.time() * 1000) % 100000
                candidate = registry.open_park_entry_candidate(
                    snapshot_camera_id,
                    fake_track_id,
                )
                
                # 2. Inject our cropped entry zone as the car snapshot
                registry.update_park_entry_candidate_snapshot(
                    candidate.candidate_id, crop, quality_score=999.0
                )

                bound_plate = registry.bind_next_pending_anpr_to_candidate(
                    candidate.candidate_id
                )
                if bound_plate:
                    print(
                        f"[API] Instant candidate created & bound from "
                        f"{snapshot_camera_id} for plate {plate}"
                    )
                    return True
                print(
                    f"[API] Candidate created from {snapshot_camera_id} crop but "
                    f"binding failed for plate {plate}"
                )
        return False
    # ── SSE Endpoints ───────────────────────────────────────

    @app.get("/api/alerts/stream", response_class=EventSourceResponse)
    async def alerts_stream():
        """
        Server-Sent Events (SSE) endpoint for real-time alerts.
        
        Broadcasts parking events (parked, vacant, violations) to connected clients
        using flattened JSON format for root-level alert attributes.
        """
        if event_bus is None:
            raise HTTPException(
                status_code=503, 
                detail="Real-time alerts (EventBus) not enabled on this server."
            )

        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Callback to bridge engine thread and async queue
        def callback(event: SlotEvent):
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as e:
                print(f"[API] Error in SSE callback: {e}")

        # Register with EventBus
        event_bus.subscribe(callback)
        print(f"[API] Client connected to alerts stream")

        try:
            # 1. Send initial connection confirmation (handshake)
            yield {
                "is_alert": False,
                "severity": "info",
                "alert_type": "connection_established",
                "msg": "Real-time alerts stream established"
            }

            while True:
                # Wait for next event
                event: SlotEvent = await queue.get()
                
                # Filter: Only send actual alerts to this stream
                if not event.is_alert:
                    continue

                # Yield the dict directly so fields match the required order/naming
                yield event.to_dict()
                
        except asyncio.CancelledError:
            print(f"[API] Client disconnected from alerts stream")
            raise
        finally:
            # Always unsubscribe
            event_bus.unsubscribe(callback)

    # ── Test/Debug Endpoints ───────────────────────────────────

    @app.get("/api/test/trigger")
    async def trigger_test_event(slot_id: str = "TEST_SLOT_01"):
        """
        Simulate a real parking event for a given slot_id.

        Queries the database to check if the slot is restricted (is_violation_zone=True).
        - Restricted slot → is_alert=true, severity=critical, alert_type derived from slot policy
        - Normal slot    → is_alert=false, severity=info,     alert_type=vehicle_parked

        Try with a restricted slot (e.g. B11_CFO, G1) or a normal slot (e.g. A1, G4).
        """
        if event_bus is None:
            raise HTTPException(status_code=503, detail="EventBus not available")

        # Query the DB to determine if this slot is restricted
        is_restricted = False
        slot_found = False
        if db_manager is not None:
            from src.repositories import ParkingSlotRepository
            session = db_manager.SessionLocal()
            try:
                db_slot = ParkingSlotRepository.get_by_id(session, slot_id)
                if db_slot:
                    slot_found = True
                    is_restricted = bool(db_slot.is_violation_zone)
            except Exception as e:
                print(f"[API] DB query failed in trigger: {e}")
            finally:
                session.close()

        # Build event based on real DB flag and slot policy
        if is_restricted:
            session = db_manager.SessionLocal() if db_manager is not None else None
            try:
                alert_type = get_alert_type_for_slot(session, slot_id) if session else "vehicle_violation"
            finally:
                if session is not None:
                    session.close()
        else:
            alert_type = "vehicle_parked"

        test_evt = SlotEvent(
            event_type=alert_type,
            slot_id=slot_id,
            track_id=999,
            timestamp=datetime.now().isoformat(),
            camera_id="SIM_CAM",
            floor="Simulation",
            is_alert=is_restricted,
            severity="critical" if is_restricted else "info"
        )

        event_bus.emit(test_evt)
        return {
            "status": "success",
            "slot_id": slot_id,
            "slot_found_in_db": slot_found,
            "is_violation_zone_in_db": is_restricted,
            "event_sent": test_evt.to_dict()
        }

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
            camera_id=event.camera_id,
        )

        print(f"[API] ✓ Plate {record.plate} registered")
        
        image_saved = _capture_instant_snapshot(
            record.plate,
            record.direction,
            camera_id=event.camera_id,
            image_bytes=image_bytes,
        )

        return ANPREventResponse(
            status="ok",
            plate=record.plate,
            direction=record.direction,
            image_saved=image_saved,
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
        )

        print(f"[API] ✓ Plate {record.plate} registered")

        image_saved = _capture_instant_snapshot(
            record.plate,
            record.direction,
            image_bytes=image_bytes,
        )

        return ANPREventResponse(
            status="ok",
            plate=record.plate,
            direction=record.direction,
            image_saved=image_saved,
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
            
            # Map existing state structure to standardized SlotStatus
            result.append(SlotStatus(
                slot_id=slot_id,
                slot_name=s.get("slot_name") or s.get("label", slot_id),
                floor=s.get("floor"),
                zone_id=s.get("zone_id"),
                zone_name=s.get("zone_name"),
                occupied=s.get("occupied", False),
                state=s.get("state", "UNKNOWN"),
                plate_number=plate,
                camera_id=s.get("camera_id"),
                is_restricted=s.get("is_violation_zone", False),
                restriction_type=get_slot_restriction_type(slot_id) if s.get("is_violation_zone") else None,
                snapshot_url=_build_slot_snapshot_url(slot_id),
            ))
        return result

    @app.get("/api/slots/{slot_id}/snapshot/live")
    async def get_live_slot_snapshot(slot_id: str):
        """Return a live JPEG crop for the requested slot."""
        return _build_live_slot_snapshot_response(slot_id)

    @app.get("/api/slots/{slot_id}/snapshot/latest")
    async def get_latest_slot_snapshot(slot_id: str):
        """Return the latest saved slot snapshot, falling back to a live crop."""
        if db_manager is not None:
            from src.repositories import ParkingSlotRepository

            session = db_manager.SessionLocal()
            try:
                db_slot = ParkingSlotRepository.get_by_id(session, slot_id)
                if db_slot is not None and db_slot.last_snapshot_path:
                    image_path = os.path.join(snapshot_base_dir, db_slot.last_snapshot_path)
                    if os.path.exists(image_path):
                        return FileResponse(image_path, media_type="image/jpeg")
            finally:
                session.close()

        return _build_live_slot_snapshot_response(slot_id)

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
                    slot_name=s.get("slot_name") or s.get("label", slot_id),
                    floor=s.get("floor"),
                    zone_id=s.get("zone_id"),
                    zone_name=s.get("zone_name"),
                    occupied=s.get("occupied", False),
                    state=s.get("state", "UNKNOWN"),
                    plate_number=plate,
                    camera_id=s.get("camera_id"),
                    is_restricted=s.get("is_violation_zone", False),
                    restriction_type=get_slot_restriction_type(slot_id) if s.get("is_violation_zone") else None,
                    snapshot_url=_build_slot_snapshot_url(slot_id),
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
        """Enriched health reporting including engine status."""
        health_data = {
            "status": "ok", 
            "service": "Damanat PMS Video Analytics",
            "timestamp": datetime.now().isoformat()
        }
        if get_engine_status:
            health_data.update(get_engine_status())
        return health_data

    return app


# --- Standalone Server ---

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
