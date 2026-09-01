"""
api.py — FastAPI server for ANPR integration and slot status queries.

Endpoints:
  POST /api/anpr/event               — Receive plate + image from ANPR server
  POST /api/line-crossing            — Receive B1-entry crossing image (CAM-03); crop car; seed ReID
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
import hmac
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, Header, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

from src.vehicle_registry import VehicleRegistry
from src.events.event_bus import EventBus
from src.entry.analyzer import PLATE_CROP_SUBDIRECTORY
from src.entry.domain import EntryCapacityExceeded, EntryInvalid, EntryUnavailable
from src.models.state_machine import SlotEvent
from src.services.alert_service import get_alert_type_for_slot, notification_suppressed
from src.services.named_slot_service import get_slot_restriction_type, is_restricted_slot  # now takes a slot ORM object


# --- Pydantic Models ---

class ANPREventRequest(BaseModel):
    """ANPR event payload (JSON body with optional base64 image)."""
    plate: str
    direction: str = "entry"  # "entry" or "exit"
    image_base64: Optional[str] = None  # Base64-encoded JPEG image
    camera_id: Optional[str] = None     # Which ANPR camera sent this
    captured_at: Optional[str] = None   # Source event time; timezone required
    # Backward-compatible alias accepted from older clients.  When both fields
    # are supplied they must represent the same instant.
    timestamp: Optional[str] = None
    # Optional per-read OCR confidence in [0,1] from the ANPR server. When
    # supplied, an entry read below matching.anpr_min_accept_confidence is HELD
    # (not registered as a pending plate) — this stops a low-confidence night
    # misread from minting a second identity for one car. Omit (None) to accept
    # every read, preserving current behaviour for servers that don't send it.
    confidence: Optional[float] = None


class ReIDRenameRequest(BaseModel):
    """Re-file a car under a corrected plate.

    Sent by PMS-AI after `plate_correction_service.apply_correction` rewrites a
    stay whose ENTRY plate was misread, so VA stops re-minting the misread.
    """

    from_plate: str = Field(alias="from")
    to_plate: str = Field(alias="to")

    model_config = {"populate_by_name": True}


class ReIDCompareRequest(BaseModel):
    """Score one crop against the persisted galleries of named plates.

    PMS-AI sends the crop of a car leaving and the plates of the open sessions it
    might belong to; VA answers with a similarity per plate and decides nothing.
    Used to resolve an exit whose plate matched no session because the ENTRY read
    was wrong — see PMS-AI `exit_match_service`.
    """

    image_base64: str
    plates: List[str]


def _parse_anpr_source_timestamp(event: ANPREventRequest) -> Optional[datetime]:
    """Parse the ANPR source time without ever substituting delivery time.

    Exit is destructive, so it requires an offset-aware source timestamp.
    Legacy non-exit callers may omit one and continue to use the registry clock.
    Any supplied value is strict: malformed or timezone-naive input is rejected,
    and the old/new fields cannot disagree.
    """

    def _parse(raw: str, field_name: str) -> datetime:
        value = raw.strip()
        if not value:
            raise HTTPException(
                status_code=422,
                detail=f"{field_name}_invalid_iso_timestamp",
            )
        if value.endswith(("Z", "z")):
            value = value[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"{field_name}_invalid_iso_timestamp",
            ) from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise HTTPException(
                status_code=422,
                detail=f"{field_name}_requires_timezone",
            )
        return parsed

    captured = _parse(event.captured_at, "captured_at") if event.captured_at is not None else None
    legacy = _parse(event.timestamp, "timestamp") if event.timestamp is not None else None
    if captured is not None and legacy is not None and captured != legacy:
        raise HTTPException(status_code=422, detail="source_timestamp_conflict")

    source_time = captured or legacy
    if event.direction.strip().lower() == "exit" and source_time is None:
        raise HTTPException(status_code=422, detail="exit_requires_captured_at")
    return source_time


class LineCrossingRequest(BaseModel):
    """B1-entrance line-crossing image from the external detector (CAM-03),
    sent after the ANPR event. The entering car is detected, cropped, and used
    to seed the ReID candidate for the pending plate."""
    image_base64: str                    # Base64-encoded JPEG of the full CAM-03 frame
    camera_id: Optional[str] = "CAM-03"  # Camera that produced the crossing
    plate: Optional[str] = None          # Optional ANPR plate for correlation/logging
    timestamp: Optional[str] = None      # ISO timestamp (defaults to now)


class LineCrossingResponse(BaseModel):
    """Response after processing a line-crossing image."""
    status: str
    plate: Optional[str] = None
    cropped: bool          # whether a vehicle was detected & cropped
    bound: bool            # whether it bound to a pending ANPR entry
    timestamp: str


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

def _first(*values):
    """First value that isn't None or empty — used to prefer a slot's live
    in-process metadata over the DB row, and the DB row over nothing."""
    for value in values:
        if value:
            return value
    return None


def create_app(
    vehicle_registry: Optional[VehicleRegistry] = None,
    get_slot_statuses=None,
    get_camera_frame=None,
    get_slot_snapshot_source=None,
    get_park_entry_crop=None,
    detect_vehicle_crop=None,
    get_engine_status=None,
    event_bus: Optional[EventBus] = None,
    db_manager=None,
    snapshot_base_dir: str = "vehicle_images",
    public_base_url: str = "",
    snapshot_url_prefix: str = "/pms-video-analytics/snapshots",
    gateway_path_prefix: str = "",
    entry_coordinator=None,
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
    os.makedirs(snapshot_base_dir, exist_ok=True)
    _snapshot_base_abs = os.path.abspath(snapshot_base_dir)
    registry = vehicle_registry or VehicleRegistry(
        image_dir=snapshot_base_dir,
        public_base_url=public_base_url,
        snapshot_url_prefix=snapshot_url_prefix,
        gateway_path_prefix=gateway_path_prefix,
    )

    # V2 entry validation is isolated from the legacy ANPR/session path and is
    # feature-gated off by default. Tests may inject a model-free coordinator;
    # production resolves the existing ReID/LPD/Paddle components lazily.
    from src.entry.router import create_entry_router, install_entry_transport_guard
    from src.entry.settings import SERVICE_KEY_HEADER
    from src.entry.runtime import (
        build_entry_coordinator,
        entry_callback_retry_lifespan,
    )

    active_entry_coordinator = (
        entry_coordinator
        if entry_coordinator is not None
        else build_entry_coordinator(
            registry,
            image_dir=snapshot_base_dir,
        )
    )
    app = FastAPI(
        title="Damanat PMS Video Analytics API",
        description="Parking management system with ANPR integration and slot monitoring.",
        version="1.0.0",
        root_path=gateway_path_prefix,
        lifespan=entry_callback_retry_lifespan(active_entry_coordinator),
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
        relative_path = os.path.relpath(requested_abs, _snapshot_base_abs)
        if relative_path == PLATE_CROP_SUBDIRECTORY or relative_path.startswith(
            PLATE_CROP_SUBDIRECTORY + os.sep
        ):
            # OCR diagnostics may contain license plates and are intentionally
            # filesystem-only; the general snapshot route must not expose them.
            raise HTTPException(status_code=404, detail="Snapshot not found")
        if not os.path.isfile(requested_abs):
            raise HTTPException(status_code=404, detail="Snapshot not found")
        return FileResponse(requested_abs, media_type="image/jpeg")

    app.state.entry_coordinator = active_entry_coordinator
    install_entry_transport_guard(app, active_entry_coordinator)
    app.include_router(create_entry_router(active_entry_coordinator))

    def _require_internal_anpr_auth(provided_service_key: Optional[str]) -> None:
        """Only PMS-AI may mutate VA's ANPR registry in V2 modes."""
        if active_entry_coordinator.settings.mode.value == "off":
            return
        expected = active_entry_coordinator.settings.service_key
        supplied = provided_service_key or ""
        if not expected or not hmac.compare_digest(expected, supplied):
            raise HTTPException(status_code=401, detail="invalid_service_key")

    def _require_legacy_anpr_exit(direction: str) -> None:
        """Keep only the PMS exit bridge once V2 owns entry admission."""
        if active_entry_coordinator.settings.mode.value != "authoritative":
            return
        if direction.strip().lower() != "exit":
            raise HTTPException(
                status_code=410,
                detail="legacy_anpr_entry_disabled_in_entry_v2",
            )

    async def _close_entry_v2_journey(
        record,
        source_timestamp: Optional[datetime],
    ) -> None:
        if record.direction.strip().lower() != "exit":
            return
        if active_entry_coordinator.settings.mode.value == "off":
            return
        if source_timestamp is None:
            raise HTTPException(status_code=422, detail="exit_requires_captured_at")
        try:
            closed = await run_in_threadpool(
                active_entry_coordinator.record_exit,
                record.plate,
                source_timestamp,
            )
        except EntryInvalid as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (EntryCapacityExceeded, EntryUnavailable) as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
                headers={"Retry-After": "1"},
            ) from exc
        if closed:
            print(
                f"[EntryV2] exit closed {closed} journey(s) "
                f"plate={record.plate} captured_at={source_timestamp.isoformat()}"
            )

    def _anpr_response_timestamp(
        record,
        source_timestamp: Optional[datetime],
    ) -> str:
        """Echo the validated exit source time; keep legacy entry behavior."""
        if (
            record.direction.strip().lower() == "exit"
            and source_timestamp is not None
        ):
            return source_timestamp.isoformat()
        return record.timestamp.isoformat()

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
        frame=None,
        event_id: Optional[str] = None,
    ) -> bool:
        # B-entry = the CAM-03 B1 confirmation snapshot pushed via the ANPR API
        # (plate + image). Attach it to the plate's session as the primary ReID
        # identity reference at B1, rather than opening a new gate candidate.
        if direction == "B-entry":
            import cv2
            import numpy as np

            if frame is None and image_bytes:
                frame = cv2.imdecode(
                    np.frombuffer(image_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
            if frame is None or frame.size == 0:
                print(f"[API] B-entry for plate {plate}: no usable image — skipped")
                return False
            # Crop to the entering car: the CAM-03 frame is a wide fisheye view
            # with the car in a corner (+ parked cars in the background), so
            # embedding the full frame is the dominant cause of low ReID scores.
            # Store a tight crop instead; DO NOT fall back to full frame — if no
            # car is detected, skip the snapshot rather than poison the gallery
            # with a full-frame anchor.
            cropped = False
            if detect_vehicle_crop is not None:
                try:
                    _c = detect_vehicle_crop(frame)
                    if _c is not None and getattr(_c, "size", 0) > 0:
                        frame = _c
                        cropped = True
                except Exception as _exc:
                    print(f"[API] B-entry vehicle crop failed: {_exc!r}")

            if not cropped:
                print(
                    f"[API] B-entry for plate {plate}: no vehicle crop detected "
                    f"— skipping (avoid full-frame anchor)"
                )
                return False

            session_id = registry.confirm_b1_entrance_by_plate(plate, frame)
            if session_id:
                print(
                    f"[API] CAM-03 B-entry snapshot attached for plate {plate} "
                    f"-> session {session_id} (primary ReID reference at B1)"
                )
                return True
            print(
                f"[API] B-entry for plate {plate}: no matching session yet "
                f"(gate entry not seen?) — snapshot not attached"
            )
            return False

        # ramp-entry = the CAM-23 ramp-top snapshot pushed via the ANPR API. Add
        # it to the plate's session gallery as a SECONDARY appearance reference
        # (an extra viewpoint for cross-camera ReID) — it does NOT override the
        # primary B-entry (CAM-03) reference.
        if direction == "ramp-entry":
            import cv2
            import numpy as np

            if frame is None and image_bytes:
                frame = cv2.imdecode(
                    np.frombuffer(image_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
            if frame is None or frame.size == 0:
                print(f"[API] ramp-entry for plate {plate}: no usable image — skipped")
                return False
            # Crop to the car before storing as a gallery reference (same reason
            # as B-entry — a wide frame embeds poorly). DO NOT fall back to full
            # frame — if no car is detected, skip the snapshot rather than poison
            # the gallery with a full-frame anchor.
            cropped = False
            if detect_vehicle_crop is not None:
                try:
                    _c = detect_vehicle_crop(frame)
                    if _c is not None and getattr(_c, "size", 0) > 0:
                        frame = _c
                        cropped = True
                except Exception as _exc:
                    print(f"[API] ramp-entry vehicle crop failed: {_exc!r}")

            if not cropped:
                print(
                    f"[API] ramp-entry for plate {plate}: no vehicle crop detected "
                    f"— skipping gallery snapshot (avoid full-frame anchor)"
                )
                return False

            session_id = registry.add_gallery_snapshot_by_plate(
                plate, frame, source_cam=camera_id or "CAM-23",
            )
            if session_id:
                print(
                    f"[API] CAM-23 ramp-entry snapshot added for plate {plate} "
                    f"-> session {session_id} (secondary ReID reference)"
                )
                return True
            print(
                f"[API] ramp-entry for plate {plate}: no matching session yet "
                f"(gate entry not seen?) — snapshot not added"
            )
            return False

        if direction != "entry":
            return False

        import cv2
        import numpy as np
        import time

        # Accept either a pre-decoded BGR frame (e.g. an already-cropped car
        # from the line-crossing endpoint) or raw JPEG bytes to decode here.
        if frame is None and image_bytes:
            frame = cv2.imdecode(
                np.frombuffer(image_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )

        if frame is not None:
            if frame.size > 0:
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

                # Bind this candidate to the SPECIFIC ANPR event we just
                # registered — the candidate's image is that plate's own car, so
                # pairing by identity avoids the FIFO cross-bind that swaps two
                # cars entering close together. When the specific bind fails
                # (event already confirmed / expired / coalesced), do NOT fall
                # back to FIFO: this image belongs to THIS plate's car, and
                # FIFO would attach it to the oldest pending entry — a
                # different car — cementing exactly the swap the event-specific
                # bind exists to prevent. FIFO remains only for callers without
                # an event_id (the plate-less line-crossing / zone-crop paths).
                if event_id:
                    bound_plate = registry.bind_anpr_event_to_candidate(
                        candidate.candidate_id, event_id
                    )
                    if not bound_plate:
                        # Retire the just-opened candidate so it cannot linger
                        # and match a future B1 confirmation for another car.
                        registry.drop_provisional_binding(candidate.candidate_id)
                        print(
                            f"[API] ANPR-image candidate for plate {plate}: event "
                            f"{event_id} no longer bindable (already confirmed or "
                            f"expired) — skipped, NOT falling back to FIFO"
                        )
                        return False
                else:
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

                    # Crop the ANPR frame to the car so the held-back reference is
                    # a tight vehicle crop — the frame is a wide gate shot (road,
                    # buildings, sun) that embeds poorly. Require successful crop;
                    # DO NOT fall back to full frame. This crop is stashed as the
                    # session's pending_anpr_vector and only promoted to a matchable
                    # reference after CAM-03 confirmation.
                    _anpr_ref = None
                    if detect_vehicle_crop is not None:
                        try:
                            _car = detect_vehicle_crop(frame)
                            if _car is not None and getattr(_car, "size", 0) > 0:
                                _anpr_ref = _car
                        except Exception as _exc:
                            print(f"[API] ANPR vehicle crop failed: {_exc!r}")

                    if _anpr_ref is None:
                        print(
                            f"[API] ANPR entry for plate {plate}: no vehicle crop detected "
                            f"— skipping (avoid full-frame anchor)"
                        )
                        return False

                    registry.confirm_anpr_session_directly(
                        plate=bound_plate,
                        image=_anpr_ref,
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
            candidate_camera_ids = ["CAM-03"]

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
        print("[API] Client connected to alerts stream")

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

                # Filter: alert types configured as notification-suppressed.
                # They are already persisted by report_alert and remain visible
                # via /api/alerts/* — they just don't interrupt an operator.
                # SlotEvent.to_dict() publishes event_type as "alert_type".
                if notification_suppressed(event.event_type):
                    continue

                # Yield the dict directly so fields match the required order/naming
                yield event.to_dict()
                
        except asyncio.CancelledError:
            print("[API] Client disconnected from alerts stream")
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
                    is_restricted = is_restricted_slot(db_slot)
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

        db_alert_id = None
        if is_restricted and db_manager is not None:
            from src.services.alert_service import report_alert
            session = db_manager.SessionLocal()
            try:
                alert = report_alert(session, slot_id, severity="critical", camera_id="SIM_CAM")
                if alert:
                    db_alert_id = alert.id
            except Exception as e:
                print(f"[API] Failed to persist test alert: {e}")
            finally:
                session.close()

        test_evt = SlotEvent(
            event_type=alert_type,
            slot_id=slot_id,
            track_id=999,
            timestamp=datetime.now().isoformat(),
            camera_id="SIM_CAM",
            floor="Simulation",
            is_alert=is_restricted,
            severity="critical" if is_restricted else "info",
            alert_id=db_alert_id,
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
    async def anpr_event_json(
        event: ANPREventRequest,
        service_key: Optional[str] = Header(None, alias=SERVICE_KEY_HEADER),
    ):
        """
        Receive an ANPR event via JSON body.

        Accepts plate number and optionally a base64-encoded vehicle image.
        """
        _require_internal_anpr_auth(service_key)
        _require_legacy_anpr_exit(event.direction)
        source_timestamp = _parse_anpr_source_timestamp(event)

        img_status = "no"
        if event.image_base64 is None:
            img_status = "no (field is None)"
        elif event.image_base64 == "":
            img_status = "no (empty string)"
        else:
            img_status = f"yes ({len(event.image_base64)} base64 chars ~ {len(event.image_base64)*3//4//1024}KB)"

        print(f"\n{'='*60}")
        print("[API] ANPR EVENT RECEIVED (JSON)")
        print(f"[API]   Plate     : {event.plate}")
        print(f"[API]   Direction : {event.direction}")
        print(f"[API]   Camera    : {event.camera_id or 'N/A'}")
        print(f"[API]   Image     : {img_status}")
        print(f"[API]   Time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        # Source-side confidence gate: hold a low-confidence entry read before it
        # can mint a pending plate (a low-confidence night misread is the seed of
        # the two-plate-for-one-car incident). No-op when the ANPR server omits a
        # confidence or the threshold is 0.
        min_conf = float(
            getattr(registry.matching_config, "anpr_min_accept_confidence", 0.0)
        )
        if (
            event.direction == "entry"
            and event.confidence is not None
            and min_conf > 0.0
            and float(event.confidence) < min_conf
        ):
            print(
                f"[API] ANPR entry for plate {event.plate} HELD: confidence "
                f"{float(event.confidence):.2f} < {min_conf:.2f} — low-confidence "
                f"read not registered (two-plate-genesis guard)"
            )
            return ANPREventResponse(
                status="held_low_confidence",
                plate=event.plate,
                direction=event.direction,
                image_saved=False,
                timestamp=datetime.now().isoformat(),
            )

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
            timestamp=source_timestamp,
            camera_id=event.camera_id,
        )
        await _close_entry_v2_journey(record, source_timestamp)

        print(f"[API] [OK] Plate {record.plate} registered")
        
        image_saved = _capture_instant_snapshot(
            record.plate,
            record.direction,
            camera_id=event.camera_id,
            image_bytes=image_bytes,
            event_id=record.event_id,
        )

        return ANPREventResponse(
            status="ok",
            plate=record.plate,
            direction=record.direction,
            image_saved=image_saved,
            timestamp=_anpr_response_timestamp(record, source_timestamp),
        )

    @app.post("/api/anpr/event/upload", response_model=ANPREventResponse)
    async def anpr_event_upload(
        plate: str = Form(...),
        direction: str = Form("entry"),
        captured_at: Optional[str] = Form(None),
        timestamp: Optional[str] = Form(None),
        image: Optional[UploadFile] = File(None),
        service_key: Optional[str] = Header(None, alias=SERVICE_KEY_HEADER),
    ):
        """
        Receive an ANPR event via multipart form upload.

        Alternative to JSON endpoint — accepts image as file upload.
        """
        _require_internal_anpr_auth(service_key)
        print(f"\n{'='*60}")
        print("[API] ANPR EVENT RECEIVED (FILE UPLOAD)")
        print(f"[API]   Plate     : {plate}")
        print(f"[API]   Direction : {direction}")
        print(f"[API]   Image     : {image.filename if image else 'none'}")
        print(f"[API]   Time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        source_timestamp = _parse_anpr_source_timestamp(
            ANPREventRequest(
                plate=plate,
                direction=direction,
                captured_at=captured_at,
                timestamp=timestamp,
            )
        )

        image_bytes = None
        if image:
            image_bytes = await image.read()

        record = registry.register_anpr_event(
            plate=plate,
            direction=direction,
            timestamp=source_timestamp,
        )
        await _close_entry_v2_journey(record, source_timestamp)

        print(f"[API] [OK] Plate {record.plate} registered")

        image_saved = _capture_instant_snapshot(
            record.plate,
            record.direction,
            image_bytes=image_bytes,
            event_id=record.event_id,
        )

        return ANPREventResponse(
            status="ok",
            plate=record.plate,
            direction=record.direction,
            image_saved=image_saved,
            timestamp=_anpr_response_timestamp(record, source_timestamp),
        )

    @app.post("/api/reid/compare")
    async def reid_compare(
        payload: ReIDCompareRequest,
        service_key: Optional[str] = Header(None, alias=SERVICE_KEY_HEADER),
    ):
        """Similarity between one query crop and each named plate's gallery.

        Answers "which of these cars is this?" and nothing else — no session is
        read or written, no gallery is modified. The caller applies the margin
        and owns the decision, because the cost of a wrong answer is theirs.

        `current_tag_only=True` is mandatory: a gallery folder can hold refs
        embedded under different model contracts (BHD-9990 in production carries
        4 refs under one tag and 16 under another) and a cosine distance across
        two contracts is meaningless. Scoring vectors directly, as this does,
        must therefore drop the stale ones.

        A plate with no usable refs returns `score: null` — absence of evidence,
        which the caller must not read as dissimilarity.
        """
        _require_internal_anpr_auth(service_key)

        import cv2
        import numpy as np

        from src.reid_matcher.reid_burst import is_overexposed, sharpness_score
        from src.reid_matcher.reid_matcher import (
            VehicleReIDMatcher,
            get_reid_matcher,
        )

        store = getattr(registry, "gallery_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="gallery_disabled")

        try:
            buffer = np.frombuffer(base64.b64decode(payload.image_base64), np.uint8)
            query_crop = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except Exception:
            query_crop = None
        if query_crop is None or query_crop.size == 0:
            raise HTTPException(status_code=400, detail="undecodable_image")

        # Report quality rather than silently scoring a crop nothing could match
        # from. The caller refuses on a bad query instead of trusting a low score.
        quality_ok = not is_overexposed(query_crop)
        sharpness = float(sharpness_score(query_crop))

        matcher = get_reid_matcher()
        query_vec = matcher.extract_feature(query_crop)
        if query_vec is None:
            raise HTTPException(status_code=422, detail="no_query_feature")

        results = []
        for plate in payload.plates[:20]:
            vectors, model_tag, _cameras = store.load_vectors(
                plate, current_tag_only=True
            )
            if not vectors:
                results.append({"plate": plate, "score": None, "refs": 0})
                continue
            best = max(
                VehicleReIDMatcher.compute_similarity(query_vec, vec)
                for vec in vectors
            )
            results.append(
                {
                    "plate": plate,
                    "score": round(float(best), 4),
                    "refs": len(vectors),
                    "model_tag": model_tag,
                }
            )

        results.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0.0)))
        return {
            "query_quality_ok": quality_ok,
            "query_sharpness": round(sharpness, 2),
            "results": results,
        }

    @app.post("/api/reid/rename")
    async def reid_rename(
        payload: ReIDRenameRequest,
        service_key: Optional[str] = Header(None, alias=SERVICE_KEY_HEADER),
    ):
        """Re-file a car under the plate it should have had all along.

        PMS-AI corrects a stay whose ENTRY plate was misread — the exit read is
        the only evidence that can catch that. Without this call the correction
        stops at PMS-AI's tables: VA's gallery folder, its live parked session
        and `parking_slots.current_plate` all still say the misread, and the next
        slot update writes it straight back over the fix.

        Three places, each independent, none allowed to fail the other two.
        Reported per place rather than as one boolean so a partial rename is
        visible instead of silently looking like success — `gallery_renamed:
        false` with `slots_updated: 1` is a real state and the caller should be
        able to see it.

        Idempotent: replaying it once the rename has happened touches nothing and
        still answers 200, because PMS-AI calls this fire-and-forget and cannot
        distinguish a lost reply from a failed rename.
        """
        _require_internal_anpr_auth(service_key)

        old_plate = (payload.from_plate or "").strip()
        new_plate = (payload.to_plate or "").strip()
        if not old_plate or not new_plate:
            raise HTTPException(status_code=400, detail="both plates are required")
        if old_plate == new_plate:
            return {
                "status": "ok", "gallery_renamed": False,
                "sessions_updated": 0, "slots_updated": 0,
            }

        store = getattr(registry, "gallery_store", None)
        gallery_renamed = False
        if store is not None:
            gallery_renamed = bool(store.rename(old_plate, new_plate))

        sessions_updated = 0
        try:
            sessions_updated = registry.rename_plate(old_plate, new_plate)
        except Exception as exc:  # a stale in-memory plate must not 500 the call
            print(f"[API] registry rename {old_plate} -> {new_plate} failed: {exc!r}")

        slots_updated = 0
        if db_manager is not None:
            session = db_manager.SessionLocal()
            try:
                from src.model import ParkingSlot

                slots_updated = (
                    session.query(ParkingSlot)
                    .filter(ParkingSlot.current_plate == old_plate)
                    .update({ParkingSlot.current_plate: new_plate},
                            synchronize_session=False)
                )
                session.commit()
            except Exception as exc:
                print(f"[API] slot plate rename {old_plate} -> {new_plate} "
                      f"failed: {exc!r}")
                session.rollback()
            finally:
                session.close()

        print(
            f"[API] plate corrected {old_plate} -> {new_plate} "
            f"(gallery={gallery_renamed} sessions={sessions_updated} "
            f"slots={slots_updated})"
        )
        return {
            "status": "ok",
            "gallery_renamed": gallery_renamed,
            "sessions_updated": sessions_updated,
            "slots_updated": slots_updated,
        }

    @app.post("/api/line-crossing", response_model=LineCrossingResponse)
    async def line_crossing(event: LineCrossingRequest):
        """
        Receive a B1-entrance line-crossing image from the external detector
        (CAM-03), sent AFTER the ANPR event. We detect & crop the entering car
        from the frame (it also contains parked cars), then seed the Park_Entry
        ReID candidate and bind it to the pending plate.

        ``plate`` is optional: when given it is logged for correlation; the
        crop is bound to the oldest pending ANPR entry either way.
        """
        import base64 as _b64
        import cv2
        import numpy as np

        print(f"\n{'='*60}")
        print(f"[API] LINE-CROSSING IMAGE RECEIVED (CAM={event.camera_id})")
        print(f"[API]   Plate (hint): {event.plate or 'N/A'}")
        print(f"[API]   Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        if not event.image_base64:
            raise HTTPException(status_code=400, detail="image_base64 is required")
        try:
            frame = cv2.imdecode(
                np.frombuffer(_b64.b64decode(event.image_base64), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        if frame is None or frame.size == 0:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Detect & crop the entering car (the frame also has parked cars). Fall
        # back to the whole frame if no detector is wired or nothing is found.
        crop = None
        if detect_vehicle_crop is not None:
            try:
                crop = detect_vehicle_crop(frame)
            except Exception as exc:
                print(f"[API] line-crossing vehicle crop failed: {exc!r}")
        cropped = crop is not None and getattr(crop, "size", 0) > 0
        car = crop if cropped else frame
        print(f"[API]   Vehicle crop: {'yes' if cropped else 'no (using full frame)'}")

        bound = _capture_instant_snapshot(
            event.plate or "",
            "entry",
            camera_id=event.camera_id or "CAM-03",
            frame=car,
        )
        return LineCrossingResponse(
            status="ok",
            plate=event.plate,
            cropped=cropped,
            bound=bool(bound),
            timestamp=(event.timestamp or datetime.now().isoformat()),
        )

    # ── Slot Status Endpoints ───────────────────────────────

    @app.get("/api/slots", response_model=List[SlotStatus])
    async def get_all_slots():
        """Get status of all parking slots across all cameras.

        MULTI-PROCESS: under supervisor.py the API runs inside ONE of five camera
        groups. ``get_slot_statuses()`` walks ``engine.pipelines`` and
        ``registry.get_slot_plate()`` reads the in-memory ``_parked`` map, so both
        only see the cameras THIS worker owns — every slot on another group's
        camera was silently reported as vacant-and-plateless. The DB is the only
        view of the whole facility, so the roster comes from ``parking_slots``
        (the owning worker persists plate/occupancy there), and this worker's live
        state is overlaid on top for the slots it does own.
        """
        statuses = get_slot_statuses() if get_slot_statuses is not None else []
        live = {s["slot_id"]: s for s in statuses if s.get("slot_id")}

        db_slots = {}
        if db_manager:
            _session = db_manager.SessionLocal()
            try:
                from src.repositories import ParkingSlotRepository
                db_slots = {
                    s.slot_id: s
                    for s in ParkingSlotRepository.get_all(_session)
                    # parking_slots also stores ROI masks and the Park_Entry /
                    # B1_Entrence special zones — never surface those as slots.
                    if getattr(s, "slot_type", "parking") == "parking"
                }
            finally:
                _session.close()

        # Every DB slot, plus any live slot not yet persisted (so a slot can
        # never vanish from the API just because its row is missing).
        roster = list(db_slots) + [sid for sid in live if sid not in db_slots]

        result = []
        for slot_id in roster:
            s = live.get(slot_id)
            db_slot = db_slots.get(slot_id)

            if s is not None:
                # This worker owns the camera: its registry is authoritative,
                # including "no plate" — do NOT fall back to a stale DB column.
                plate = registry.get_slot_plate(slot_id)
                occupied = bool(s.get("occupied", False))
                state = s.get("state", "UNKNOWN")
            else:
                plate = db_slot.current_plate
                occupied = not bool(db_slot.is_available)
                state = "OCCUPIED" if occupied else "VACANT"

            is_restricted = bool(
                (db_slot.reservation_type != "GENERAL" or db_slot.is_violation_zone)
                if db_slot is not None
                else (s or {}).get("is_violation_zone", False)
            )
            result.append(SlotStatus(
                slot_id=slot_id,
                slot_name=_first(
                    (s or {}).get("slot_name"), (s or {}).get("label"),
                    getattr(db_slot, "slot_name", None), slot_id,
                ),
                floor=_first((s or {}).get("floor"), getattr(db_slot, "floor", None)),
                zone_id=_first((s or {}).get("zone_id"), getattr(db_slot, "zone_id", None)),
                zone_name=_first((s or {}).get("zone_name"), getattr(db_slot, "zone_name", None)),
                occupied=occupied,
                state=state,
                plate_number=plate,
                camera_id=_first(
                    (s or {}).get("camera_id"), getattr(db_slot, "camera_id", None)
                ),
                is_restricted=is_restricted,
                restriction_type=get_slot_restriction_type(db_slot) if db_slot else None,
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
                    raw = db_slot.last_snapshot_path
                    # `last_snapshot_path` is now a public URL (e.g.
                    # http://host/pms-video-analytics/snapshots/slot_X_latest.jpg)
                    # built by engine_runtime._build_snapshot_url. Strip back to
                    # the snapshot_base_dir-relative path before joining onto
                    # disk. Tolerate legacy rows that hold a bare filename or a
                    # site-relative path with no scheme.
                    if "/snapshots/" in raw:
                        rel = raw.split("/snapshots/", 1)[1]
                    else:
                        rel = raw.lstrip("/")
                    image_path = os.path.join(snapshot_base_dir, rel)
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
        db_slots = {}
        if db_manager:
            _session = db_manager.SessionLocal()
            try:
                from src.repositories import ParkingSlotRepository
                db_slots = {s.slot_id: s for s in ParkingSlotRepository.get_all(_session)}
            finally:
                _session.close()
        result = []
        for s in statuses:
            if s.get("floor", "").upper() == floor.upper():
                slot_id = s.get("slot_id", "")
                plate = registry.get_slot_plate(slot_id)
                db_slot = db_slots.get(slot_id)
                is_restricted = bool(
                    (db_slot.reservation_type != "GENERAL" or db_slot.is_violation_zone)
                    if db_slot else s.get("is_violation_zone", False)
                )
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
                    is_restricted=is_restricted,
                    restriction_type=get_slot_restriction_type(db_slot) if db_slot else None,
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
    async def health(response: Response):
        """Enriched health reporting including engine status.

        The default ``ok`` is only a liveness signal (this API answered). When an engine
        status callback is wired in, its COMPUTED ``status`` overrides it and an
        ``unhealthy`` verdict is surfaced as HTTP 503 so orchestrators/monitors act on it
        instead of seeing a green 200 over a wedged engine.
        """
        health_data = {
            "status": "ok",
            "service": "Damanat PMS Video Analytics",
            "timestamp": datetime.now().isoformat()
        }
        if get_engine_status:
            health_data.update(get_engine_status())
        entry_state = active_entry_coordinator.state_summary()
        callback_load = (
            entry_state["pending_callback_count"]
            + entry_state["reserved_callback_count"]
        )
        callback_capacity = (
            active_entry_coordinator.settings.max_pending_callbacks
        )
        attempt_load = entry_state["attempt_count"]
        attempt_capacity = active_entry_coordinator.settings.max_pending_attempts
        crossing_load = (
            entry_state["crossing_count"]
            + entry_state.get("provisional_crossing_count", 0)
        )
        crossing_capacity = active_entry_coordinator.settings.max_pending_crossings
        journey_load = entry_state.get("journey_capacity_load", 0)
        journey_capacity = active_entry_coordinator.settings.journey_capacity
        pending_exit_count = entry_state.get("pending_exit_count", 0)
        health_data["entry_v2"] = {
            "mode": active_entry_coordinator.settings.mode.value,
            "available": active_entry_coordinator.available,
            "unavailable_reason": active_entry_coordinator.unavailable_reason,
            "attempt_count": entry_state["attempt_count"],
            "group_count": entry_state["group_count"],
            "crossing_count": entry_state["crossing_count"],
            "open_journey_count": entry_state.get("open_journey_count", 0),
            "finalized_journey_count": entry_state.get(
                "finalized_journey_count", 0
            ),
            "protected_journey_count": entry_state.get(
                "protected_journey_count", 0
            ),
            "journey_capacity_load": journey_load,
            "pending_exit_count": pending_exit_count,
            "ambiguous_exit_count": entry_state.get("ambiguous_exit_count", 0),
            "provisional_crossing_count": entry_state.get(
                "provisional_crossing_count", 0
            ),
            "pending_callback_count": entry_state["pending_callback_count"],
            "reserved_callback_count": entry_state["reserved_callback_count"],
            "analysis_inflight_count": entry_state["analysis_inflight_count"],
            "late_ocr_conflict_count": entry_state.get(
                "late_ocr_conflict_count", 0
            ),
            "permanent_callback_failure_count": entry_state[
                "permanent_callback_failure_count"
            ],
            "max_pending_attempts": attempt_capacity,
            "max_pending_callbacks": callback_capacity,
            "max_pending_crossings": crossing_capacity,
            "journey_capacity": journey_capacity,
        }
        # Entry V2/V3 conditions are collected here and applied to the
        # SERVICE verdict only when explicitly linked. Entry V2 is one pipeline
        # inside VideoAnalytics; in shadow it is observation-only and must not
        # be able to report the whole service degraded. `pending_exit_count > 0`
        # did exactly that — 39 exits retained from the two days v3 could not
        # confirm anything held VA amber while every camera, stream and
        # inference path was healthy, and the gateway aggregated that into an
        # overall "degraded".
        #
        # The information is not lost: the counters stay in the `entry_v2`
        # block and every condition below is reported under `entry_v2_reasons`
        # whether or not it is linked.
        mode_on = active_entry_coordinator.settings.mode.value != "off"
        entry_v2_reasons: list[str] = []
        entry_v2_severity = "ok"

        def _entry_v2(severity: str, reason: str) -> None:
            nonlocal entry_v2_severity
            entry_v2_reasons.append(reason)
            if severity == "unhealthy" or entry_v2_severity == "unhealthy":
                entry_v2_severity = "unhealthy"
            else:
                entry_v2_severity = "degraded"

        if mode_on and not active_entry_coordinator.available:
            _entry_v2(
                "unhealthy",
                "entry_v2 unavailable: "
                + (
                    active_entry_coordinator.unavailable_reason
                    or "unknown configuration/runtime error"
                ),
            )
        if callback_load > 0:
            _entry_v2(
                "unhealthy" if callback_load >= callback_capacity else "degraded",
                f"entry_v2 callback backlog: {callback_load}/{callback_capacity}",
            )
        if mode_on and attempt_load >= attempt_capacity:
            _entry_v2(
                "unhealthy",
                "entry_v2 attempt capacity exhausted: "
                f"{attempt_load}/{attempt_capacity}",
            )
        if mode_on and crossing_load >= crossing_capacity:
            _entry_v2(
                "unhealthy",
                "entry_v2 crossing capacity exhausted: "
                f"{crossing_load}/{crossing_capacity}",
            )
        if mode_on and (
            journey_load >= journey_capacity
            or pending_exit_count >= journey_capacity
        ):
            _entry_v2(
                "unhealthy",
                "entry_v2 journey lifecycle capacity exhausted: "
                f"journeys={journey_load}/{journey_capacity}, "
                f"pending_exits={pending_exit_count}/{journey_capacity}",
            )
        elif mode_on and pending_exit_count > 0:
            # Retained exits are ordinary operation, not impairment: an exit
            # whose entry was never confirmed has nothing to match and waits.
            # Capacity exhaustion above is the condition that actually hurts.
            _entry_v2(
                "degraded",
                "entry_v2 unmatched exit boundaries: "
                f"{pending_exit_count}/{journey_capacity}",
            )

        health_data["entry_v2_reasons"] = entry_v2_reasons
        health_data["entry_v2_status"] = entry_v2_severity
        health_data["entry_v2_linked_to_service_health"] = (
            active_entry_coordinator.settings.entry_v2_affects_service_health
        )
        if (
            active_entry_coordinator.settings.entry_v2_affects_service_health
            and entry_v2_reasons
        ):
            health_data["health_reasons"] = list(
                health_data.get("health_reasons") or []
            ) + entry_v2_reasons
            if entry_v2_severity == "unhealthy":
                health_data["status"] = "unhealthy"
            elif health_data.get("status") == "ok":
                health_data["status"] = "degraded"
        if health_data.get("status") == "unhealthy":
            response.status_code = 503
        return health_data

    return app


# --- Standalone Server ---

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
