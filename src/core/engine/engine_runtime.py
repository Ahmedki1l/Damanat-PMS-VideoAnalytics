import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.camera_manager import CameraConfig
from src.core.engine.camera_pipeline import CameraPipeline
from src.detection.tracking_manager import TrackingManager
from src.detection.detector import is_untracked
from src.models.state_machine import SlotState
from src.services.parking_service import (
    bootstrap_camera_slots_from_json,
    load_camera_slots,
)
from src.services.slot_status_service import log_vehicle_event, update_current_slot_plate
from src.vehicle_registry.vehicle_registry_identity import (
    is_identity_disabled,
    is_reid_disabled_floor,
)


logger = logging.getLogger(__name__)


class _SlotTrace:
    """End-to-end stage timings for one frame, emitted when a slot changes state.

    WHY THIS EXISTS. Slot flips were measured only at the state machine
    (``[SLOTLAT]``), which reported 0.61s and 1.01s on 2026-07-18 while the
    operator observed minutes. Those two facts are only compatible if the time is
    spent OUTSIDE the state machine — but every other stage between the camera
    and the database row the frontend reads was unmeasured, so "where" was
    unanswerable. This makes the whole chain visible in one line:

        capture -> submit -> infer -> queue -> consumer
                -> roi -> assign -> state_machine
                -> violation_filter -> db_commit -> event_bus

    ``capture_ts`` is the wall clock stamped by the grab thread, so
    ``total_ms`` is true photons-to-committed latency, not just in-process time.
    The upstream stamps are optional: the serial engine passes none, and those
    segments are simply omitted rather than reported as zero.

    Cost is a handful of ``perf_counter()`` calls per frame; the log line is
    emitted ONLY on an actual state change, which is rare.
    """

    __slots__ = ("cam_id", "capture_ts", "submitted_at", "inferred_at",
                 "dequeued_at", "stages", "db_commit_ms")

    def __init__(self, cam_id, capture_ts, submitted_at, inferred_at, dequeued_at):
        self.cam_id = cam_id
        self.capture_ts = capture_ts
        self.submitted_at = submitted_at
        self.inferred_at = inferred_at
        self.dequeued_at = dequeued_at
        self.stages: Dict[str, float] = {}
        self.db_commit_ms: Optional[float] = None

    def mark(self, name: str, since: float) -> float:
        """Record ms elapsed since ``since`` under ``name``; return a fresh stamp."""
        now = time.perf_counter()
        self.stages[name] = (now - since) * 1000.0
        return now

    def _upstream(self) -> str:
        """Segments before the consumer thread, from the wall-clock stamps."""
        out = []
        c, s, i, d = (self.capture_ts, self.submitted_at,
                      self.inferred_at, self.dequeued_at)
        if c and s:
            out.append(f"grab->submit={ (s - c) * 1000.0:.0f}ms")
        if s and i:
            out.append(f"infer={ (i - s) * 1000.0:.0f}ms")
        if i and d:
            out.append(f"out_q={ (d - i) * 1000.0:.0f}ms")
        return " ".join(out)

    def emit(self, events) -> None:
        """Log one ``[SLOTTRACE]`` line per state-change event in this frame."""
        if not events or not logger.isEnabledFor(logging.INFO):
            return
        stages = " ".join(f"{k}={v:.1f}ms" for k, v in self.stages.items())
        if self.db_commit_ms is not None:
            stages += f" db_commit={self.db_commit_ms:.1f}ms"
        upstream = self._upstream()
        # Photons -> row committed. This is the number the operator experiences.
        total = ""
        if self.capture_ts:
            total = f" TOTAL_capture->committed={ (time.time() - self.capture_ts) * 1000.0:.0f}ms"
        for ev in events:
            etype = getattr(ev, "event_type", "?")
            if etype not in ("vehicle_parked", "slot_vacant"):
                continue
            logger.info(
                "[SLOTTRACE] slot=%s cam=%s event=%s | %s | %s |%s",
                getattr(ev, "slot_id", "?"), self.cam_id, etype,
                upstream or "upstream=n/a", stages, total,
            )


class ParkingEngineRuntimeMixin:
    def _build_slot_snapshot_url(self, slot_id: str) -> str:
        return f"/api/slots/{slot_id}/snapshot/live"

    def _crop_vehicle_bbox_snapshot(
        self,
        frame,
        detection=None,
        bbox: Optional[tuple[float, float, float, float]] = None,
        padding_ratio: float = 0.12,
    ) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None

        source_bbox = bbox
        if source_bbox is None and detection is not None:
            source_bbox = tuple(float(v) for v in detection.bbox)
        if source_bbox is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in source_bbox]
        h, w = frame.shape[:2]
        pad_x = max(12, int((x2 - x1) * padding_ratio))
        pad_y = max(12, int((y2 - y1) * padding_ratio))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2].copy()
        return crop if crop.size > 0 else None

    def _crop_slot_snapshot(self, frame, slot) -> Optional[np.ndarray]:
        if frame is None or slot is None or frame.size == 0:
            return None

        polygon_points = np.array(
            [[int(x), int(y)] for x, y in slot.polygon.exterior.coords[:-1]],
            dtype=np.int32,
        )
        if polygon_points.size == 0:
            return None

        h, w = frame.shape[:2]
        x, y, width, height = cv2.boundingRect(polygon_points)
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        if width <= 0 or height <= 0:
            return None

        crop = frame[y:y + height, x:x + width].copy()
        shifted_points = polygon_points - np.array([x, y])
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [shifted_points], 255)
        masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
        return masked_crop if masked_crop.size > 0 else None

    def _save_slot_snapshot(
        self,
        frame,
        slot,
        detection=None,
        bbox: Optional[tuple[float, float, float, float]] = None,
    ) -> Optional[str]:
        crop = self._crop_vehicle_bbox_snapshot(frame, detection=detection, bbox=bbox)
        if crop is None or crop.size == 0:
            crop = self._crop_slot_snapshot(frame, slot)
        if crop is None or crop.size == 0:
            return None

        try:
            base_dir = self.config.output.snapshot_base_dir
            os.makedirs(base_dir, exist_ok=True)
            filename = f"slot_{slot.id}_latest.jpg"
            full_path = os.path.join(base_dir, filename)
            cv2.imwrite(full_path, crop)
            # Return the externally-reachable URL so the alerts table /
            # parking_slots.last_snapshot_path / Gateway responses all
            # carry full URLs instead of bare relative filenames that
            # frontends can't render directly.
            return self._build_snapshot_url(filename)
        except Exception as exc:
            print(f"[WARN] Failed to save slot snapshot for {slot.id}: {exc}")
            return None

    def _safe_snapshot_token(self, value: str, fallback: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "").strip("._")
        return cleaned or fallback

    def _build_snapshot_url(self, relative_path: str) -> str:
        """Turn a snapshot_base_dir-relative path (e.g. 'alerts/foo.jpg' or
        'slot_B11_CFO_latest.jpg') into the externally-reachable URL served
        by api.py's `/pms-video-analytics/snapshots/{filepath:path}` route.

        Reads `output.public_base_url`, `output.snapshot_url_prefix`, and
        `output.gateway_path_prefix` from config (same precedence as
        VehicleRegistryQueryMixin._get_snapshot_url) so the URL shape stays
        consistent across every consumer of VA snapshots — alerts, vehicle
        registry queries, and the API's own slot views.

        When `public_base_url` is empty, returns a site-relative URL
        (legacy behaviour, lets dev environments without an external host
        keep working).
        """
        if not relative_path:
            return ""
        rel = relative_path.replace(os.sep, "/").lstrip("/")
        out = self.config.output
        base = (getattr(out, "public_base_url", "") or "").rstrip("/")
        gateway = (getattr(out, "gateway_path_prefix", "") or "").strip("/")
        prefix = (getattr(out, "snapshot_url_prefix", "snapshots") or "snapshots").strip("/")
        path_parts = "/".join(part for part in [gateway, prefix, rel] if part)
        return f"{base}/{path_parts}" if base else f"/{path_parts}"

    def _save_alert_snapshot(self, crop, alert_type: str, slot_id: str, camera_id: str,
                             fallback_frame=None) -> Optional[str]:
        """Save the alert evidence image to disk, return its public URL.

        Prefers the vehicle-bbox `crop` (tighter framing for the operator).
        If the crop is empty/None — which happens when the vehicle's bbox is
        missing for the frame the alert fired on (Bug #v0-1 in Version 0:
        intrusion alerts landing without an evidence snapshot) — falls back
        to `fallback_frame` (the full camera frame). The full frame is wider
        but always meaningful, so the alert never lands snapshot-less.

        The returned value is the externally-reachable URL (e.g.
        ``http://localhost:8000/pms-video-analytics/snapshots/alerts/<file>.jpg``)
        built from `output.public_base_url` + `output.snapshot_url_prefix`,
        so callers can persist it directly to `alerts.snapshot_path` and
        downstream consumers (Gateway, frontend) render it without further
        rewriting. When `public_base_url` is empty, returns a site-relative
        URL — legacy behaviour for dev environments.

        Returns None only if BOTH crop and fallback_frame are empty/None, or
        if cv2.imwrite fails on disk.
        """
        # Decide which image to save: crop preferred, full frame as fallback.
        image = None
        if crop is not None and crop.size > 0:
            image = crop
        elif fallback_frame is not None and fallback_frame.size > 0:
            image = fallback_frame
            print(
                f"[INFO] Alert snapshot fallback: full frame for "
                f"{alert_type} ({camera_id} / {slot_id}); vehicle bbox crop was empty."
            )
        else:
            print(
                f"[WARN] Alert snapshot UNAVAILABLE: no crop and no fallback frame for "
                f"{alert_type} ({camera_id} / {slot_id})."
            )
            return None

        try:
            base_dir = self.config.output.snapshot_base_dir
            directory = os.path.join(base_dir, "alerts")
            os.makedirs(directory, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = (
                f"{self._safe_snapshot_token(alert_type, 'alert')}_"
                f"{self._safe_snapshot_token(slot_id, 'slot')}_"
                f"{self._safe_snapshot_token(camera_id, 'camera')}_"
                f"{timestamp}.jpg"
            )
            relative_path = os.path.join("alerts", filename)
            full_path = os.path.join(directory, filename)
            if not cv2.imwrite(full_path, image):
                raise RuntimeError("cv2.imwrite returned False")
            # Return the externally-reachable URL so consumers can render
            # the snapshot directly without rewriting the path. The same
            # file is still available on disk under snapshot_base_dir/alerts.
            return self._build_snapshot_url(relative_path)
        except Exception as exc:
            print(
                f"[WARN] Failed to save alert snapshot for {alert_type} "
                f"({camera_id} / {slot_id}): {exc}"
            )
            return None

    def _persist_slot_snapshot_path(self, slot_id: str, snapshot_filename: str) -> None:
        if not self.db_manager or not snapshot_filename:
            return

        session = self.db_manager.SessionLocal()
        try:
            from src.repositories import ParkingSlotRepository

            db_slot = ParkingSlotRepository.get_by_id(session, slot_id)
            if db_slot is None:
                return
            db_slot.last_snapshot_path = snapshot_filename
            session.commit()
        except Exception as exc:
            session.rollback()
            print(f"[ERROR] Failed to persist slot snapshot path for {slot_id}: {exc}")
        finally:
            session.close()

    # In-process rate gate: at most one vehicles-row write per plate per
    # _PRESENCE_MIN_INTERVAL_S seconds. Without this, the per-frame loop
    # would issue an UPDATE every camera tick (~14/s across all cameras)
    # for every actively-tracked plate, which is pointless DB churn.
    _PRESENCE_MIN_INTERVAL_S = 5.0

    # Exit-janitor cadence — how often the engine sweeps the registry to
    # purge plates whose parking_sessions row has been closed by PMS-AI
    # without VA seeing the corresponding ANPR exit event.
    _EXIT_JANITOR_INTERVAL_S = 30.0

    # How often a worker re-checks the DB for cars that entered AFTER it booted.
    # Must stay well under the gate->slot drive time (minutes) so the entrant's
    # session is in the ReID pool before it reaches whichever camera watches the
    # slot it parks in. See _session_sync_tick.
    _SESSION_SYNC_INTERVAL_S = 10.0

    def _open_session_rows(self):
        """``(plate_number, floor)`` for every car currently inside, or None on error.

        Deliberately unfiltered by camera: a worker must learn about cars that
        entered through a gate it does not own (see _session_sync_tick).
        """
        try:
            from sqlalchemy import text as _text

            session = self.db_manager.SessionLocal()
            try:
                # Runs on the main-loop thread, so it must never block the
                # pipeline. LOCK_TIMEOUT makes a lock-blocked read raise after
                # 3s instead of hanging; NOLOCK avoids taking shared locks at
                # all. A stale read is fine — we re-poll (see _exit_janitor_tick).
                session.execute(_text("SET LOCK_TIMEOUT 3000"))
                return session.execute(
                    _text(
                        "SELECT plate_number, floor FROM dbo.parking_sessions WITH (NOLOCK) "
                        "WHERE status = 'open'"
                    )
                ).fetchall()
            finally:
                session.close()
        except Exception as exc:
            logger.warning("[gallery] open-session query failed: %r", exc)
            return None

    def _recently_entered_from_gallery(self):
        """``(plate, floor, entered_at)`` for cars whose ON-DISK gallery shows a fresh
        ANPR gate shot — i.e. cars VA itself just watched enter.

        supervisor.py's contract is that identity crosses the process boundary through
        the DB **and the on-disk gallery**. Until now the sync used only the DB half:
        ``_open_session_rows`` selects ``parking_sessions WHERE status='open'``, a table
        PMS-AI owns. On 2026-07-11 PMS-AI stopped inserting rows altogether (its last
        row was a *closed* one from 17:00, while it kept POSTing ANPR events happily) —
        so the gate worker knew DJS-7842, and the worker owning the camera it actually
        parked on had never heard of it. The car was invisible where it mattered.

        The gallery does not have that problem: the gate seeds ``gallery/<plate>/`` with
        the ANPR shot the moment the car is read, and every worker can see it. So VA can
        answer "who just drove in" from its OWN evidence, with no dependency on PMS-AI
        writing anything.

        The ANPR ref's timestamp IS the gate-read time, which also lets a non-gate worker
        populate ``_last_anpr_entry_at`` — without it the car is never "in flight" there
        and the elimination cannot fire.
        """
        registry = self.vehicle_registry
        store = getattr(registry, "gallery_store", None)
        if store is None:
            return []
        root = getattr(store, "_root", None)
        if not root or not os.path.isdir(root):
            return []

        gt_cams = {
            str(c).upper()
            for c in getattr(self.config.matching, "ground_truth_cameras", ()) or ()
            if "ANPR" in str(c).upper()
        } or {"ANPR"}
        window = float(
            getattr(self.config.matching, "slot_acquire_inflight_seconds", 300.0) or 300.0
        )
        now = datetime.now()
        out = []
        try:
            for entry in os.scandir(root):
                if not entry.is_dir():
                    continue
                meta_path = os.path.join(entry.path, "meta.json")
                if not os.path.isfile(meta_path):
                    continue
                try:
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                except Exception:
                    continue
                plate = meta.get("plate")
                if not plate:
                    continue
                stamps = [
                    r.get("ts")
                    for r in meta.get("refs", []) or []
                    if str(r.get("camera", "")).upper() in gt_cams and r.get("ts")
                ]
                if not stamps:
                    continue
                try:
                    entered_at = datetime.fromisoformat(max(stamps))
                except Exception:
                    continue
                if (now - entered_at).total_seconds() <= window:
                    out.append((plate, None, entered_at))
        except Exception as exc:
            logger.debug("[gallery] recent-entry scan failed: %r", exc)
        return out

    def _hydrate_open_sessions(self, rows) -> int:
        """``build_session_from_gallery`` for each open plate not yet hydrated here.

        Returns how many NEW sessions were built. Plates whose gallery has no
        matchable reference yet (an entrant with only the ``gate_only`` shot)
        are deliberately NOT marked as seen, so a later tick picks them up once
        CAM-03 adds the first matchable crop.
        """
        hydrated = getattr(self, "_hydrated_plates", None)
        if hydrated is None:
            hydrated = self._hydrated_plates = set()

        added, still_open = 0, set()
        for row in rows:
            plate = row[0]
            floor = row[1] if len(row) > 1 else None
            if not plate:
                continue
            still_open.add(plate)
            if plate in hydrated:
                continue
            try:
                if self.vehicle_registry.build_session_from_gallery(plate, floor=floor):
                    hydrated.add(plate)
                    added += 1
            except Exception as exc:
                logger.warning("[gallery] restore failed for plate=%s: %r", plate, exc)
        # Forget plates whose session has closed, so a re-entry re-hydrates.
        hydrated &= still_open
        return added

    def _restore_vehicle_galleries(self) -> None:
        """Reload the persisted per-plate gallery for every car still inside the
        facility (open ``parking_sessions``) so ReID can re-identify it after a
        restart. Enriches the vectorless sessions created by
        ``_restore_plate_locks``. No-op when the gallery feature is off or there
        is no DB."""
        if not self.db_manager or not self.vehicle_registry:
            return
        if getattr(self.vehicle_registry, "gallery_store", None) is None:
            return
        rows = self._open_session_rows()
        if rows is None:
            return
        restored = self._hydrate_open_sessions(rows)
        if restored:
            print(f"[INFO] Restored {restored} vehicle gallery(ies) from disk (cars still inside).")

    def _session_sync_tick(self) -> None:
        """Pick up cars that entered AFTER this worker booted.

        Only the ``--api`` group runs the ANPR webhook, so only THAT process
        learns a new entrant's plate and builds its session. Every other group
        owns the slot cameras the car actually parks on, and was running global
        ReID against a session pool frozen at its own startup: the car parks,
        matches nothing, is given an anonymous ``plate=None`` session, and the
        slot binds to None — rendering as "(pending)" forever. (It fails
        silently because ``is_plate_inside(None)`` returns True.)

        supervisor.py's contract is that identity crosses the process boundary
        through the DB + the on-disk gallery. ``_restore_vehicle_galleries`` is
        the read side of that contract — it was just wired to a one-shot boot
        call. This runs it on a timer, so every worker sees an entrant within
        ``_SESSION_SYNC_INTERVAL_S``, long before the car reaches its slot.
        """
        if not self.db_manager or not self.vehicle_registry:
            return
        if getattr(self.vehicle_registry, "gallery_store", None) is None:
            return

        now_ts = time.time()
        last = getattr(self, "_session_sync_last_run_at", 0.0)
        if now_ts - last < self._SESSION_SYNC_INTERVAL_S:
            return
        self._session_sync_last_run_at = now_ts

        # Refresh this worker's view of the plate locks held by every OTHER worker, so a
        # car already parked on a camera we don't own is excluded as a candidate here.
        self._refresh_external_plate_locks()

        # A car that relocated released its previous slot in memory; null that slot's DB
        # row too, or /api/slots reports the same car in two places.
        for slot_id in self.vehicle_registry.take_released_slots():
            if self.db_manager:
                self._persist_slot_plate_binding(slot_id, None, 0.0, False, "")

        rows = self._open_session_rows()
        if rows is None:
            rows = []

        # VA's OWN evidence, independent of PMS-AI ever writing a row. Also carries the
        # gate-read time, which this worker needs in _last_anpr_entry_at before the car
        # can count as "in flight" for slot acquisition.
        known = {r[0] for r in rows if r and r[0]}
        for plate, floor, entered_at in self._recently_entered_from_gallery():
            prior = self.vehicle_registry.last_anpr_entry_at(plate)
            if prior is None or entered_at > prior:
                self.vehicle_registry._last_anpr_entry_at[plate] = entered_at
            if plate not in known:
                rows.append((plate, floor))
                known.add(plate)

        if not rows:
            return
        added = self._hydrate_open_sessions(rows)
        if added:
            print(f"[INFO] Session sync: picked up {added} car(s) that entered "
                  f"after this worker started.")

    def _refresh_external_plate_locks(self) -> None:
        """Mirror ``parking_slots``' plate locks into the registry.

        One small query on the tick that already opens a DB session, so this costs no
        new polling and no new connection. Best-effort: on failure we keep the previous
        map rather than clearing it — a momentarily stale lock is a far smaller problem
        than suddenly un-excluding every parked car at once.
        """
        if not self.db_manager or not self.vehicle_registry:
            return
        session = self.db_manager.SessionLocal()
        try:
            from src.repositories import ParkingSlotRepository

            locks = {}
            for row in ParkingSlotRepository.get_plate_locks(session):
                plate = (row.current_plate or "").strip()
                if not plate:
                    continue
                locks[plate] = {
                    "slot_id": row.slot_id,
                    "camera_id": row.camera_id or "",
                    "locked": bool(row.plate_locked),
                    "locked_at": row.plate_locked_at,
                }
            self.vehicle_registry.set_external_plate_locks(locks)
        except Exception as exc:
            logger.debug("[plate-lock] external lock refresh failed: %r", exc)
        finally:
            session.close()

    def _exit_janitor_tick(self) -> None:
        """Once per `_EXIT_JANITOR_INTERVAL_S`, find plates VA still has in
        memory whose latest parking_sessions row is closed (per PMS-AI), and
        call vehicle_registry._handle_exit(plate, now) to purge the in-memory
        tracking state. Catches missed CAM-EXIT ANPR events and stops VA from
        re-id-matching cars that have already left the garage.

        Called from the main loop (next to _cleanup_stale_data). The gate
        ensures it doesn't run on every frame.
        """
        if not self.db_manager or not self.vehicle_registry:
            return
        now_ts = time.time()
        last = getattr(self, "_exit_janitor_last_run_at", 0.0)
        if now_ts - last < self._EXIT_JANITOR_INTERVAL_S:
            return
        self._exit_janitor_last_run_at = now_ts

        # Age out per-plate gallery folders idle past the retention TTL (checked
        # hourly, piggy-backing the janitor cadence). No-op when disabled.
        store = getattr(self.vehicle_registry, "gallery_store", None)
        if store is not None:
            gc_last = getattr(self, "_gallery_gc_last_run_at", 0.0)
            if now_ts - gc_last > 3600.0:
                self._gallery_gc_last_run_at = now_ts
                try:
                    store.gc(self.config.matching.gallery_retention_days)
                except Exception as exc:
                    logger.warning("[gallery] GC failed: %r", exc)

        # Snapshot the plates the registry currently holds. Done under the
        # registry's lock-protected accessor (or via a stable copy) so we
        # don't iterate a dict that another thread is mutating.
        try:
            tracked_plates = self.vehicle_registry.get_tracked_plates()
        except AttributeError:
            # Older registry without the helper — fall back to _parked +
            # session map plates.
            tracked_plates = set()
            for sess in getattr(self.vehicle_registry, "_parked", {}).values():
                if sess.plate:
                    tracked_plates.add(sess.plate)
            for sess in getattr(self.vehicle_registry, "_sessions", {}).values():
                if sess.plate:
                    tracked_plates.add(sess.plate)
        if not tracked_plates:
            return

        try:
            from sqlalchemy import bindparam, text as _text

            session = self.db_manager.SessionLocal()
            try:
                # This runs on the main loop thread, so it must never block the
                # pipeline. LOCK_TIMEOUT makes a lock-blocked read raise (err
                # 1222) after 3s instead of hanging forever (and un-killably,
                # since the wait is a native ODBC socket call); the NOLOCK hint
                # avoids taking shared locks at all. A stale/dirty read is fine —
                # a plate that just closed is re-checked on the next tick.
                session.execute(_text("SET LOCK_TIMEOUT 3000"))
                # One round-trip: get the latest status per plate. Plates
                # with no rows aren't in the result — those are fine, they
                # haven't entered yet. `expanding=True` lets SQLAlchemy
                # turn the IN binding into a parameterized list at execute time.
                # entry_time comes back too: it is what tells us whether this
                # 'closed' row is STALE. See the staleness check below.
                stmt = _text(
                    "SELECT plate_number, status, entry_time FROM ("
                    "  SELECT plate_number, status, entry_time, "
                    "         ROW_NUMBER() OVER (PARTITION BY plate_number ORDER BY entry_time DESC) AS rn "
                    "  FROM dbo.parking_sessions WITH (NOLOCK) "
                    "  WHERE plate_number IN :plates"
                    ") t WHERE rn = 1"
                ).bindparams(bindparam("plates", expanding=True))
                rows = session.execute(stmt, {"plates": list(tracked_plates)}).fetchall()
            finally:
                session.close()
        except Exception as exc:
            logger.warning("[exit_janitor] DB probe failed: %r", exc)
            return

        closed = [(r[0], r[2]) for r in rows if r[1] == "closed"]
        if not closed:
            return

        purged_at = datetime.now()
        for plate, row_entry_time in closed:
            try:
                # Is this 'closed' row STALE — i.e. does it pre-date the last time VA
                # itself watched this car drive through the gate?
                #
                # The old check was TIME-based ("did it re-enter in the last 60s?") and
                # it silently deleted live cars. On 2026-07-11 DJS-7842 re-entered at
                # 17:49:06, was still driving to its slot 77s later, and the janitor
                # purged its identity — killing the plate bind, the gallery capture and
                # every downstream fix. Worse, PMS-AI had stopped inserting rows
                # entirely: the newest row was a 'closed' one from 17:00, so NO grace
                # value could ever have saved the car. A timer cannot express this.
                #
                # Ordering can. If VA's own ANPR saw this plate enter AFTER the row's
                # entry_time, the row describes an OLDER visit and says nothing about
                # the car now inside. VA's gate reads are authoritative for "is it in";
                # the DB is authoritative only for a visit it actually recorded. This
                # tolerates PMS-AI lag of any length — including forever — while still
                # purging a car that genuinely left after its most recent entry.
                va_entry_at = None
                getter = getattr(self.vehicle_registry, "last_anpr_entry_at", None)
                if getter is not None:
                    va_entry_at = getter(plate)
                if (
                    va_entry_at is not None
                    and row_entry_time is not None
                    and va_entry_at > row_entry_time
                ):
                    logger.info(
                        "[exit_janitor] skip purge plate=%s: STALE closed row "
                        "(row entry=%s, but VA saw this car enter the gate at %s) — "
                        "PMS-AI has not opened the new row yet",
                        plate, row_entry_time, va_entry_at,
                    )
                    continue
                # Belt-and-braces: the original in-memory grace still applies for the
                # case where the DB has no usable entry_time at all.
                if getattr(self.vehicle_registry, "has_recent_reentry", None) and \
                        self.vehicle_registry.has_recent_reentry(plate):
                    logger.debug(
                        "[exit_janitor] skip purge plate=%s: fresh re-entry "
                        "within grace (DB open row still lagging)",
                        plate,
                    )
                    continue
                self.vehicle_registry._handle_exit(plate, purged_at)
                logger.info(
                    "[exit_janitor] purged in-memory state for plate=%s "
                    "(parking_sessions.status=closed)",
                    plate,
                )
            except Exception as exc:
                logger.warning("[exit_janitor] _handle_exit(%s) failed: %r", plate, exc)

    def update_vehicle_presence(
        self,
        plate: str,
        *,
        floor: Optional[str] = None,
        camera_id: Optional[str] = None,
    ) -> None:
        """Write `vehicles.floor` / `floor_id` AND mirror the same value onto
        the OPEN `parking_sessions` row so the Gateway's entry-exit endpoint
        (which JOINs parking_sessions for floor / floor_id / parked_at) shows
        the live floor even before a slot is bound.

        Called from every track-confirmation path (Park_Entry capture,
        B1_Entrence confirmation, slot bind, plus the per-frame
        TrackingManager observation). The slot-bind path is also written by
        PMS-AI's parking_session_service.bind_slot — both sources are
        idempotent so they can race safely.

        Rate-gated to once per plate per ~5s so per-frame callers don't
        hammer the DB. Safe to call with floor=None — only updates fields
        that are actually known.

        `parked_at` semantics: set ONLY when currently NULL on the open
        session (i.e. the session has no slot-bind timestamp yet). This
        marks "first floor observation" for sessions that arrive at the
        slot detection cameras before any slot bind, while preserving the
        slot-bind timestamp once bind_slot has set it.
        """
        if not plate or not self.db_manager:
            return

        # Lazy-init the per-plate gate map.
        gate = getattr(self, "_presence_last_write_at", None)
        if gate is None:
            gate = {}
            self._presence_last_write_at = gate
        now_ts = time.time()
        last = gate.get(plate, 0.0)
        if now_ts - last < self._PRESENCE_MIN_INTERVAL_S:
            return
        gate[plate] = now_ts

        session = self.db_manager.SessionLocal()
        try:
            from sqlalchemy import text as _text

            # Check if vehicle exists (raw SQL — VA has no Vehicle ORM model;
            # the vehicles table is owned by the Gateway's schema).
            row = session.execute(
                _text("SELECT id, floor FROM dbo.vehicles WHERE plate_number = :p"),
                {"p": plate},
            ).first()
            if row is None:
                # No registry row yet — VA's Park_Entry pipeline will create
                # it once ANPR matches. Don't create a partial row here.
                return

            vehicle_id, current_floor = row
            if not floor or current_floor == floor:
                return

            # Resolve floor_id from the floors lookup table once; reused for
            # both the vehicles UPDATE and the parking_sessions UPDATE.
            fid = session.execute(
                _text("SELECT id FROM dbo.floors WHERE name = :n"),
                {"n": floor},
            ).scalar()

            # 1. vehicles row (canonical "where is the car right now").
            session.execute(
                _text("UPDATE dbo.vehicles SET floor = :f, floor_id = :fid WHERE id = :vid"),
                {"f": floor, "fid": fid, "vid": vehicle_id},
            )

            # 2. open parking_sessions row (drives the Gateway's entry-exit
            #    response shape via JOIN). Only one open session per plate
            #    by invariant (UC1 dedup + close_session). Update the latest
            #    open row; no-op if the plate isn't currently inside.
            #    parked_at is COALESCE so a slot-bind timestamp from
            #    parking_session_service.bind_slot wins; we only fill it in
            #    the gap where VA observed the car on a floor before the
            #    slot detection camera reported a bind.
            # Bind a Python facility-local naive datetime instead of using
            # MSSQL's SYSUTCDATETIME() — the DB convention is naive facility-
            # local (operator wall clock), and SYSUTCDATETIME() returns UTC
            # which would land 3h behind the wall clock.
            from src.utils.datetime_helper import facility_now_naive
            now_naive = facility_now_naive()
            session.execute(
                _text(
                    "UPDATE dbo.parking_sessions "
                    "SET floor = :f, floor_id = :fid, "
                    "    parked_at = COALESCE(parked_at, :now), "
                    "    updated_at = :now "
                    "WHERE plate_number = :p AND status = 'open'"
                ),
                {"f": floor, "fid": fid, "p": plate, "now": now_naive},
            )

            session.commit()
            logger.debug(
                "[presence] plate=%s floor=%s camera=%s (vehicles + parking_sessions)",
                plate, floor, camera_id,
            )
        except Exception as exc:
            session.rollback()
            logger.warning("[presence] write failed for plate=%s: %r", plate, exc)
        finally:
            session.close()

    def _build_camera_configs(self) -> List[CameraConfig]:
        camera_configs: List[CameraConfig] = []
        for camera in self.config.cameras:
            camera_config = CameraConfig(
                id=camera.id,
                name=camera.name,
                floor=camera.floor,
                ip=camera.ip,
                user=camera.user,
                password=camera.password,
                slots_file=camera.slots_file,
                rtsp_port=camera.rtsp_port,
            )
            camera_config.build_rtsp_url(channel=self.config.processing.stream_channel)
            camera_configs.append(camera_config)
        return camera_configs

    def _initialize_camera_pipelines(self, camera_configs: List[CameraConfig]) -> int:
        total_slots = 0
        all_active_slot_ids = set()

        for camera_config in camera_configs:
            pipeline, parking_slots = self._build_camera_pipeline(
                camera_config,
                all_active_slot_ids,
            )
            self.pipelines[camera_config.id] = pipeline
            total_slots += pipeline.slot_count
            for slot in parking_slots:
                all_active_slot_ids.add(slot.id)

        self._free_plates_on_disabled_cameras()
        return total_slots

    def _free_plates_on_disabled_cameras(self) -> None:
        """One-shot startup cleanup: for cameras whose plate matching is
        disabled (ground-floor cameras, via ``is_reid_disabled_floor``), clear
        any plate left bound to one of their slots in ``slot_status`` and notify
        the PMS API to unbind. The slot's ``is_available`` flag is left
        untouched — only the plate identity is freed. Idempotent: a no-op if
        there are no stale bindings."""
        if not self.db_manager:
            return
        cleared = []
        session = self.db_manager.SessionLocal()
        try:
            from src.repositories import SlotStatusRepository

            for cam_id, pipeline in self.pipelines.items():
                if not is_reid_disabled_floor(pipeline.floor):
                    continue
                for slot in pipeline.slots:
                    latest = SlotStatusRepository.get_latest_by_slot(session, slot.id)
                    if not latest or latest.status != "occupied":
                        continue
                    if not latest.plate_number:
                        continue
                    previous_plate = latest.plate_number
                    update_current_slot_plate(
                        session,
                        slot_id=slot.id,
                        plate=None,
                        camera_id=cam_id,
                    )
                    cleared.append((cam_id, slot.id, previous_plate))
        except Exception as exc:
            session.rollback()
            print(f"[ERROR] Failed to free plates on disabled cameras: {exc}")
        finally:
            session.close()

        for cam_id, slot_id, plate in cleared:
            print(
                f"[INFO] Cleared stale plate {plate!r} from {cam_id}/{slot_id} "
                f"(plate matching disabled on this camera)"
            )
        if cleared:
            print(
                f"[INFO] Freed {len(cleared)} slot plate binding(s) on startup "
                f"for cameras with matching disabled"
            )

    def _build_camera_pipeline(self, camera_config: CameraConfig, all_active_slot_ids: set):
        # Reference resolution (what the slot JSONs were drawn at)
        ref_res = (
            self.config.processing.slot_ref_width,
            self.config.processing.slot_ref_height,
        )

        # Actual stream resolution — read from the camera stream if available
        actual_res = None
        if hasattr(self, "cam_manager"):
            w, h = self.cam_manager.get_resolution(camera_config.id)
            if w > 0 and h > 0:
                actual_res = (w, h)

        parking_slots = []
        special_zones = []
        roi_polygon = None
        boundaries = []
        self._bootstrap_camera_slots_if_needed(camera_config)
        if self.db_manager:
            session = self.db_manager.SessionLocal()
            try:
                parking_slots, special_zones, roi_polygon, boundaries = load_camera_slots(
                    session,
                    camera_id=camera_config.id,
                    ref_resolution=ref_res,
                    actual_resolution=actual_res,
                )
            except Exception as exc:
                print(f"[ERROR] Failed to load slots from database for {camera_config.id}: {exc}")
            finally:
                session.close()

        self.special_zones[camera_config.id] = {zone.id: zone for zone in special_zones}
        if special_zones:
            print(
                f"[INFO] {camera_config.id} has {len(special_zones)} special zone(s): "
                f"{[zone.id for zone in special_zones]}"
            )

        # Zoning: store per-camera boundary polygons (area-to-area crossings).
        # Consumed by the BoundaryCrossingDetector when zoning is active; with
        # no boundaries authored this is simply empty (no behaviour change).
        if not hasattr(self, "boundaries"):
            self.boundaries = {}
        self.boundaries[camera_config.id] = {b.id: b for b in boundaries}
        if boundaries:
            print(
                f"[INFO] {camera_config.id} has {len(boundaries)} boundary zone(s): "
                f"{[(b.id, b.area_from, b.area_to) for b in boundaries]}"
            )

        violation_slots, initial_statuses, plate_bindings = self._load_camera_db_state(
            parking_slots,
            all_active_slot_ids,
        )

        pipeline = CameraPipeline(
            camera_id=camera_config.id,
            floor=camera_config.floor,
            slots=parking_slots,
            config=self.config,
            violation_slots=violation_slots,
            initial_statuses=initial_statuses,
            roi_polygon=roi_polygon,
        )
        # Restart recovery: re-seed persisted plate-lock bindings into the fresh
        # state machines + registry so a restart doesn't drop or drift plates.
        self._restore_plate_locks(pipeline, plate_bindings, camera_config.floor)
        return pipeline, parking_slots

    def _restore_plate_locks(self, pipeline, plate_bindings, floor) -> None:
        """Seed persisted plate bindings (locked or provisional) back into the
        pipeline's state machines and the registry after a restart.

        Ground floors run occupancy only (ReID disabled) and have no registry
        bindings — skip them. If the car actually left during downtime, the
        slot starts OCCUPIED and the normal LEAVING→VACANT debounce clears +
        unlinks it, so no stale lock survives a departed car.
        """
        if not plate_bindings or not self.vehicle_registry:
            return
        if is_reid_disabled_floor(floor):
            return
        # Slots whose plate was restored from persistence but not yet re-derived
        # by the (freshly empty) ReID gallery. Protected from the per-frame
        # "clear on no-match" path so a restart doesn't drop the plate before a
        # live track re-confirms it; cleared once re-derived or the slot vacates.
        if not hasattr(self, "_restored_plate_slots"):
            self._restored_plate_slots = set()
        now = datetime.now()
        for slot_id, b in plate_bindings.items():
            sm = pipeline.state_machines.get(slot_id)
            if sm is None:
                continue
            sm.bind_identity(
                b["plate"],
                self._build_slot_snapshot_url(slot_id),
                confidence=b["confidence"],
                lock=b["locked"],
            )
            self._restored_plate_slots.add(slot_id)
            self.vehicle_registry.restore_parked_binding(
                slot_id=slot_id,
                slot_name=b["slot_name"],
                plate=b["plate"],
                confidence=b["confidence"],
                camera_id=b["camera_id"] or pipeline.camera_id,
                floor=floor,
                locked=b["locked"],
                timestamp=b["locked_at"] or now,
            )
            print(
                f"[RESTORE] slot={slot_id} plate={b['plate']} "
                f"conf={b['confidence']:.2f} locked={b['locked']}"
            )

    def _bootstrap_camera_slots_if_needed(self, camera_config) -> None:
        if not self.db_manager:
            return

        session = self.db_manager.SessionLocal()
        try:
            from src.repositories import ParkingSlotRepository

            existing_rows = ParkingSlotRepository.filter_camera_slots(session, camera_config.id)
            if existing_rows:
                return

            migrated = bootstrap_camera_slots_from_json(
                session,
                camera_id=camera_config.id,
                floor=camera_config.floor,
                slots_file=camera_config.slots_file,
                default_zone_id=camera_config.name,
                default_zone_name=camera_config.name,
            )
            if migrated:
                print(
                    f"[DB] Bootstrapped slot definitions for {camera_config.id} "
                    f"from legacy JSON '{camera_config.slots_file}'"
                )
        except Exception as exc:
            session.rollback()
            print(f"[ERROR] Failed to bootstrap slots for {camera_config.id}: {exc}")
        finally:
            session.close()

    def _load_camera_db_state(self, parking_slots, all_active_slot_ids: set):
        violation_slots = set()
        initial_statuses = {}
        # Restart recovery: persisted plate-lock bindings for occupied slots,
        # seeded back into the state machines + registry after the pipeline is
        # built (see _restore_plate_locks). slot_id -> binding dict.
        plate_bindings = {}
        reserved_for_map = getattr(self, "_reserved_for_map", {})
        special_slots = getattr(self, "_special_slots", set())

        if not self.db_manager or not parking_slots:
            return violation_slots, initial_statuses, plate_bindings

        session = self.db_manager.SessionLocal()
        try:
            from src.repositories import ParkingSlotRepository

            slot_ids = [slot.id for slot in parking_slots]
            db_slots = [ParkingSlotRepository.get_by_id(session, slot_id) for slot_id in slot_ids]
            for db_slot in db_slots:
                if not db_slot:
                    continue
                if db_slot.is_violation_zone:
                    violation_slots.add(db_slot.slot_id)
                if db_slot.reservation_type == "EMPLOYEE":
                    violation_slots.add(db_slot.slot_id)
                    reserved_for_map[db_slot.slot_id] = db_slot.reserved_for
                elif db_slot.reservation_type == "SPECIAL":
                    violation_slots.add(db_slot.slot_id)
                    special_slots.add(db_slot.slot_id)
                initial_statuses[db_slot.slot_id] = db_slot.is_available
                all_active_slot_ids.add(db_slot.slot_id)

                # A persisted plate on an occupied slot must be restored so the
                # API reports it immediately and the first post-restart frame
                # can't overwrite it with a fresh provisional guess.
                if (not db_slot.is_available) and getattr(db_slot, "current_plate", None):
                    plate_bindings[db_slot.slot_id] = {
                        "plate": db_slot.current_plate,
                        "confidence": float(getattr(db_slot, "plate_confidence", 0.0) or 0.0),
                        "locked": bool(getattr(db_slot, "plate_locked", False)),
                        "locked_at": getattr(db_slot, "plate_locked_at", None),
                        "slot_name": db_slot.slot_name or db_slot.slot_id,
                        "camera_id": db_slot.camera_id or "",
                    }
                elif not db_slot.is_available:
                    # Occupied, but we never learned who it is. Only vehicle_parked arms
                    # the identity pass, and that transition already happened — before this
                    # worker booted. Without arming here the slot stays anonymous until the
                    # car leaves and someone else parks, which for a long-stay car is all day.
                    self._arm_ocr_for_slot(db_slot.slot_id)
        except Exception as exc:
            print(f"[ERROR] Failed to load initial slot states from DB: {exc}")
        finally:
            session.close()

        self._reserved_for_map = reserved_for_map
        self._special_slots = special_slots
        return violation_slots, initial_statuses, plate_bindings

    def _build_floor_camera_groups(self, camera_configs: List[CameraConfig]) -> Dict[str, List[str]]:
        floor_cameras: Dict[str, List[str]] = {}
        for camera_config in camera_configs:
            floor_cameras.setdefault(camera_config.floor, []).append(camera_config.id)
        return floor_cameras

    def _store_passthrough_frame(self, frame, cam_id: str, grid_frames: Dict[str, np.ndarray]):
        label_frame = frame.copy()
        cv2.putText(
            label_frame,
            cam_id,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        grid_frames[cam_id] = label_frame

    def _new_slot_trace(
        self, cam_id, capture_ts, submitted_at, inferred_at, dequeued_at
    ) -> "_SlotTrace":
        """Build the per-frame stage trace. See :class:`_SlotTrace`."""
        return _SlotTrace(
            cam_id=cam_id,
            capture_ts=capture_ts,
            submitted_at=submitted_at,
            inferred_at=inferred_at,
            dequeued_at=dequeued_at,
        )

    def _process_detections_and_events(
        self, cam_id: str, frame, pipeline, detections,
        *, capture_ts=None, submitted_at=None, inferred_at=None, dequeued_at=None,
    ):
        """Post-detection pipeline body, in two independent halves:

            ROI -> assign -> slot occupancy -> EMIT        (occupancy, bounded)
            ROI -> zones -> slot identity                  (identity, may be slow)

        Occupancy is computed and published FIRST, so a slow OCR/ReID pass can no
        longer delay a slot becoming occupied or vacant. Identity then resolves and
        enriches the record through its own persistence path
        (``parking_slots.current_plate`` / ``update_current_slot_plate``), which
        already exists for exactly this "occupied first, identified later" case.

        Ordering note: the identity half must run AFTER ``assign``, which stamps
        synthetic negative ``track_id``s in place (``slot_assigner.py:77``). Identity
        guards therefore test :func:`~src.detection.detector.is_untracked` rather than
        ``== -1``, so they stay correct in this order.

        Extracted from ``run_multi_camera`` so the single-process async engine
        can reuse the exact same body from its consumer thread. Returns the slot
        ``assignment`` (the multi-camera visualization path still needs it).

        MUST run on a single thread — it mutates shared registry / slot-state /
        event-bus state and is not internally synchronized. In the async engine
        it is called only from the one consumer thread (see ``run_single_process``).
        """
        from src import perf_trace

        # End-to-end stage trace. Populated only when the caller supplies the
        # upstream stamps (the single-process consumer does); the serial path
        # passes none and every stage below simply records into a throwaway dict.
        tr = self._new_slot_trace(cam_id, capture_ts, submitted_at, inferred_at,
                                  dequeued_at)

        _t = time.perf_counter()
        n_raw = len(detections) if detections else 0
        with perf_trace.stage("roi"):
            detections = pipeline.filter_detections_to_roi(detections)
        _t = tr.mark("roi", _t)

        # ---- OCCUPANCY: is the slot full or empty? -------------------------- #
        # Runs FIRST and is published before any identity work begins. Occupancy is
        # the product-critical signal — a slot must flip, and the UI must be able to
        # see it flip, without waiting on OCR or ReID. Measured on prod 2026-07-18
        # this whole block costs <= ~37ms while the identity block below peaked at
        # 517ms; when identity ran first (as it used to) every one of those 517ms
        # was added directly to how long a slot looked stale.
        with perf_trace.stage("assign"):
            assignment = pipeline.assigner.assign(detections)
        _t = tr.mark("assign", _t)
        self._log_camera_detections(cam_id, pipeline, n_raw, detections, assignment)
        with perf_trace.stage("slot"):
            all_events, slot_ctx = self._update_slot_occupancy(
                cam_id, frame, pipeline, assignment
            )
        _t = tr.mark("state_machine", _t)
        # Publish occupancy NOW. Previously uninstrumented, which hid a ReID compare
        # loop and a synchronous DB lookup inside _filter_violation_events (restricted
        # slots only) on this path — measure it rather than move the stall somewhere
        # invisible.
        with perf_trace.stage("emit"):
            if all_events:
                final_events = self._filter_violation_events(
                    frame,
                    assignment,
                    cam_id,
                    all_events,
                )
                _t = tr.mark("violation_filter", _t)
                self._persist_final_events(final_events, trace=tr)

        # ---- IDENTITY: who is in the slot? --------------------------------- #
        # Everything below may land a beat late. The plate reaches the UI via
        # parking_slots.current_plate / the SlotStatus row, not via the occupancy
        # event payload, so a late answer enriches the record rather than delaying it.
        with perf_trace.stage("zones"):
            self._process_special_zones(cam_id, frame, detections)
        with perf_trace.stage("identity"):
            self._update_slot_identity(cam_id, frame, pipeline, slot_ctx)
        return assignment

    def _process_special_zones(self, cam_id: str, frame, detections) -> None:
        if cam_id == "CAM-23":
            logger.info(
                "[PARK_ENTRY][DIAG] CAM-23 frame: registry=%s detections=%d track_ids=%s",
                self.vehicle_registry is not None,
                len(detections) if detections else 0,
                [d.track_id for d in detections] if detections else [],
            )
        if not self.vehicle_registry or not detections:
            return

        camera_special_zones = self.special_zones.get(cam_id, {})

        if cam_id == "CAM-23" and "Park_Entry" in camera_special_zones:
            self._process_park_entry_zone(
                cam_id,
                frame,
                detections,
                camera_special_zones["Park_Entry"],
            )

        if "Entrence" in "".join(camera_special_zones.keys()):
            confirmation_zone = next(
                (zone for zone_id, zone in camera_special_zones.items() if "Entrence" in zone_id),
                None,
            )
            if confirmation_zone:
                self._process_confirmation_zone(
                    cam_id,
                    frame,
                    detections,
                    confirmation_zone,
                )

        # Ground-floor cameras run YOLO occupancy only — skip all ReID embedding
        # compute (TrackingManager feature extraction + global tracking). This
        # is the core-saving gate; identity for ground is via ANPR plates, not
        # appearance. Gated by floor so any ground camera is covered, not just a
        # hardcoded id pair.
        floor = self.pipelines[cam_id].floor if cam_id in self.pipelines else ""
        if not is_identity_disabled(cam_id, floor) and detections:
            if cam_id not in self._tracking_managers:
                self._tracking_managers[cam_id] = TrackingManager(cam_id)
            tracking_manager = self._tracking_managers[cam_id]
            tracking_manager.process_detections(frame, detections)
            self._process_global_tracking(cam_id, frame, detections, tracking_manager)
            self._drive_area_state(cam_id, detections)

    def _drive_area_state(self, cam_id: str, detections) -> None:
        """Feed per-frame observations + boundary crossings to the registry's
        area state machine so ``current_area`` tracks where each car is.

        No-op unless zoning is enabled (``boundary_crossing_detector`` is built
        only then).

        Order matters. Boundary geometry is evaluated first so we know which
        tracks are *currently inside* a band; those are NOT settled — a car in
        the band is in the ambiguous transition zone and must stay IN_TRANSIT
        (eligible in both adjacent areas) until it leaves into a confirmed area.
        Otherwise ``settle_track_area`` would re-settle it into the source area
        every frame after the one-shot outside→inside crossing, collapsing
        IN_TRANSIT to a single frame and breaking cross-area handoff.
        """
        if self.boundary_crossing_detector is None or not self.vehicle_registry:
            return

        boundaries = list(self.boundaries.get(cam_id, {}).values())
        inside_band: set = set()
        if boundaries:
            tracks = [
                (d.track_id, d.bbox) for d in detections
                if not is_untracked(d.track_id)
            ]
            crossings = self.boundary_crossing_detector.detect(
                cam_id, tracks, boundaries
            )
            inside_band = self.boundary_crossing_detector.tracks_inside(cam_id)
            for crossing in crossings:
                self.vehicle_registry.apply_boundary_crossing(
                    cam_id, crossing.track_id, crossing.area_from, crossing.area_to
                )

        for detection in detections:
            tid = detection.track_id
            if is_untracked(tid) or tid in inside_band:
                continue
            self.vehicle_registry.settle_track_area(cam_id, tid)

    # Plate-lock: cap on forced dead-zone OCR attempts per slot (§3b), so a
    # genuinely-unreadable plate doesn't OCR every frame forever.
    _LOCK_MAX_OCR_ATTEMPTS = 4

    # How many OCR reads we are willing to spend identifying one car in one slot.
    # PaddleOCR is ~200ms and runs on the frame loop, so an unbounded retry would stall
    # every other camera in this worker. The budget covers the whole manoeuvre: a car
    # takes several seconds to turn into a slot, and we only need ONE frame in which the
    # plate faces the camera.
    _OCR_ID_MAX_ATTEMPTS = 12
    _OCR_ID_MIN_INTERVAL_S = 0.7

    def _try_ocr_identify(self, cam_id, frame, slot, state_machine, detection) -> None:
        """Read the plate off a car that has just PARKED, bind it, lock it, and learn it.

        Armed ONLY by a VACANT -> OCCUPIED transition (see ``_arm_ocr_for_slot``), so a
        car already sitting in a slot is not re-read on every frame forever. Once the
        plate is READ:

          * it is LOCKED to the slot and stays until the car leaves;
          * the car's PARKED POSE is written to its gallery. That reference is the thing
            the system has never had — every existing reference comes from the gate, and
            gate-vs-parked is the cross-view case where appearance scores WRONG (0.583 for
            the right car, 0.634 for a different one). We could not capture it before
            because capturing it needs to know who the car IS, and appearance could not
            say. OCR breaks that circle: the identity is read, so it is safe to learn from.
        """
        if not getattr(self, "_ocr_armed", {}).get(slot.id):
            return  # only fires on a fresh park (armed by vehicle_parked)

        now_ts = time.time()

        # Slots the operator has told us can NEVER show a plate to this camera (verified
        # by eye, see matching.slot_no_plate_view). Do not spend a single OCR read on the
        # wall: appearance is the only witness that will ever exist here, so go straight
        # to it, from the first attempt rather than the ninth.
        if slot.id in self._no_plate_view_slots():
            self._retry_reid_identify(cam_id, frame, slot, state_machine, detection, now_ts)
            return

        attempts = self._ocr_id_attempts.get(slot.id, 0)
        if attempts >= self._OCR_ID_MAX_ATTEMPTS:
            # OCR's budget is spent and the slot is STILL nameless. Appearance is the
            # only witness left, so keep asking it — see _retry_reid_identify.
            self._retry_reid_identify(cam_id, frame, slot, state_machine, detection, now_ts)
            return
        if now_ts - self._ocr_id_last_at.get(slot.id, 0.0) < self._OCR_ID_MIN_INTERVAL_S:
            return

        # GROUP-LEVEL pacing, on top of the per-slot interval. The per-slot 0.7s
        # alone cannot pace the loop: with ~10 armed slots (every post-restart
        # boot) some slot is always eligible, so EVERY frame paid a 2-8 SECOND
        # inline PaddleOCR read — measured 2026-07-16 at ocr=8.8s of a 19s frame
        # on b2-a. One read per slot_ocr_min_gap_s per process keeps the camera
        # loop alive; round-robin over cameras spreads the reads across slots.
        # Not counted as an attempt and the per-slot clock is untouched, so no
        # slot loses budget to the pacing.
        gap = self._ocr_group_gap_s()
        if gap > 0 and now_ts - getattr(self, "_ocr_group_last_at", 0.0) < gap:
            return

        # An async read for this slot is already in flight — the fresh crop will be
        # submitted next time the slot is eligible, so skip WITHOUT spending an attempt.
        worker = self._ocr_worker_if_async()
        if worker is not None and worker.pending_or_inflight(slot.id):
            return

        crop = self._bbox_crop(frame, detection)
        if crop is None or crop.size == 0:
            return

        self._ocr_id_attempts[slot.id] = attempts + 1
        self._ocr_id_last_at[slot.id] = now_ts
        self._ocr_group_last_at = now_ts

        # Context the registry cannot see: which slot this is, whether it is reserved,
        # which area the camera watches, and how far into the attempt budget we are.
        # These land in the decision log and become ranker features / analysis keys.
        reserved_map = getattr(self, "_reserved_for_map", None) or {}
        special = getattr(self, "_special_slots", None) or set()
        pipelines = getattr(self, "pipelines", None) or {}
        area_registry = getattr(self, "area_registry", None)
        decision_ctx = {
            "floor": getattr(pipelines.get(cam_id), "floor", None),
            "area": area_registry.area_for_camera(cam_id) if area_registry else None,
            "is_reserved": slot.id in reserved_map or slot.id in special,
            "reserved_for": reserved_map.get(slot.id),
            "attempt": attempts + 1,
            "max_attempts": self._OCR_ID_MAX_ATTEMPTS,
        }

        # ASYNC: plan the read here (ReID ranking is main-thread), hand the heavy
        # PaddleOCR read to the worker, and fold the result back in a frame or two
        # later via _drain_slot_ocr_results. The loop keeps flipping slots meanwhile.
        if worker is not None:
            plan = self.vehicle_registry.plan_slot_ocr(
                slot.id, crop, cam_id, decision_ctx=decision_ctx
            )
            if plan is None:
                return  # no OCR plugin
            from src.core.engine.async_slot_ocr import SlotOcrJob

            token = getattr(self, "_ocr_generation", {}).get(slot.id)
            worker.submit(
                SlotOcrJob(
                    slot_id=slot.id, cam_id=cam_id, crop=crop, plan=plan,
                    attempts=attempts + 1, token=token,
                )
            )
            return

        # SYNC (matching.slot_ocr_async = false): read inline, exactly as before.
        plate = self.vehicle_registry.try_ocr_identify_slot(
            slot.id, crop, cam_id, decision_ctx=decision_ctx
        )
        if not plate:
            self._maybe_bind_reid_solo(
                cam_id, slot.id, state_machine, crop, attempts + 1, decision_ctx
            )
            return
        self._bind_ocr_identified_plate(
            cam_id, slot.id, state_machine, plate, crop, attempts + 1
        )

    def _ocr_worker_if_async(self):
        """The async OCR worker, or None when matching.slot_ocr_async is off (the
        inline path) or the worker has not been started (single-frame tools/tests)."""
        mc = getattr(getattr(self, "vehicle_registry", None), "matching_config", None)
        if not getattr(mc, "slot_ocr_async", False):
            return None
        return getattr(self, "_ocr_worker", None)

    def _start_slot_ocr_worker(self) -> None:
        """Spin up the background OCR worker when matching.slot_ocr_async is on.

        A no-op (and _ocr_worker stays unset) when the flag is off, there is no
        registry, or no plate-OCR plugin is loaded — the engine then runs the
        historical inline path. Idempotent."""
        if getattr(self, "_ocr_worker", None) is not None:
            return
        registry = getattr(self, "vehicle_registry", None)
        mc = getattr(registry, "matching_config", None)
        if registry is None or not getattr(mc, "slot_ocr_async", False):
            return
        if not hasattr(registry, "read_slot_plate"):
            return
        from src.core.engine.async_slot_ocr import AsyncSlotOcr

        self._ocr_worker = AsyncSlotOcr(read_fn=registry.read_slot_plate)
        self._ocr_worker.start()
        logger.info(
            "[slot-ocr] async worker started — slot-identify OCR reads run off the "
            "camera loop; results bind a frame late (matching.slot_ocr_async)"
        )

    def _stop_slot_ocr_worker(self) -> None:
        worker = getattr(self, "_ocr_worker", None)
        if worker is not None:
            worker.stop()
            self._ocr_worker = None

    def _drain_slot_ocr_results(self) -> None:
        """Fold every finished async OCR read back into its slot — MAIN THREAD.

        Called once per processed frame. Each result is re-validated against the
        slot's CURRENT state before it can bind: a read that finished after the car
        left, after the slot was re-armed for a different car, or after another path
        already named it, is discarded. So the async path can only ever bind the same
        answer the inline path would have — just a frame or two later."""
        worker = getattr(self, "_ocr_worker", None)
        if worker is None:
            return
        for res in worker.drain():
            self._apply_async_ocr_result(res)

    def _apply_async_ocr_result(self, res) -> None:
        job = res.job
        if getattr(job, "kind", "slot") == "track":
            self._apply_async_track_ocr_result(res)
            return
        slot_id, cam_id = job.slot_id, job.cam_id
        # Re-validate the occupancy this read belongs to still stands.
        if not getattr(self, "_ocr_armed", {}).get(slot_id):
            return  # vacated, or already bound+disarmed
        if getattr(self, "_ocr_generation", {}).get(slot_id) != job.token:
            return  # slot was re-armed for a different car since this read began
        if self.vehicle_registry.get_slot_plate(slot_id):
            self._ocr_armed[slot_id] = False
            return  # named by another path while this read was in flight
        pipeline = (getattr(self, "pipelines", None) or {}).get(cam_id)
        if pipeline is None:
            return
        state_machine = pipeline.state_machines.get(slot_id)
        if state_machine is None:
            return
        if state_machine.state != SlotState.OCCUPIED or state_machine.plate_number:
            return  # no longer an unnamed, occupied slot

        plate = self.vehicle_registry.confirm_slot_ocr(
            job.plan, job.crop, res.text, res.conf
        )
        if plate:
            self._bind_ocr_identified_plate(
                cam_id, slot_id, state_machine, plate, job.crop, job.attempts
            )
            return
        # OCR could not name it — fall through to appearance-only, same gates as sync.
        self._maybe_bind_reid_solo(
            cam_id, slot_id, state_machine, job.crop, job.attempts,
            job.plan.decision_ctx or {},
        )

    def _apply_async_track_ocr_result(self, res) -> None:
        """Fold a finished APPROACH read back in — MAIN THREAD.

        The track equivalent of ``_apply_async_ocr_result``. Re-validated against the
        track's CURRENT state before it can bind, so a read that finished after the car
        was named by another path is discarded rather than re-bound. Everything here is
        main-thread work by necessity: ``confirm_track_ocr`` uses the ReID matcher, and
        the transit hop mutates registry sessions.
        """
        job = res.job
        cam_id = job.cam_id
        key, tid, detection = job.ctx
        registry = self.vehicle_registry
        if registry is None:
            return
        if registry.get_plate_for_track(cam_id, tid):
            return  # named by another path while this read was in flight

        plate = registry.confirm_track_ocr(job.plan, job.crop, res.text)
        if plate:
            logger.info(
                "[ocr-id] track (%s, %s) IDENTIFIED as %s while driving — "
                "the plate is known BEFORE it parks",
                cam_id, tid, plate,
            )
            return

        # OCR could not name it — same fall-through as the sync path. ``detection`` is
        # the one the crop came from, so it is a frame or two old by now; it is used
        # only to ask which slot polygon the car is in, which does not change over that
        # span for a car still manoeuvring.
        now_ts = time.time()
        self._try_adopt_transit_identity(cam_id, tid, key, job.crop, now_ts, detection)

    def _bind_ocr_identified_plate(
        self, cam_id, slot_id, state_machine, plate, crop, attempt_num
    ) -> None:
        """READ, not inferred — bind it, LOCK it, stop reading this slot, learn the
        parked pose. Shared by the inline and async paths."""
        conf = 1.0
        state_machine.bind_identity(
            plate, self._build_slot_snapshot_url(slot_id), confidence=conf, lock=True
        )
        self.vehicle_registry.bind_plate_to_slot(slot_id, plate, cam_id, floor=None)
        if self.db_manager:
            self._persist_slot_plate_binding(slot_id, plate, conf, True, cam_id)
        self._ocr_armed[slot_id] = False
        logger.info(
            "[ocr-id] slot=%s BOUND + LOCKED plate=%s on attempt %d/%d (cam=%s) — "
            "held until the car leaves",
            slot_id, plate, attempt_num, self._OCR_ID_MAX_ATTEMPTS, cam_id,
        )
        # ...and teach the gallery this car's parked pose, now that we KNOW who it is.
        self.vehicle_registry.save_parked_reference(plate, crop, cam_id)

    def _maybe_bind_reid_solo(
        self, cam_id, slot_id, state_machine, crop, attempt_num, decision_ctx
    ) -> None:
        """OCR could not name the car. On a slot whose plate is never in frame — CAM-21
        sees B1_CRO in pure profile, 455 attempts and zero reads — waiting for a second
        witness that can never arrive just leaves the slot NULL forever. So once OCR has
        had most of its budget, let appearance decide ALONE, and only when it is genuinely
        certain (try_reid_identify_slot: the bar is the margin over the runner-up, not the
        raw score). Shared by the inline and async paths."""
        mc = self.vehicle_registry.matching_config
        after = int(getattr(mc, "slot_reid_solo_after_attempts", 8))
        if not getattr(mc, "slot_reid_solo_enabled", False) or attempt_num < after:
            return
        plate = self.vehicle_registry.try_reid_identify_slot(
            slot_id, crop, cam_id,
            is_reserved=bool((decision_ctx or {}).get("is_reserved")),
            decision_ctx=decision_ctx,
        )
        if not plate:
            return
        # Appearance-only: bind it PROVISIONALLY. Not locked, so a later OCR read can
        # still overrule it, and deliberately NOT taught to the gallery — a solo bind is
        # inference, not evidence, and a wrong one would poison the references it was
        # inferred from.
        conf = float(getattr(mc, "slot_reid_solo_min_score", 0.70))
        state_machine.bind_identity(
            plate, self._build_slot_snapshot_url(slot_id), confidence=conf, lock=False
        )
        self.vehicle_registry.bind_plate_to_slot(
            slot_id, plate, cam_id, floor=None, source="reid_solo"
        )
        if self.db_manager:
            self._persist_slot_plate_binding(slot_id, plate, conf, False, cam_id)
        logger.info(
            "[reid-solo] slot=%s BOUND plate=%s (cam=%s) on attempt %d/%d — "
            "appearance only, NOT locked and NOT taught to the gallery; OCR may still "
            "overrule it",
            slot_id, plate, cam_id, attempt_num, self._OCR_ID_MAX_ATTEMPTS,
        )

    def _no_plate_view_slots(self) -> set:
        """Slots whose plate is never in frame (matching.slot_no_plate_view). Cached —
        this is read on every frame of every occupied slot."""
        cached = getattr(self, "_no_plate_view_cache", None)
        if cached is None:
            mc = getattr(self.vehicle_registry, "matching_config", None)
            cached = self._no_plate_view_cache = set(
                getattr(mc, "slot_no_plate_view", None) or []
            )
            if cached:
                logger.info(
                    "[reid-solo] %d slot(s) flagged no-plate-view — OCR skipped, "
                    "appearance-only from the first attempt: %s",
                    len(cached), ", ".join(sorted(cached)),
                )
        return cached

    def _ocr_group_gap_s(self) -> float:
        """Process-wide floor between two slot-identify OCR reads
        (matching.slot_ocr_min_gap_s). Cached — read on every armed frame."""
        cached = getattr(self, "_ocr_group_gap_cache", None)
        if cached is None:
            mc = getattr(self.vehicle_registry, "matching_config", None)
            try:
                cached = float(getattr(mc, "slot_ocr_min_gap_s", 5.0))
            except (TypeError, ValueError):
                cached = 5.0
            self._ocr_group_gap_cache = cached
        return cached

    def _retry_reid_identify(
        self, cam_id, frame, slot, state_machine, detection, now_ts: float
    ) -> None:
        """Keep asking APPEARANCE to name a parked car OCR could not read.

        WHY THIS EXISTS. The 12-attempt budget covers the manoeuvre — a few seconds —
        and then `_try_ocr_identify` returned forever. On a slot whose plate is never in
        frame (CAM-13 films B22 in side profile at point-blank range; CAM-21 does the
        same to B1_CRO, 455 attempts and zero reads) that gave appearance exactly five
        shots, ~4 seconds apart, on one pose in one light — and then the slot was NULL
        until the car left. That is not "ReID could not identify it". That is "ReID was
        asked five times in four seconds and never asked again".

        The retry is worth it because THE GALLERY IS NOT STATIC. A car unidentifiable at
        park time gains references as it is seen elsewhere, so the same query can start
        clearing the margin an hour later. Cost is one embedding (~15ms) per slot per
        interval — no OCR, which is the ~200-670ms part.

        NOT A LOWER BAR. `try_reid_identify_slot` is called with the same score/margin
        gates as the in-budget path; this only changes HOW OFTEN it is asked. The bind
        stays PROVISIONAL (never locked, never taught to the gallery) so a later OCR read
        can still overrule it.

        Note the interval is deliberately slow. Re-asking a noisy scorer often enough
        will eventually cross any threshold by luck alone, and on these slots there is no
        OCR witness that could ever catch a wrong answer — so the margin gate is the only
        thing standing between a customer and a stranger's plate. Seconds would be
        reckless; a minute samples genuinely changed conditions instead of the same frame
        over and over.
        """
        mc = self.vehicle_registry.matching_config
        if not getattr(mc, "slot_reid_solo_enabled", False):
            return
        # Named already (by OCR, an earlier solo bind, or a restart restore) — done.
        if self.vehicle_registry.get_slot_plate(slot.id):
            return
        interval = float(getattr(mc, "slot_reid_retry_interval_s", 60.0) or 0.0)
        if interval <= 0.0:
            return  # retry disabled
        if now_ts - self._reid_retry_last_at.get(slot.id, 0.0) < interval:
            return
        self._reid_retry_last_at[slot.id] = now_ts

        crop = self._bbox_crop(frame, detection)
        if crop is None or crop.size == 0:
            return

        reserved_map = getattr(self, "_reserved_for_map", None) or {}
        special = getattr(self, "_special_slots", None) or set()
        is_reserved = slot.id in reserved_map or slot.id in special
        plate = self.vehicle_registry.try_reid_identify_slot(
            slot.id, crop, cam_id,
            is_reserved=bool(is_reserved),
            decision_ctx={
                "is_reserved": is_reserved,
                "reserved_for": reserved_map.get(slot.id),
                "attempt": self._OCR_ID_MAX_ATTEMPTS,  # budget spent; this is the retry
                "max_attempts": self._OCR_ID_MAX_ATTEMPTS,
                "reid_retry": True,
            },
        )
        if not plate:
            return

        conf = float(getattr(mc, "slot_reid_solo_min_score", 0.70))
        state_machine.bind_identity(
            plate, self._build_slot_snapshot_url(slot.id), confidence=conf, lock=False
        )
        self.vehicle_registry.bind_plate_to_slot(
            slot.id, plate, cam_id, floor=None, source="reid_solo"
        )
        if self.db_manager:
            self._persist_slot_plate_binding(slot.id, plate, conf, False, cam_id)
        logger.info(
            "[reid-solo] slot=%s BOUND plate=%s (cam=%s) on a RETRY after OCR's %d "
            "attempts were spent — appearance only, NOT locked and NOT taught to the "
            "gallery; OCR may still overrule it",
            slot.id, plate, cam_id, self._OCR_ID_MAX_ATTEMPTS,
        )

    def _arm_ocr_for_slot(self, slot_id: str) -> None:
        """A car just parked here: give it a fresh OCR budget."""
        if not hasattr(self, "_ocr_id_attempts"):
            self._ocr_id_attempts, self._ocr_id_last_at, self._ocr_armed = {}, {}, {}
        if not hasattr(self, "_reid_retry_last_at"):
            self._reid_retry_last_at = {}
        self._ocr_id_attempts[slot_id] = 0
        self._ocr_id_last_at[slot_id] = 0.0
        self._reid_retry_last_at.pop(slot_id, None)
        self._ocr_armed[slot_id] = True
        # Occupancy generation: bumped every arm so an ASYNC read that finishes
        # after the car left — and a new car has since parked — is recognised as
        # stale and dropped instead of bound to the wrong occupant.
        gen = getattr(self, "_ocr_generation", None)
        if gen is None:
            gen = self._ocr_generation = {}
        gen[slot_id] = gen.get(slot_id, 0) + 1
        worker = getattr(self, "_ocr_worker", None)
        if worker is not None:
            worker.forget(slot_id)  # discard any queued read for the previous car

    def _update_slot_state(self, cam_id: str, frame, pipeline, assignment):
        """Occupancy then identity, in one call — the historical signature.

        Kept for the serial ``run_multi_camera`` path and for callers that do not
        care about the split. The single-process engine calls the two halves
        separately so it can EMIT occupancy before identity runs; see
        ``_process_detections_and_events``.
        """
        all_events, slot_ctx = self._update_slot_occupancy(
            cam_id, frame, pipeline, assignment
        )
        self._update_slot_identity(cam_id, frame, pipeline, slot_ctx)
        return all_events

    def _update_slot_occupancy(self, cam_id: str, frame, pipeline, assignment):
        """PHASE 1 — is the slot occupied? Nothing here may wait on identity.

        This is the product-critical path: a slot must flip OCCUPIED/VACANT (and the
        UI must be able to see it) without waiting on OCR or ReID. Measured on prod
        2026-07-18 this whole phase costs at most ~10ms, while the identity work that
        used to run BEFORE it peaked at 517ms — the delay was pure statement order.

        Returns ``(all_events, slot_ctx)`` where ``slot_ctx`` carries the per-slot
        state phase 2 needs, so identity does not have to recompute the assignment.

        Deliberately KEPT here rather than deferred to phase 2:
          * ``_arm_ocr_for_slot`` — arming is what makes the later read possible; it
            is a dict write.
          * the whole ``slot_vacant`` teardown — it drops the plate, the lock and any
            queued read. If that lagged behind the vacancy, ``/api/slots`` would
            serve a plate for an empty slot (it reads ``registry.get_slot_plate`` for
            live slots). "Vacant" must mean "and nobody is in it" atomically.
        """
        all_events = []
        slot_ctx = []
        # Per-slot forced-OCR attempt counter (bounded dead-zone pass, §3b).
        if not hasattr(self, "_forced_ocr_attempts"):
            self._forced_ocr_attempts = {}
        # Ground-floor cameras host real slots but must not participate in plate
        # identity matching — they only run occupancy state machines. Short-
        # circuit here (by floor) so we don't even ask the registry; the
        # registry guards remain as defense-in-depth.
        plate_matching_enabled = not is_identity_disabled(
            cam_id, pipeline.floor
        )

        for slot in pipeline.slots:
            state_machine = pipeline.state_machines[slot.id]
            vehicle_in_slot = slot.id in assignment.slot_vehicle_map
            track_id = None
            detection = None
            if vehicle_in_slot:
                track_id, detection = assignment.slot_vehicle_map[slot.id]
                if detection is not None:
                    state_machine.latest_detection_bbox = tuple(
                        float(v) for v in detection.bbox
                    )
            elif state_machine.state == SlotState.VACANT:
                state_machine.latest_detection_bbox = None

            events = state_machine.update(
                vehicle_present=vehicle_in_slot,
                track_id=track_id,
            )
            parked_events = []
            self._log_slot_hold(cam_id, slot, state_machine, assignment, vehicle_in_slot)

            for event in events:
                event.camera_id = cam_id
                event.floor = pipeline.floor
                event.slot_name = slot.label
                event.zone_id = slot.zone_id
                event.zone_name = slot.zone_name

                if event.event_type == "vehicle_parked":
                    # VACANT -> OCCUPIED: this is the moment, and the ONLY moment, that
                    # arms the OCR read for this slot.
                    self._arm_ocr_for_slot(slot.id)
                    snapshot_filename = self._save_slot_snapshot(
                        frame,
                        slot,
                        detection=detection,
                        bbox=state_machine.latest_detection_bbox,
                    )
                    if snapshot_filename:
                        self._persist_slot_snapshot_path(slot.id, snapshot_filename)

                if (
                    event.event_type == "vehicle_parked"
                    and self.vehicle_registry
                    and plate_matching_enabled
                ):
                    # PHASE 2 work, deferred — see _identify_parked_slot. Recorded
                    # rather than run so the occupancy event can be emitted first.
                    parked_events.append(event)
                elif event.event_type == "slot_vacant" and self.vehicle_registry:
                    # unlink_slot also drops any plate-lock on the slot.
                    plate = self.vehicle_registry.unlink_slot(slot.id)
                    self._forced_ocr_attempts.pop(slot.id, None)
                    # Car left: release the OCR lock on this slot. The next car that
                    # parks here re-arms it via vehicle_parked.
                    if hasattr(self, "_ocr_id_attempts"):
                        self._ocr_id_attempts.pop(slot.id, None)
                        self._ocr_id_last_at.pop(slot.id, None)
                        self._ocr_armed.pop(slot.id, None)
                    worker = getattr(self, "_ocr_worker", None)
                    if worker is not None:
                        worker.forget(slot.id)  # drop any queued read for this slot
                    # Car left — the restored plate is no longer valid for this
                    # slot; drop restart-stickiness so a new car is resolved fresh.
                    if getattr(self, "_restored_plate_slots", None):
                        self._restored_plate_slots.discard(slot.id)
                    if plate:
                        event.plate_number = plate
                    if self.db_manager:
                        self._persist_slot_plate_binding(
                            slot.id, None, 0.0, False, cam_id
                        )

            slot_ctx.append(
                (slot, state_machine, vehicle_in_slot, track_id, detection,
                 parked_events, plate_matching_enabled)
            )
            all_events.extend(events)

        return all_events, slot_ctx

    # How often, per slot, to log what is holding it OCCUPIED. A slot stuck for
    # minutes then produces a readable trail rather than one line per frame.
    _SLOTHOLD_INTERVAL_S = 10.0

    # How often, per slot-hosting camera, to log the detection ledger. 30s keeps
    # ~21 slot cameras to ~40 lines/min while still bounding how long a silent
    # failure can hide.
    _CAMDETS_INTERVAL_S = 30.0

    def _log_camera_detections(
        self, cam_id, pipeline, n_raw, detections, assignment
    ) -> None:
        """Per-camera detection LEDGER — where did this frame's detections go?

        Exists because of CAM-04 on 2026-07-18: a car parked in slot B3 for ~2
        minutes and the 12-minute log contained ZERO lines for B2/B3 — no
        [SLOTLAT], no [SLOTHOLD], no track activity. The camera was healthy
        (~2.4 fps, 68/69 windows), so the failure was upstream of slot
        assignment, but NOTHING distinguished the three candidate causes:

          raw=0                        -> YOLO sees nothing (stream frozen at the
                                          encoder, or the detector misses the car)
          raw>0, roi=0                 -> the ROI polygon is eating detections
                                          (filter_detections_to_roi drops silently)
          roi>0, assigned=0, near=...  -> detections exist but the probe lands
                                          outside every slot polygon (geometry /
                                          scaling / probe mismatch — `near` shows
                                          the closest slot and the pixel gap)

        One line per slot-hosting camera per _CAMDETS_INTERVAL_S, INFO-gated.
        Cameras without slots never log (an empty aisle camera has nothing to
        account for).
        """
        if not logger.isEnabledFor(logging.INFO) or not pipeline.slots:
            return
        seen = getattr(self, "_camdets_last_at", None)
        if seen is None:
            seen = self._camdets_last_at = {}
        now_ts = time.time()
        if now_ts - seen.get(cam_id, 0.0) < self._CAMDETS_INTERVAL_S:
            return
        seen[cam_id] = now_ts

        n_roi = len(detections) if detections else 0
        n_assigned = len(assignment.slot_vehicle_map)
        unassigned = assignment.unassigned or []

        near = ""
        if unassigned:
            # Show where the first unassigned car's probe fell and which slot
            # polygon it was closest to — the number that separates "geometry is
            # off by 40px" from "the car is nowhere near a slot".
            d = unassigned[0]
            px, py = d.bottom_center
            best_id, best_dist = None, float("inf")
            for slot in pipeline.slots:
                dist = ((px - slot.centroid_x) ** 2 + (py - slot.centroid_y) ** 2) ** 0.5
                if dist < best_dist:
                    best_id, best_dist = slot.id, dist
            near = (
                f" | first_unassigned: track={d.track_id} "
                f"bbox={tuple(round(float(v), 1) for v in d.bbox)} "
                f"probe=({px:.0f},{py:.0f}) nearest_slot={best_id} "
                f"dist={best_dist:.0f}px"
            )
        logger.info(
            "[CAMDETS] cam=%s raw=%d roi=%d assigned=%d unassigned=%d%s",
            cam_id, n_raw, n_roi, n_assigned, len(unassigned), near,
        )

    def _log_slot_hold(
        self, cam_id, slot, state_machine, assignment, vehicle_in_slot
    ) -> None:
        """Record WHICH detection is holding a slot occupied — diagnostic only.

        Exists because "slot X is occupied" was previously unfalsifiable. On
        2026-07-18 slot B3 stayed OCCUPIED for 7min19s after the car had left and
        was already parked in B5; the log proved only that *something* was seen in
        B3 on ~99.6% of frames, never what. Distinguishing a real parked car from a
        neighbour's box bleeding over the polygon edge required guessing.

        Emits ``[SLOTHOLD]`` with the track id, the box, the probe point tested
        against the polygon, whether it won by point-in-polygon or by the overlap
        fallback, the fraction of the vehicle inside this slot, and the rival slots
        the same box also overlaps. A small ``overlap`` or a strong ``rivals``
        entry means the box is straddling a boundary, not parked in the slot.
        """
        if not logger.isEnabledFor(logging.INFO) or not vehicle_in_slot:
            return
        if state_machine.state not in (SlotState.OCCUPIED, SlotState.LEAVING):
            return
        seen = getattr(self, "_slothold_last_at", None)
        if seen is None:
            seen = self._slothold_last_at = {}
        now_ts = time.time()
        if now_ts - seen.get(slot.id, 0.0) < self._SLOTHOLD_INTERVAL_S:
            return
        seen[slot.id] = now_ts

        ev = (getattr(assignment, "evidence", None) or {}).get(slot.id)
        if not ev:
            return
        rivals = " ".join(f"{sid}:{ov:.2f}" for sid, ov in ev["rivals"]) or "-"
        logger.info(
            "[SLOTHOLD] slot=%s cam=%s state=%s held_by track=%s bbox=%s probe=%s "
            "via=%s overlap=%.2f rivals=%s",
            slot.id, cam_id, state_machine.state.value, ev["track_id"],
            ev["bbox"], ev["probe"], ev["method"], ev["overlap"], rivals,
        )

    def _update_slot_identity(self, cam_id: str, frame, pipeline, slot_ctx) -> None:
        """PHASE 2 — WHO is in the slot. Runs after occupancy has been emitted.

        Everything here is allowed to land a beat late: the plate reaches the UI
        through ``parking_slots.current_plate`` / the ``SlotStatus`` row rather than
        the occupancy event payload (``update_current_slot_plate`` exists precisely
        for "the slot became occupied first and identity resolved slightly later").
        Nothing in phase 1 reads what this writes, so deferring it cannot delay or
        change an occupancy decision.
        """
        for (
            slot, state_machine, vehicle_in_slot, track_id, detection,
            parked_events, plate_matching_enabled,
        ) in slot_ctx:
            for event in parked_events:
                self._identify_parked_slot(
                    cam_id, frame, pipeline, slot, state_machine, track_id,
                    detection, event,
                )

            # OCR IDENTIFICATION — the only mechanism that can fill current_plate
            # correctly. Runs while the car is STILL MANOEUVRING (ENTERING) as well as
            # once parked, because many slots never see a plate in the final pose: CAM-21
            # frames B1_CRO in pure side profile, so a parked car there reads ''. But a
            # car that ends up side-on TURNED INTO the slot, and during that turn its
            # plate swings past the camera. Those are the only frames where the plate
            # exists, and OCR-ing only the settled pose throws them away.
            # OCR IDENTIFICATION — armed by the VACANT -> OCCUPIED transition only, so a
            # car already parked is not re-read forever. Reads the plate off the car,
            # locks it to the slot until the car leaves, and learns the parked pose.
            if (
                self.vehicle_registry
                and vehicle_in_slot
                and plate_matching_enabled
                and detection is not None
                and state_machine.state == SlotState.OCCUPIED
                and not state_machine.plate_number
            ):
                self._try_ocr_identify(cam_id, frame, slot, state_machine, detection)

            if (
                self.vehicle_registry
                and vehicle_in_slot
                and plate_matching_enabled
                and pipeline.state_machines[slot.id].state
                in (SlotState.OCCUPIED, SlotState.LEAVING)
            ):
                self._resolve_locked_plate(
                    cam_id, frame, pipeline, slot, state_machine, track_id, detection
                )
            elif (
                self.vehicle_registry
                and vehicle_in_slot
                and not plate_matching_enabled
                and pipeline.state_machines[slot.id].plate_number
            ):
                # Safety net for plate-disabled cameras (e.g. CAM-01/CAM-02):
                # we no longer attempt try_link_to_slot here, so any stale
                # plate left on the state machine from a previous deploy
                # would never be cleared. Clear it directly and evict the
                # registry's `_parked` entry so `/api/slots` stops returning
                # the old plate.
                state_machine.bind_identity(
                    None,
                    self._build_slot_snapshot_url(slot.id),
                )
                self.vehicle_registry.unlink_slot(slot.id)
                if self.db_manager:
                    self._persist_late_slot_plate(slot.id, None, cam_id)

    def _identify_parked_slot(
        self, cam_id, frame, pipeline, slot, state_machine, track_id, detection, event
    ) -> None:
        """The identity half of a ``vehicle_parked`` transition.

        Lifted verbatim out of the occupancy loop so the event can be emitted before
        this runs. Note ``try_link_to_slot`` is inert in production —
        ``matching.slot_plate_requires_ocr`` is true (``config.yaml:380``) and the
        function returns None on its third statement
        (``vehicle_registry_identity.py:3608``), so today this reliably takes the
        ``else`` branch. It is kept intact rather than pruned because flipping that
        flag back is the documented way to restore appearance-based linking.
        """
        # Attempt to get plate first to save crop with correct filename
        plate = self.vehicle_registry.get_plate_for_track(cam_id, track_id)
        snapshot_path = None
        if plate and detection is not None:
            snapshot_path = self._save_car_crop(frame, detection, plate, cam_id)

        linked_plate = self.vehicle_registry.try_link_to_slot(
            slot_id=slot.id,
            slot_name=slot.label,
            zone_id=slot.zone_id,
            zone_name=slot.zone_name,
            camera_id=cam_id,
            floor=pipeline.floor,
            track_id=track_id,
            timestamp=datetime.now(),
            snapshot_path=snapshot_path,
            # Real vacant->occupied transition: the only place the
            # single-slot/single-pending-plate auto-lock may fire.
            allow_auto_lock=True,
        )
        if linked_plate:
            event.plate_number = linked_plate
            location = self.vehicle_registry.get_plate_location(linked_plate)
            if location:
                event.snapshot_url = location.get("snapshot_url", "")
            # Bind the initial plate as PROVISIONAL (unlocked) with its
            # confidence; the per-frame resolver upgrades it and locks
            # once the confidence/OCR bar is met.
            conf = self.vehicle_registry.get_slot_binding_confidence(slot.id)
            state_machine.bind_identity(
                linked_plate,
                self._build_slot_snapshot_url(slot.id),
                confidence=conf,
            )
            if self.db_manager:
                self._persist_slot_plate_binding(
                    slot.id, linked_plate, conf, False, cam_id
                )
        else:
            state_machine.bind_identity(
                None,
                self._build_slot_snapshot_url(slot.id),
            )

    def _resolve_locked_plate(
        self, cam_id, frame, pipeline, slot, state_machine, track_id, detection
    ) -> None:
        """Per-frame provisional→lock resolution for an OCCUPIED/LEAVING slot.

        - If the slot is already LOCKED: freeze — do nothing (the plate cannot
          change until the slot goes VACANT).
        - Else resolve a voting-gated candidate and UPGRADE the bound plate only
          to a strictly-higher-confidence reading (never downgrade to a weaker
          different plate).
        - LOCK when the plate is voting-committed AND (ReID >= lock_confidence
          OR the bounded forced-OCR pass has OCR-confirmed the plate, §3b).
        """
        registry = self.vehicle_registry
        # Freeze: a locked slot is never re-resolved. This is the engine-side
        # half of the freeze; the registry refuses to relocate/clear it too.
        if state_machine.is_plate_locked():
            return

        lock_conf = registry.matching_config.lock_confidence

        previous_plate = state_machine.plate_number
        plate = registry.try_link_to_slot(
            slot_id=slot.id,
            slot_name=slot.label,
            zone_id=slot.zone_id,
            zone_name=slot.zone_name,
            camera_id=cam_id,
            floor=pipeline.floor,
            track_id=track_id,
            timestamp=datetime.now(),
            # Per-frame resolver for an ALREADY-occupied slot: never auto-lock
            # here, or a slot occupied since startup would grab the next
            # arrival's pending plate at conf 1.0 (the B25=LLJ-9005 bug).
            allow_auto_lock=False,
        )

        if not plate:
            # A BOUND PLATE IS CLEARED ONLY WHEN THE SLOT CHANGES STATE — never here.
            #
            # This resolver runs on every frame of an occupied slot, and it used to wipe
            # any binding the registry could not re-derive on that particular frame. That
            # was catastrophic once try_link_to_slot went dead behind
            # slot_plate_requires_ocr: it returns None on EVERY frame now, so every
            # provisional binding was erased the instant after it was made. B19 was bound
            # to ERS-7949 and destroyed three times in six seconds. The same mechanism
            # quietly undid appearance-only binds and restart-restored plates alike.
            #
            # The car has not moved — the slot is still OCCUPIED, which is the only thing
            # this function actually knows. Absence of a re-derivation is not evidence the
            # car left. The slot going VACANT is, and that path (slot_vacant -> unlink_slot)
            # already clears the plate and releases the lock. That is the single place a
            # binding dies.
            return

        # A live track re-derived a plate for this slot — it is now backed by
        # the running registry, so drop the restart-stickiness protection and
        # let normal clear/upgrade behaviour resume.
        if getattr(self, "_restored_plate_slots", None):
            self._restored_plate_slots.discard(slot.id)

        new_conf = registry.get_slot_binding_confidence(slot.id)

        # Upgrade-only: accept the first binding, a same-plate refresh (keep the
        # higher score), or a strictly-more-confident different plate. Never
        # downgrade to a weaker different plate — that is the drift we prevent.
        if plate == previous_plate:
            conf = max(new_conf, state_machine.plate_confidence)
            state_machine.bind_identity(
                plate, self._build_slot_snapshot_url(slot.id), confidence=conf
            )
        elif not previous_plate or new_conf > state_machine.plate_confidence:
            conf = new_conf
            state_machine.bind_identity(
                plate, self._build_slot_snapshot_url(slot.id), confidence=conf
            )
            if self.db_manager:
                self._persist_slot_plate_binding(
                    slot.id, plate, conf, False, cam_id
                )
        else:
            # Weaker different plate — keep the current provisional binding.
            return

        # §3b forced OCR: any parked provisional slot below the lock bar would
        # never lock on ReID alone. Force a bounded OCR pass to break the tie so
        # the slot can lock via the OCR arm. This covers the "dead zone" above
        # the marginal band AND cars confirmed at low ReID via the OCR ensemble
        # (which would otherwise be stuck provisional forever). The pass only
        # confirms when OCR *agrees with the already-bound plate*, so it's a
        # strong signal independent of the ReID score; capped per slot to bound
        # cost. Cars already at/above the bar lock via ReID without any OCR.
        ocr_ok = registry.get_slot_ocr_confirmed(slot.id)
        if (
            not ocr_ok
            and detection is not None
            and new_conf < lock_conf
        ):
            attempts = self._forced_ocr_attempts.get(slot.id, 0)
            if attempts < self._LOCK_MAX_OCR_ATTEMPTS:
                self._forced_ocr_attempts[slot.id] = attempts + 1
                crop = self._bbox_crop(frame, detection)
                if crop is not None:
                    ocr_ok = registry.try_ocr_confirm_slot(slot.id, crop)
                    if ocr_ok:
                        # The OCR pass may have CORRECTED the binding (rebind
                        # on confident mismatch) — re-read the slot's plate so
                        # the lock below freezes the corrected identity, not
                        # the stale local `plate`.
                        corrected = registry.get_slot_plate(slot.id)
                        if corrected and corrected != plate:
                            plate = corrected
                            new_conf = registry.get_slot_binding_confidence(slot.id)

        # Lock gate. `plate` being non-None already means the voter committed
        # (try_link_to_slot is voting-gated), so this is "voting AND (ReID bar
        # OR OCR)".
        if new_conf >= lock_conf or ocr_ok:
            state_machine.bind_identity(
                plate,
                self._build_slot_snapshot_url(slot.id),
                confidence=new_conf,
                lock=True,
            )
            registry.lock_slot(slot.id)
            self._forced_ocr_attempts.pop(slot.id, None)
            # Evidence: persist the highest-confidence car crop as the slot
            # snapshot at the moment of lock.
            if detection is not None:
                crop_name = self._save_car_crop(frame, detection, plate, cam_id)
                if crop_name:
                    self._persist_slot_snapshot_path(slot.id, crop_name)
            if self.db_manager:
                self._persist_slot_plate_binding(
                    slot.id, plate, new_conf, True, cam_id
                )

    def _bbox_crop(self, frame, detection):
        """Return a padded BGR crop of ``detection.bbox`` (or None) — used to
        feed the forced-OCR pass without writing to disk."""
        try:
            x1, y1, x2, y2 = [int(v) for v in detection.bbox]
            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            crop = frame[y1:y2, x1:x2]
            return crop if crop.size > 0 else None
        except Exception:
            return None

    def _persist_slot_plate_binding(
        self, slot_id: str, plate, confidence: float, locked: bool, camera_id: str
    ) -> None:
        """Persist the plate-lock binding onto the ``parking_slots`` row so it
        survives a restart (read back by _load_camera_db_state)."""
        if not self.db_manager:
            return
        session = self.db_manager.SessionLocal()
        try:
            from src.repositories import ParkingSlotRepository

            db_slot = ParkingSlotRepository.get_by_id(session, slot_id)
            if db_slot is None:
                return
            db_slot.current_plate = plate or None
            db_slot.plate_confidence = float(confidence or 0.0)
            db_slot.plate_locked = bool(locked)
            db_slot.plate_locked_at = datetime.now() if locked else None
            session.commit()
        except Exception as exc:
            session.rollback()
            print(f"[ERROR] Failed to persist slot plate binding for {slot_id}: {exc}")
        finally:
            session.close()

    def _persist_late_slot_plate(self, slot_id: str, plate: str, camera_id: str) -> None:
        session = self.db_manager.SessionLocal()
        try:
            update_current_slot_plate(
                session,
                slot_id=slot_id,
                plate=plate,
                camera_id=camera_id,
            )
        except Exception as exc:
            session.rollback()
            print(f"[ERROR] Failed to persist late slot plate for {slot_id}: {exc}")
        finally:
            session.close()

    def _filter_violation_events(self, frame, assignment, cam_id: str, events):
        final_events = []
        for event in events:
            slot_state_machine = self.pipelines[cam_id].state_machines.get(event.slot_id)
            is_named_reserved_slot = (
                event.slot_id in self._reserved_for_map
                or event.slot_id in self._special_slots
            )
            if (
                not slot_state_machine
                or (not slot_state_machine.is_violation_zone and not is_named_reserved_slot)
            ):
                final_events.append(event)
                continue

            if event.event_type != "vehicle_parked":
                final_events.append(event)
                continue

            _, detection = assignment.slot_vehicle_map.get(event.slot_id, (None, None))
            # Don't early-return when detection is missing — that bypasses
            # _save_alert_snapshot and the alert ends up either with the rolling
            # `slot_<id>_latest.jpg` (stale) or no snapshot at all. Carry an
            # empty crop instead; _save_alert_snapshot's fallback_frame=frame
            # path will save the full camera frame as evidence (Bug fix:
            # production audit 2026-05-05 showed 49/151 intrusion alerts
            # using the rolling fallback path purely because of this branch).
            if detection:
                crop = self._crop_vehicle_bbox_snapshot(frame, detection=detection)
                if crop is None:
                    crop = np.empty((0, 0, 3), dtype=np.uint8)
            else:
                crop = np.empty((0, 0, 3), dtype=np.uint8)

            now_ts = time.time()
            self._recent_violators = [
                violator
                for violator in self._recent_violators
                if now_ts - violator["timestamp"] < self._violation_history_limit
            ]

            is_duplicate = False
            if crop.size > 0 and self.vehicle_registry:
                for violator in self._recent_violators:
                    score = self.vehicle_registry.matcher.compare(crop, violator["crop"])
                    if score > self._violation_match_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                alert_type = self._get_slot_alert_type(event.slot_id, getattr(event, "plate_number", ""))
                if alert_type is None:
                    final_events.append(event)
                    continue
                if (
                    not self.config.alerts.enable_restricted_zone_alerts
                    and alert_type in ("special_needs_violation", "vehicle_intrusion")
                ):
                    final_events.append(event)
                    continue
                event.event_type = alert_type
                event.is_alert = True
                event.severity = (
                    "critical"
                    if alert_type == "vehicle_violation"
                    else "warning"
                )
                # Pass the full frame as fallback so the alert always carries an
                # evidence image even when the per-vehicle crop is empty
                # (Version 0 / Issue #v0-1 fix). Operators previously got
                # intrusion alerts with no snapshot when the bbox was missing.
                event.snapshot_path = self._save_alert_snapshot(
                    crop,
                    alert_type=alert_type,
                    slot_id=event.slot_id,
                    camera_id=cam_id,
                    fallback_frame=frame,
                )
                event.snapshot_url = event.snapshot_path
                if crop.size > 0:
                    self._recent_violators.append(
                        {"crop": crop.copy(), "timestamp": now_ts, "camera_id": cam_id}
                    )
                final_events.append(event)
                print(f"[ALERT] {alert_type.replace('_', ' ').title()}! {cam_id} | Slot:{event.slot_id}")
            else:
                final_events.append(event)

        return final_events

    def _get_slot_alert_type(self, slot_id: str, plate_number: str) -> str | None:
        if slot_id in self._special_slots:
            return "special_needs_violation"
        named_slot_title = self._reserved_for_map.get(slot_id)
        if named_slot_title:
            if self._is_named_slot_vehicle_allowed(plate_number, named_slot_title):
                return None
            return "vehicle_intrusion"
        return (
            "vehicle_violation"
            if "violation" in slot_id.lower()
            else "vehicle_intrusion"
        )

    def _is_named_slot_vehicle_allowed(self, plate_number: str, expected_title: str) -> bool:
        if not self.db_manager or not plate_number or not expected_title:
            return False

        session = self.db_manager.SessionLocal()
        try:
            from src.repositories import VehicleRepository

            vehicle = VehicleRepository.get_by_plate(session, plate_number)
            return bool(vehicle and vehicle.title == expected_title)
        except Exception as exc:
            print(
                f"[ERROR] Failed to validate named-slot ownership for plate "
                f"{plate_number}: {exc}"
            )
            return False
        finally:
            session.close()

    def _persist_final_events(self, events, trace=None) -> None:
        if not events:
            return

        if not self.db_manager:
            self.event_bus.emit_batch(events)
            if trace is not None:
                trace.emit(events)
            return

        _t_db = time.perf_counter()
        session = self.db_manager.SessionLocal()
        try:
            for event in events:
                if event.event_type in (
                    "vehicle_parked",
                    "slot_vacant",
                    "vehicle_violation",
                    "vehicle_intrusion",
                    "special_needs_violation",
                ):
                    is_parked = event.event_type in (
                        "vehicle_parked",
                        "vehicle_violation",
                        "vehicle_intrusion",
                        "special_needs_violation",
                    )
                    plate = getattr(event, "plate_number", None)
                    # Capture the alert_id from log_vehicle_event
                    _, db_alert_id = log_vehicle_event(
                        session,
                        event.slot_id,
                        plate,
                        is_parked,
                        camera_id=event.camera_id,
                        severity=event.severity,
                        snapshot_path=getattr(event, "snapshot_path", None),
                    )
                    # Enrich the event with the database-generated ID
                    if db_alert_id:
                        event.alert_id = db_alert_id

            # Emit AFTER updating with database IDs
            if trace is not None:
                # The commit inside log_vehicle_event is what makes the flip
                # visible to the frontend, so stop the clock here — before the
                # event-bus fan-out, which the DB reader never waits on.
                trace.db_commit_ms = (time.perf_counter() - _t_db) * 1000.0
            self.event_bus.emit_batch(events)
            if trace is not None:
                trace.emit(events)

        except Exception as exc:
            session.rollback()
            print(f"[ERROR] Failed to update slot DB status: {exc}")
        finally:
            session.close()

    def _show_multi_camera_output(
        self,
        cam_id: str,
        frame,
        pipeline,
        assignment,
        detections,
        grid_frames,
        floor_cameras,
        show_camera,
        floor_cols: int,
        grid_cell_width: int,
        grid_cell_height: int,
    ) -> bool:
        self._draw_frame(frame, pipeline, assignment, cam_id, detections)
        grid_frames[cam_id] = frame

        if show_camera:
            if show_camera == cam_id:
                cv2.imshow(f"PMS - {cam_id}", frame)
        else:
            for floor_name, floor_cam_ids in floor_cameras.items():
                grid = self._build_grid(
                    grid_frames,
                    floor_cam_ids,
                    floor_cols,
                    grid_cell_width,
                    grid_cell_height,
                )
                cv2.imshow(f"Damanat PMS - {floor_name}", grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] 'q' pressed - exiting.")
            return True
        return False
