"""Evidence processors, including lazy adapters over existing VA models."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from .domain import (
    EvidenceUnavailable,
    FrameEvidence,
    PlateEvidence,
    PlateReadState,
)
from .settings import EntrySettings


def _is_hik_sourced(metadata) -> bool:
    """Was this frame pulled from HikCentral by us, rather than pushed by a camera?

    HikCentral is a PULL source: these bytes exist because our service asked for
    them. The marker matters because such a frame carries HikCentral's own
    composited plate panel, which our OCR must not read back as if it were an
    independent look at the car.
    """
    try:
        return str((metadata or {}).get("evidence_source", "")).lower() == "hikcentral"
    except Exception:  # pragma: no cover - defensive
        return False


def _safe_dominant_colour(frame):
    """Mean HSV of the crop centre, or None. Never raises.

    Colour is an optional, subtractive signal: a frame that yields no colour
    simply cannot veto anything, and every consumer fails open. Losing an entry
    because a colour probe threw would be absurd, so it cannot.
    """
    try:
        from src.reid_matcher import dominant_color_hsv

        return dominant_color_hsv(frame)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("[EntryV2] colour probe failed: %r", exc)
        return None


logger = logging.getLogger(__name__)
PLATE_CROP_SUBDIRECTORY = "entry_plate_crops"
VEHICLE_CROP_SUBDIRECTORY = "entry_vehicle_crops"
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_JPEG_START_OF_FRAME_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def _encoded_image_dimensions(encoded: bytes) -> Optional[Tuple[int, int]]:
    """Read JPEG/PNG dimensions without allocating decoded pixel storage."""
    if encoded.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(encoded) < 24 or encoded[12:16] != b"IHDR":
            return None
        width = int.from_bytes(encoded[16:20], "big")
        height = int.from_bytes(encoded[20:24], "big")
        return (width, height) if width > 0 and height > 0 else None

    if not encoded.startswith(b"\xff\xd8"):
        # PMS normalizes every accepted Hikvision vehicle crop to JPEG. Reject
        # other formats before OpenCV can allocate based on an unbounded header.
        return None

    offset = 2
    dimensions: Optional[Tuple[int, int]] = None
    while offset < len(encoded):
        if encoded[offset] != 0xFF:
            return None
        while offset < len(encoded) and encoded[offset] == 0xFF:
            offset += 1
        if offset >= len(encoded):
            return None
        marker = encoded[offset]
        offset += 1
        if marker in {0xD9, 0xDA}:  # end-of-image or start-of-scan
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(encoded):
            return None
        segment_length = int.from_bytes(encoded[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(encoded):
            return None
        if marker in _JPEG_START_OF_FRAME_MARKERS:
            if segment_length < 7:
                return None
            height = int.from_bytes(encoded[offset + 3 : offset + 5], "big")
            width = int.from_bytes(encoded[offset + 5 : offset + 7], "big")
            current = (width, height)
            if width <= 0 or height <= 0:
                return None
            if dimensions is not None and dimensions != current:
                return None
            dimensions = current
        offset += segment_length
    return dimensions


class EvidenceProcessor(Protocol):
    def analyze(
        self,
        *,
        event_id: str,
        camera_id: str,
        source_role: str,
        images: Sequence[bytes],
        metadata: Mapping[str, Any],
    ) -> Tuple[FrameEvidence, ...]: ...


class ExistingModelsEvidenceProcessor:
    """Lazy bridge to OSNet OpenVINO, local LPD, and PaddleOCR-mobile.

    Every multipart image is contractually a vehicle crop (burst crops are
    allowed; Hikvision plate-only/OSD parts are not). Plate OCR is run only on a plate ROI
    produced by the local detector; it never OCRs a raw/full camera frame.  No
    image is retained on this object or in the returned evidence. When an image
    directory is supplied, each LPD plate crop is also written to the dedicated
    ``entry_plate_crops`` diagnostics folder after OCR scores it.
    """

    def __init__(
        self,
        registry,
        settings: EntrySettings,
        *,
        image_dir: Optional[str] = None,
    ):
        self._registry = registry
        self._settings = settings
        self._plate_crop_dir = (
            Path(image_dir) / PLATE_CROP_SUBDIRECTORY if image_dir else None
        )
        self._vehicle_crop_dir = (
            Path(image_dir) / VEHICLE_CROP_SUBDIRECTORY if image_dir else None
        )
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._reid = None
        self._plate_detector = None
        self._ocr = None
        self._vehicle_detector = None
        self._vehicle_detector_unavailable = False

    def _overlay_regions_for(self, source_role: str, metadata):
        """Where Hikvision may have composited a plate panel into this frame.

        Applies to WHOLE-FRAME sources only: the ANPR overview (PMS forwards the
        bounded full frame precisely so we localise the plate ourselves, panel
        and all) and anything pulled from HikCentral. Ramp line-crossing crops
        are already tight around the vehicle, so there is no panel in them and
        no reason to risk excluding part of a real plate.

        Returns () when nothing is configured, which makes the guard inert.
        """
        if not self._settings.overlay_exclude_regions:
            if _is_hik_sourced(metadata):
                logger.warning(
                    "[EntryV2] HikCentral-sourced frame analysed with NO overlay "
                    "guard configured - our OCR may be reading Hikvision's own "
                    "composited plate panel back to itself. Set "
                    "ENTRY_V2_OVERLAY_EXCLUDE_REGIONS from the image probe."
                )
            return ()
        if _is_hik_sourced(metadata) or (source_role or "").lower() == "anpr":
            return self._settings.overlay_exclude_regions
        return ()

    def _is_whole_frame_source(self, source_role: str, metadata) -> bool:
        """Did this evidence arrive uncropped?

        Exactly the rule `_overlay_regions_for` already applies, and for the
        same reason: the ANPR overview and anything pulled from HikCentral are
        whole frames, while ramp line-crossing evidence is already a camera-side
        crop tight around the vehicle. Anything that re-frames evidence has to
        respect that split, or it goes hunting for a car inside a picture that
        is already nothing but car.
        """
        return bool(
            _is_hik_sourced(metadata) or (source_role or "").lower() == "anpr"
        )

    def _vehicle_crop(self, frame, plate_box):
        """The subject vehicle inside a whole frame.

        Returns ``(crop, share, rule, candidates)``. A ``crop`` of None means
        keep the full frame, and that fallback is always safe: the full frame is
        the behaviour this replaces, so a detector miss can never score worse
        than never having run at all.

        WHICH box is the subject matters more than finding one. The gate looks
        out at a public street, so background and parked cars are in shot. Two
        rules decide, in order:

          1. the box containing the plate we localised in THIS frame. The ANPR
             event exists because a plate was read here, which makes the rule
             self-calibrating -- no per-camera geometry to configure or drift;
          2. failing that, the largest box. Measured on the HBR-4920 gate frame:
             subject 24.3% of frame, the only other vehicle 0.1%, a 243x margin.

        A winner under ``vehicle_min_area_ratio`` counts as no winner. A distant
        car is not the one at the barrier, and embedding it would be a confident
        wrong answer where the full frame is only a weak one.
        """
        height, width = frame.shape[:2]
        area = float(height * width)
        if area <= 0:
            return None, 0.0, "empty_frame", 0

        detector = self._vehicle_models()
        if detector is None:
            return None, 0.0, "detector_unavailable", 0
        try:
            detections = detector.detect(frame)
        except Exception:
            logger.warning(
                "[EntryV2][vehicle-crop] detector raised; keeping the full frame",
                exc_info=True,
            )
            return None, 0.0, "detector_error", 0

        boxes = []
        for detection in detections or ():
            bbox = getattr(detection, "bbox", None)
            if bbox is None or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox[:4])
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        if not boxes:
            return None, 0.0, "no_vehicle_detected", 0

        chosen = None
        rule = "largest_box"
        if plate_box is not None:
            plate_x = (float(plate_box[0]) + float(plate_box[2])) / 2.0
            plate_y = (float(plate_box[1]) + float(plate_box[3])) / 2.0
            holding = [
                box for box in boxes
                if box[0] <= plate_x <= box[2] and box[1] <= plate_y <= box[3]
            ]
            if holding:
                # Tightest box that still holds the plate: on a nested
                # detection the car is the inner box, not the lorry behind it.
                chosen = min(
                    holding, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
                )
                rule = "plate_containment"
        if chosen is None:
            chosen = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

        share = ((chosen[2] - chosen[0]) * (chosen[3] - chosen[1])) / area
        # Two bars, because the two rules carry different weight. Plate
        # containment is DIRECT evidence that this box is the car the barrier
        # just read, so it only has to clear a sanity floor. `largest_box` is a
        # guess made when the LPD found nothing, and the failure it invites is
        # specific: a car queued well back becomes the gallery reference for a
        # car at the barrier. A guess therefore has to be a big object.
        floor = self._settings.vehicle_min_area_ratio
        if rule == "largest_box":
            floor = max(floor, self._settings.vehicle_min_area_ratio_unverified)
        if share < floor:
            return None, share, "below_min_area_ratio", len(boxes)

        pad = self._settings.vehicle_crop_pad
        box_width = chosen[2] - chosen[0]
        box_height = chosen[3] - chosen[1]
        x1 = max(0, int(round(chosen[0] - box_width * pad)))
        y1 = max(0, int(round(chosen[1] - box_height * pad)))
        x2 = min(width, int(round(chosen[2] + box_width * pad)))
        y2 = min(height, int(round(chosen[3] + box_height * pad)))
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None, share, "empty_crop", len(boxes)
        return crop, share, rule, len(boxes)

    def analyze(
        self,
        *,
        event_id: str,
        camera_id: str,
        source_role: str,
        images: Sequence[bytes],
        metadata: Mapping[str, Any],
    ) -> Tuple[FrameEvidence, ...]:
        if not images:
            raise EvidenceUnavailable("at_least_one_image_is_required")
        reid, detector, ocr = self._models()
        exclude_regions = self._overlay_regions_for(source_role, metadata)
        whole_frame = (
            self._settings.vehicle_crop_enabled
            and self._is_whole_frame_source(source_role, metadata)
        )

        import cv2
        import numpy as np

        # The existing OSNet/LPD/Paddle adapters were built for one engine
        # thread and do not expose a per-request infer handle. Serialize this
        # low-rate entry path so concurrent camera HTTP workers cannot race
        # mutable inference state or oversubscribe the CPU.
        with self._inference_lock:
            evidence = []
            for index, encoded in enumerate(images):
                frame_id = f"{event_id}:{index}"
                dimensions = _encoded_image_dimensions(encoded)
                if dimensions is None:
                    continue
                width, height = dimensions
                if (
                    width > self._settings.max_decoded_image_dimension
                    or height > self._settings.max_decoded_image_dimension
                    or width * height > self._settings.max_decoded_image_pixels
                ):
                    raise EvidenceUnavailable("decoded_image_dimensions_exceeded")
                frame = cv2.imdecode(
                    np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None or frame.size == 0:
                    continue
                # Localise the plate BEFORE Re-ID. Nothing about OCR changes --
                # it still reads the frame it was measured on -- but the plate's
                # position is what tells the vehicle crop which car at this gate
                # is the one the barrier just read.
                plate_box = None
                plate_boxes = None
                if whole_frame:
                    try:
                        plate_boxes = detector.detect(frame) or ()
                    except Exception:
                        plate_boxes = None
                        logger.warning(
                            "[EntryV2][vehicle-crop] plate detector raised while "
                            "locating the subject; falling back to box area",
                            exc_info=True,
                        )
                    plate_box = _first_plate_box_outside(
                        plate_boxes, exclude_regions, width=width, height=height
                    )

                # The image Re-ID actually embeds. For a camera-side ramp crop
                # that is the frame itself; for a whole frame it is the subject
                # vehicle, because embedding the whole gate scene made the
                # gallery reference a picture of the gate rather than of a car.
                reid_frame = frame
                if whole_frame:
                    vehicle, share, rule, candidate_count = self._vehicle_crop(
                        frame, plate_box
                    )
                    if vehicle is not None:
                        reid_frame = vehicle
                    self._log_vehicle_crop(
                        event_id=event_id,
                        camera_id=camera_id,
                        source_role=source_role,
                        frame_index=index,
                        crop=vehicle,
                        share=share,
                        rule=rule,
                        candidates=candidate_count,
                    )

                vector = reid.extract_feature(reid_frame)
                embedding: Tuple[float, ...] = ()
                if vector is not None:
                    array = np.asarray(vector, dtype=np.float32).reshape(-1)
                    norm = float(np.linalg.norm(array))
                    if norm > 0.0:
                        embedding = tuple(float(value) for value in array / norm)

                # Reuses the boxes found above when this is a whole frame, so
                # the gate path runs the LPD once rather than twice. Selection
                # is deterministic, so this is the same crop the second pass
                # would have produced -- and VA has no spare CPU to prove it
                # twice. Ramp crops never detected above, so they pass None and
                # crop_plate detects as it always has.
                plate_crop = detector.crop_plate(
                    frame, exclude_regions=exclude_regions, boxes=plate_boxes
                )
                if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
                    plate = PlateEvidence(
                        evidence_id=frame_id,
                        camera_id=camera_id,
                        source_role=source_role,
                        state=PlateReadState.NO_PLATE,
                    )
                else:
                    text, confidence = ocr.read(
                        plate_crop,
                        allow_retry=False,
                        apply_plate_roi=False,
                        apply_hud_mask=False,
                    )
                    plate = PlateEvidence(
                        evidence_id=frame_id,
                        camera_id=camera_id,
                        source_role=source_role,
                        state=(
                            PlateReadState.READABLE
                            if text
                            else PlateReadState.UNREADABLE
                        ),
                        text=str(text or ""),
                        confidence=float(confidence or 0.0),
                    )
                    self._save_plate_crop(
                        plate_crop,
                        event_id=event_id,
                        camera_id=camera_id,
                        source_role=source_role,
                        frame_index=index,
                        plate=plate,
                    )
                self._log_plate_read(
                    event_id=event_id,
                    camera_id=camera_id,
                    source_role=source_role,
                    frame_index=index,
                    plate=plate,
                    crop=plate_crop,
                )
                evidence.append(
                    FrameEvidence(
                        evidence_id=frame_id,
                        embedding=embedding,
                        plate=plate,
                        # Mean HSV of the crop centre — a few array ops, no
                        # second model. VA is CPU-starved and a learned colour
                        # classifier on the gate path would compete with the
                        # detector for frames; this check is already tuned and
                        # is only ever used to REMOVE an impossible candidate.
                        # Measured on the same image Re-ID embeds. On a whole
                        # frame the "dominant colour" of a gate forecourt is
                        # asphalt and sky, which is not a fact about the car.
                        colour_hsv=_safe_dominant_colour(reid_frame),
                    )
                )

        if not evidence:
            raise EvidenceUnavailable("no_decodable_image")
        if not any(item.embedding for item in evidence):
            raise EvidenceUnavailable("reid_embedding_unavailable")
        return tuple(evidence)

    def _log_plate_read(
        self,
        *,
        event_id: str,
        camera_id: str,
        source_role: str,
        frame_index: int,
        plate: PlateEvidence,
        crop,
    ) -> None:
        """Record every OCR attempt so plate legibility per camera is greppable.

        Diagnostic only — never influences an entry decision. Complements the
        crop files, which only exist when an image directory is configured AND
        the detector found a plate: a `no_plate` frame used to leave no trace at
        all, which reads identically to "OCR was never attempted" in the logs.

        The crop's pixel width is logged beside the confidence because that is
        the number that actually moves when camera exposure/bitrate/shutter are
        retuned; confidence alone cannot distinguish "plate too small" from
        "plate blurred". Threshold is echoed so a rejected read is legible
        without cross-referencing ENTRY_V2_OCR_MIN_CONFIDENCE.
        """
        minimum = float(self._settings.ocr_min_confidence)
        if plate.state is PlateReadState.NO_PLATE:
            result = "no_plate"
        elif plate.state is PlateReadState.UNREADABLE:
            result = "unreadable"
        elif plate.confidence >= minimum:
            result = "reliable"
        else:
            result = "below_min_confidence"

        crop_height, crop_width = 0, 0
        shape = getattr(crop, "shape", None)
        if isinstance(shape, tuple) and len(shape) >= 2:
            try:
                crop_height, crop_width = int(shape[0]), int(shape[1])
            except (TypeError, ValueError):
                crop_height, crop_width = 0, 0

        logger.info(
            "[EntryV2][OCR] event=%s camera=%s role=%s frame=%02d state=%s "
            "plate=%r key=%s confidence=%.4f min_confidence=%.4f "
            "crop=%dx%d result=%s",
            event_id,
            camera_id,
            source_role,
            frame_index,
            plate.state.value,
            plate.text,
            plate.key or "-",
            plate.confidence,
            minimum,
            crop_width,
            crop_height,
            result,
        )

    def _save_plate_crop(
        self,
        crop,
        *,
        event_id: str,
        camera_id: str,
        source_role: str,
        frame_index: int,
        plate: PlateEvidence,
    ) -> None:
        """Persist diagnostic plate evidence without affecting entry decisions."""
        if self._plate_crop_dir is None:
            return

        confidence = (
            f"{plate.confidence:.4f}"
            if math.isfinite(plate.confidence)
            and 0.0 <= plate.confidence <= 1.0
            else "invalid"
        )
        filename = (
            f"event-{self._event_filename_part(event_id)}"
            f"__camera-{self._safe_filename_part(camera_id, max_length=32)}"
            f"__role-{self._safe_filename_part(source_role, max_length=24)}"
            f"__frame-{frame_index:02d}"
            f"__ocr-confidence-{confidence}"
            f"__{plate.state.value}.jpg"
        )
        target = self._plate_crop_dir / filename
        temporary_path: Optional[str] = None
        try:
            import cv2

            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            if not encoded_ok:
                raise OSError("OpenCV could not encode the plate crop")

            self._plate_crop_dir.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{filename}.",
                suffix=".tmp",
                dir=self._plate_crop_dir,
            )
            with os.fdopen(descriptor, "wb") as temporary_file:
                temporary_file.write(encoded.tobytes())
            os.replace(temporary_path, target)
            temporary_path = None
            logger.info(
                "[EntryV2][plate-crop] saved path=%s ocr_confidence=%.4f state=%s",
                target,
                plate.confidence,
                plate.state.value,
            )
        except Exception:
            logger.warning(
                "[EntryV2][plate-crop] save failed path=%s; entry analysis continues",
                target,
                exc_info=True,
            )
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.debug(
                        "[EntryV2][plate-crop] temporary cleanup failed path=%s",
                        temporary_path,
                        exc_info=True,
                    )

    @staticmethod
    def _safe_filename_part(value: str, *, max_length: int) -> str:
        cleaned = _UNSAFE_FILENAME_CHARS.sub("-", str(value or "")).strip("._-")
        return (cleaned or "unknown")[:max_length]

    @classmethod
    def _event_filename_part(cls, event_id: str) -> str:
        raw = str(event_id or "")
        readable = cls._safe_filename_part(raw, max_length=64)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return f"{readable}-{digest}"

    def _models(self):
        with self._lock:
            if self._reid is None:
                self._reid = self._registry.reid_matcher

            if self._plate_detector is None:
                configured = self._registry.match_decision.plate_detector
                if configured.__class__.__name__ == "NoopPlateRegionDetector":
                    from src.ocr.plate_region_detector import (
                        OpenVINOPlateRegionDetector,
                    )

                    configured = OpenVINOPlateRegionDetector(
                        model_dir=self._settings.lpd_model_dir,
                        confidence=self._settings.lpd_confidence,
                        iou=self._settings.lpd_iou,
                        num_threads=self._settings.lpd_threads,
                    )
                self._plate_detector = configured

            if self._ocr is None:
                configured = self._registry.match_decision.plate_ocr
                if configured.__class__.__name__ == "NoopPlateOCR":
                    from src.ocr.plate_ocr import PaddlePlateOCR

                    configured = PaddlePlateOCR(
                        model_dir=self._settings.ocr_model_dir or None,
                        cpu_threads=4,
                    )
                self._ocr = configured
        return self._reid, self._plate_detector, self._ocr

    def _vehicle_models(self):
        """The vehicle detector, or None to leave whole frames uncropped.

        Deliberately NOT the fleet detector from config. That one is
        `cam00_yolo26s_320`, a single-class fine-tune specialised on CAM-00's
        top-down fisheye; the gate looks out at a street in front elevation,
        which is the view a stock COCO model already handles. Loading the wrong
        specialist here would fail in the one place this crop exists to serve.

        A load failure is latched, not retried per frame: entry keeps working
        uncropped, and the warning fires once rather than on every car.
        """
        if self._vehicle_detector is not None or self._vehicle_detector_unavailable:
            return self._vehicle_detector
        with self._lock:
            if self._vehicle_detector is not None or self._vehicle_detector_unavailable:
                return self._vehicle_detector
            try:
                from src.detection.detector import DetectorConfig, VehicleDetector
                from src.config import DetectorPreprocessingConfig

                self._vehicle_detector = VehicleDetector(
                    DetectorConfig(
                        model_path=self._settings.vehicle_model_dir,
                        confidence=self._settings.vehicle_confidence,
                        classes=[2, 5, 7],
                        imgsz=self._settings.vehicle_imgsz,
                    ),
                    # CLAHE OFF, explicitly. DetectorPreprocessingConfig
                    # defaults `enabled` True, and VehicleDetector._preprocess_frame
                    # does NOT forward `detector_scale`, so the default would run
                    # a full BGR<->LAB round trip over all 4.17 MP of the gate
                    # frame before every 640x640 inference -- the exact waste
                    # config.py:547 documents, at its worst size, on a host whose
                    # entry path is already serialized behind _inference_lock.
                    # The Re-ID backend disables CLAHE for the same reason: the
                    # weights were never calibrated with it.
                    DetectorPreprocessingConfig(enabled=False),
                )
                logger.info(
                    "[EntryV2][vehicle-crop] detector ready model=%s conf=%.2f "
                    "pad=%.2f min_area=%.3f",
                    self._settings.vehicle_model_dir,
                    self._settings.vehicle_confidence,
                    self._settings.vehicle_crop_pad,
                    self._settings.vehicle_min_area_ratio,
                )
            except Exception:
                self._vehicle_detector_unavailable = True
                logger.warning(
                    "[EntryV2][vehicle-crop] detector unavailable (model=%s); "
                    "whole frames will be embedded uncropped, which is the "
                    "pre-crop behaviour",
                    self._settings.vehicle_model_dir,
                    exc_info=True,
                )
        return self._vehicle_detector

    def _log_vehicle_crop(
        self,
        *,
        event_id: str,
        camera_id: str,
        source_role: str,
        frame_index: int,
        crop,
        share: float,
        rule: str,
        candidates: int,
    ) -> None:
        """Say what was embedded, and keep the picture that proves it.

        Every abstention on this path used to be a bare number with no way to
        ask whether the model was wrong or the framing was. The crop on disk is
        the answer to that question.
        """
        used = crop is not None
        logger.info(
            "[EntryV2][vehicle-crop] event=%s camera=%s role=%s frame=%02d "
            "candidates=%d rule=%s share=%.3f min_share=%.3f embedded=%s",
            event_id,
            camera_id,
            source_role,
            frame_index,
            candidates,
            rule,
            share,
            self._settings.vehicle_min_area_ratio,
            "vehicle_crop" if used else "full_frame",
        )
        if not used or self._vehicle_crop_dir is None:
            return

        filename = (
            f"event-{self._event_filename_part(event_id)}"
            f"__camera-{self._safe_filename_part(camera_id, max_length=32)}"
            f"__role-{self._safe_filename_part(source_role, max_length=24)}"
            f"__frame-{frame_index:02d}"
            f"__share-{share:.3f}"
            f"__{self._safe_filename_part(rule, max_length=24)}.jpg"
        )
        target = self._vehicle_crop_dir / filename
        temporary_path: Optional[str] = None
        try:
            import cv2

            encoded_ok, encoded = cv2.imencode(
                ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )
            if not encoded_ok:
                raise OSError("OpenCV could not encode the vehicle crop")

            self._vehicle_crop_dir.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=".tmp", dir=self._vehicle_crop_dir
            )
            with os.fdopen(descriptor, "wb") as temporary_file:
                temporary_file.write(encoded.tobytes())
            os.replace(temporary_path, target)
            temporary_path = None
            logger.info("[EntryV2][vehicle-crop] saved path=%s", target)
        except Exception:
            logger.warning(
                "[EntryV2][vehicle-crop] could not save the crop; the decision "
                "is unaffected", exc_info=True,
            )
        finally:
            if temporary_path and os.path.exists(temporary_path):
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass


def _first_plate_box_outside(boxes, exclude_regions, *, width, height):
    """Best-scoring plate box whose centre escapes the composited panel.

    Same centre-based test as `PlateRegionDetector._first_box_outside`, and the
    same reason: a real plate can sit partly under Hikvision's overlay, and
    rejecting on any overlap would throw away a legible plate to avoid a false
    one. Kept here rather than reaching into the detector's private helper so
    the two can be read side by side if either ever changes.
    """
    for box in boxes or ():
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in box[:4])
        if not exclude_regions:
            return (x1, y1, x2, y2)
        centre_x = (x1 + x2) / 2.0 / max(1, width)
        centre_y = (y1 + y2) / 2.0 / max(1, height)
        if not any(
            rx1 <= centre_x <= rx2 and ry1 <= centre_y <= ry2
            for rx1, ry1, rx2, ry2 in exclude_regions
        ):
            return (x1, y1, x2, y2)
    return None


class DisabledEvidenceProcessor:
    def analyze(self, **kwargs):
        del kwargs
        raise EvidenceUnavailable("entry_v2_disabled")
