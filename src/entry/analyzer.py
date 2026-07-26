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


logger = logging.getLogger(__name__)
PLATE_CROP_SUBDIRECTORY = "entry_plate_crops"
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
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._reid = None
        self._plate_detector = None
        self._ocr = None

    def analyze(
        self,
        *,
        event_id: str,
        camera_id: str,
        source_role: str,
        images: Sequence[bytes],
        metadata: Mapping[str, Any],
    ) -> Tuple[FrameEvidence, ...]:
        del metadata  # reserved for future crop-role descriptors; never retained
        if not images:
            raise EvidenceUnavailable("at_least_one_image_is_required")
        reid, detector, ocr = self._models()

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
                vector = reid.extract_feature(frame)
                embedding: Tuple[float, ...] = ()
                if vector is not None:
                    array = np.asarray(vector, dtype=np.float32).reshape(-1)
                    norm = float(np.linalg.norm(array))
                    if norm > 0.0:
                        embedding = tuple(float(value) for value in array / norm)

                plate_crop = detector.crop_plate(frame)
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


class DisabledEvidenceProcessor:
    def analyze(self, **kwargs):
        del kwargs
        raise EvidenceUnavailable("entry_v2_disabled")
