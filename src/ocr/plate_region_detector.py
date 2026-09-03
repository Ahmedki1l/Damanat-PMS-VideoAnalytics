"""plate_region_detector.py — localise the plate inside a vehicle crop.

WHY THIS EXISTS. The slot identify path hands PaddleOCR a whole-vehicle crop and
lets Paddle's own text detector find the plate in it. That works when the crop is
one car, and fails badly when it is not:

    CAM-24 / "B12 CCO", measured 2026-07-20 on a live snapshot
      vehicle crop 1276x574 spanning SIX cars
      read -> '337ZVALD117176536466'
              ^^^^^ B7_CHRO's plate      ^^^^ B12's own plate

That read confirmed B12 only incidentally, and it missed binding the NEIGHBOUR's
plate into B12 by a single character of letter corroboration. Feeding OCR a tight
plate box removes the whole class of error: a timestamp, a camera-name OSD, a
painted floor marking and the car two bays over are all simply not inside a
130x35 plate box.

It is also much cheaper, because Paddle's text-detection stage no longer searches
a megapixel of garage. Measured over 36 live crops from 21 cameras:

    vehicle crop -> OCR    median 1447ms   p90 3961ms   max 13178ms
    plate crop   -> OCR    median  439ms   p90  770ms   max  1497ms

(full method + per-slot results in reports/plate_detector_evaluation.md)

RAW OPENVINO, NOT ULTRALYTICS. The model is a YOLO export, but we drive the IR
directly so ``INFERENCE_NUM_THREADS`` is ours to set. Ultralytics compiles with
its own config, and on this deployment an uncontrolled second pool is exactly the
failure that pinned the fleet at 0.1 fps (see the CAM-00 note in config.yaml).
Two threads is enough: 12.4ms median at 320x320.

THREAD SAFETY. ``detect``/``crop_plate`` are called from the async OCR worker
(``read_slot_plate``), so the single infer request is guarded by a lock — the same
pattern as the ReID backend (reid_openvino_backend.py:134).

NEVER FATAL. Every failure path — no model dir, no IR, bad crop, inference error —
returns None so the caller falls back to the historical full-crop read. Enabling
this can degrade to the old behaviour but must never lose a read outright.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.matching.plugins import PlateRegionDetector

logger = logging.getLogger(__name__)

Box = Tuple[float, float, float, float, float]  # x1, y1, x2, y2, score


def _find_ir_xml(model_dir: str) -> Optional[str]:
    """The single OpenVINO ``.xml`` in an exported model dir (or the xml itself)."""
    if not model_dir:
        return None
    if os.path.isfile(model_dir) and model_dir.endswith(".xml"):
        return model_dir
    if not os.path.isdir(model_dir):
        return None
    xmls = sorted(f for f in os.listdir(model_dir) if f.endswith(".xml"))
    return os.path.join(model_dir, xmls[0]) if xmls else None


class OpenVINOPlateRegionDetector(PlateRegionDetector):
    """Tiny single-class license-plate detector over an OpenVINO IR.

    Model output is YOLO-style ``[1, 5, N]`` — ``(cx, cy, w, h, score)`` per anchor
    in letterboxed input space. Single class, so there is no class dimension and
    NMS is a plain IoU sweep.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        imgsz: int = 320,
        confidence: float = 0.30,
        iou: float = 0.45,
        num_threads: int = 2,
        device: str = "CPU",
    ) -> None:
        self.model_dir = model_dir
        self.imgsz = int(imgsz)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self._lock = threading.Lock()

        import openvino as ov

        xml = _find_ir_xml(model_dir)
        if xml is None:
            raise FileNotFoundError(
                f"PlateRegionDetector: no OpenVINO .xml under {model_dir!r}"
            )
        core = ov.Core()
        model = core.read_model(xml)
        config = {"PERFORMANCE_HINT": "LATENCY"}
        if num_threads and num_threads > 0:
            config["INFERENCE_NUM_THREADS"] = str(int(num_threads))
        self._compiled = core.compile_model(model, device, config)
        self._output = self._compiled.output(0)
        logger.info(
            "[plate-lpd] ready: ir=%s imgsz=%d conf=%.2f threads=%s",
            os.path.basename(xml), self.imgsz, self.confidence, num_threads,
        )

    # -- preprocessing ----------------------------------------------------- #

    def _letterbox(self, bgr: np.ndarray):
        """Aspect-preserving resize + grey pad to ``imgsz`` square.

        Must match the export's training transform — a plain resize would distort
        the box geometry and shift every coordinate we map back.
        """
        import cv2

        h, w = bgr.shape[:2]
        s = min(self.imgsz / h, self.imgsz / w)
        nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, s, left, top

    # -- detection --------------------------------------------------------- #

    def detect(self, bgr: np.ndarray) -> List[Box]:
        """Every plate box above ``confidence``, in ``bgr`` pixel coords, best first.

        Returns ``[]`` on any failure — callers treat that as "no plate localised"
        and fall back, so this must not raise.
        """
        if bgr is None or not isinstance(bgr, np.ndarray) or bgr.size == 0:
            return []
        try:
            blob, scale, pad_x, pad_y = self._letterbox(bgr)
            with self._lock:
                raw = self._compiled([blob])[self._output]
            pred = np.asarray(raw)[0]           # [5, N]
            scores = pred[4]
            keep = scores >= self.confidence
            if not np.any(keep):
                return []
            cx, cy, bw, bh = pred[0][keep], pred[1][keep], pred[2][keep], pred[3][keep]
            sc = scores[keep]
            # letterboxed space -> original crop space
            x1 = (cx - bw / 2 - pad_x) / scale
            y1 = (cy - bh / 2 - pad_y) / scale
            x2 = (cx + bw / 2 - pad_x) / scale
            y2 = (cy + bh / 2 - pad_y) / scale
            h, w = bgr.shape[:2]
            x1 = np.clip(x1, 0, w); x2 = np.clip(x2, 0, w)
            y1 = np.clip(y1, 0, h); y2 = np.clip(y2, 0, h)
            boxes = [
                (float(a), float(b), float(c), float(d), float(s))
                for a, b, c, d, s in zip(x1, y1, x2, y2, sc)
                if c > a and d > b
            ]
            return self._nms(boxes)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[plate-lpd] detect failed: %r", exc)
            return []

    def _nms(self, boxes: List[Box]) -> List[Box]:
        boxes = sorted(boxes, key=lambda b: -b[4])
        kept: List[Box] = []
        for b in boxes:
            if all(self._iou(b, k) <= self.iou for k in kept):
                kept.append(b)
        return kept

    @staticmethod
    def _iou(a: Box, b: Box) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    @staticmethod
    def _first_box_outside(
        boxes: List[Box],
        exclude_regions: Optional[Sequence[Tuple[float, float, float, float]]],
        *,
        width: int,
        height: int,
    ) -> Optional[Box]:
        """Best-scoring box whose centre is outside every excluded region.

        Centre-based rather than overlap-based on purpose. A real plate can sit
        partly under a composited panel, and rejecting on any overlap would
        throw away a legible plate to avoid a false one. The centre is what
        says which of the two things the detector actually locked onto.
        """
        if not exclude_regions:
            return boxes[0]
        for x1, y1, x2, y2, score in boxes:
            cx = (x1 + x2) / 2.0 / max(1, width)
            cy = (y1 + y2) / 2.0 / max(1, height)
            if not any(
                rx1 <= cx <= rx2 and ry1 <= cy <= ry2
                for rx1, ry1, rx2, ry2 in exclude_regions
            ):
                return (x1, y1, x2, y2, score)
        return None

    # -- the thing callers actually want ----------------------------------- #

    def crop_plate(
        self,
        bgr: np.ndarray,
        *,
        pad_ratio: float = 0.15,
        exclude_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
        boxes: Optional[List[Box]] = None,
    ) -> Optional[np.ndarray]:
        """The highest-scoring plate region, padded, or ``None`` if none found.

        The pad matters: PaddleOCR's recogniser wants a little quiet space around
        the glyphs, and a box clipped exactly to the plate edge loses the outer
        characters on a slightly-off localisation.

        ``exclude_regions`` is the OVERLAY GUARD, and it exists because of one
        specific trap: Hikvision composites its own plate/OSD panel into the
        corner of a frame, and that panel is a sharp, high-contrast, perfectly
        rectangular rendering of a plate. The detector loves it. It will
        routinely outscore the real plate on a car twenty metres away, and then
        our "independent" OCR is reading Hikvision's own answer back to itself —
        which is not independent verification, it is a very expensive echo.

        Regions are NORMALISED ``(x1, y1, x2, y2)`` in 0..1 so a caller does not
        have to know the frame size. A candidate whose CENTRE falls inside an
        excluded region is skipped and the next-ranked box is tried, rather than
        giving up: a real plate that happens to sit near the panel must still be
        found. Only if every candidate is excluded does this return ``None``.
        """
        # `boxes` lets a caller that has ALREADY detected on this exact frame
        # hand the result back instead of paying for a second forward pass. The
        # selection below is deterministic, so re-detecting cannot produce a
        # different answer -- it only costs. Left None, behaviour is unchanged.
        if boxes is None:
            boxes = self.detect(bgr)
        if not boxes:
            return None
        h, w = bgr.shape[:2]
        box = self._first_box_outside(boxes, exclude_regions, width=w, height=h)
        if box is None:
            return None
        x1, y1, x2, y2, _ = box
        px, py = (x2 - x1) * pad_ratio, (y2 - y1) * pad_ratio
        xa, ya = max(0, int(x1 - px)), max(0, int(y1 - py))
        xb, yb = min(w, int(x2 + px)), min(h, int(y2 + py))
        if xb <= xa or yb <= ya:
            return None
        crop = bgr[ya:yb, xa:xb]
        return crop if crop.size else None


# Backwards-compatible alias
PlateRegionDetector = OpenVINOPlateRegionDetector
