"""
tracker.py — Lightweight tracker integration using Ultralytics built-in tracking.

Ultralytics YOLO has built-in support for ByteTrack and BoT-SORT.
This wrapper combines detection + tracking into a single call via model.track(),
which keeps stable IDs across frames with persist=True.

Why not a standalone tracker?
  - Ultralytics handles the tracker lifecycle internally.
  - No extra dependency needed (no separate `deep_sort` or `norfair`).
  - ByteTrack is fast and CPU-friendly.
  - Using model.track() is simpler and avoids common integration bugs.
"""

import inspect
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from src.config import DetectorConfig, TrackerConfig, DetectorPreprocessingConfig
from src.detection.detector import Detection
from src.ov_tuning import apply_openvino_overrides
from src.preprocessing import luminance_normalize_auto
from src import perf_trace

# Ultralytics tracker classes by their `tracker_type` config value.
_TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}

# Fallback key for callers that don't pass a camera_id (single-camera mode).
_DEFAULT_CAMERA = "__default__"


class TrackedDetector:
    """
    Combines YOLO detection with built-in tracking.

    Detection weights are shared (one model load), but ByteTrack *state* is
    kept per camera: ``detect_and_track`` swaps the active tracker in/out of
    the Ultralytics predictor keyed by ``camera_id`` before each call. This
    decouples detection (shared → flat RAM at many cameras) from tracking
    (per-camera → stable IDs), so round-robin frames from different cameras
    never corrupt each other's track state the way a single shared tracker did.
    """

    def __init__(
        self,
        detector_config: DetectorConfig,
        tracker_config: TrackerConfig,
        preprocessing_config: DetectorPreprocessingConfig = None,
    ):
        """
        Args:
            detector_config: Model path, confidence, classes, imgsz.
            tracker_config: Tracker type (bytetrack or botsort).
            preprocessing_config: Optional luminance normalization settings.
        """
        self.detector_config = detector_config
        self.tracker_config = tracker_config
        self.preprocessing_config = preprocessing_config or DetectorPreprocessingConfig()

        # One ByteTrack/BoT-SORT instance per camera_id (decoupled tracking).
        self._camera_trackers: Dict[str, object] = {}

        print(f"[INFO] Loading YOLO model from '{detector_config.model_path}'...")
        # Ultralytics hardcodes PERFORMANCE_HINT=LATENCY, which caps the CPU pool
        # at 8 threads however many cores exist. Must patch before YOLO() compiles
        # the model — OpenVINO sizes the pool once and never re-reads it.
        apply_openvino_overrides(
            num_threads=getattr(detector_config, "ov_num_threads", 0),
            performance_hint=getattr(detector_config, "ov_performance_hint", ""),
        )
        self.model = YOLO(detector_config.model_path, task="detect")
        self.imgsz = self._resolve_imgsz()
        self.classes = self._resolve_classes()

        # Resolve device: "auto" picks CUDA if available, else CPU
        if detector_config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = detector_config.device

        pp_status = "ON" if self.preprocessing_config.enabled else "OFF"
        cls_status = "ALL" if self.classes is None else self.classes
        print(f"[INFO] Model loaded. Tracker: {tracker_config.type} | Device: {self.device} "
              f"| imgsz: {self.imgsz} | classes: {cls_status} | Preprocessing: {pp_status}")

    def _resolve_imgsz(self) -> int:
        """The size the model ACTUALLY runs at, not the one config.yaml asks for.

        An OpenVINO export is statically shaped — models/yolo11m_320_int8_* is
        locked to 320x320 — and Ultralytics silently overrides the requested
        imgsz from the export's metadata.yaml. config.yaml's `imgsz: 640` is
        therefore dead config; trusting it would make us letterbox to the wrong
        size. Read the real value from the export, and fall back to config only
        for a plain .pt weight file with no metadata.
        """
        meta = Path(self.detector_config.model_path) / "metadata.yaml"
        if meta.is_file():
            try:
                size = YAML.load(str(meta)).get("imgsz")
                if isinstance(size, (list, tuple)) and size:
                    return int(max(size))
                if isinstance(size, int):
                    return int(size)
            except (OSError, ValueError, TypeError):
                pass
        return int(self.detector_config.imgsz)

    def _resolve_classes(self):
        """The class ids to KEEP, matched to the loaded model — not blindly from config.

        ``detector_config.classes`` holds COCO vehicle ids [2,5,7] (car/bus/truck),
        correct for a stock COCO model. But ``model_path`` is DB-owned and may point
        at the facility fine-tune, which is SINGLE-CLASS (``names={0: 'vehicle'}``):
        it only ever emits class 0, so filtering it by [2,5,7] matches nothing and
        drops EVERY detection — no car is ever seen and no slot can flip (observed on
        CAM-23 with yolo26n_ft, 2026-07-17). Rather than trust the static/DB value,
        keep only the configured ids that actually exist in this model's class map:

          - stock COCO model      -> [2,5,7] survive        -> filter to vehicles
          - single-class fine-tune-> none of [2,5,7] exist  -> return None (keep all)

        Returning None means "no class filter"; safe here because a dedicated vehicle
        model emits only vehicles. Mirrors ``_resolve_imgsz`` reading the export's own
        metadata instead of trusting dead config.
        """
        names = getattr(self.model, "names", None) or {}
        wanted = self.detector_config.classes
        if not wanted:
            return None
        valid = [c for c in wanted if c in names]
        if not valid:
            print(
                f"[INFO] Configured classes {list(wanted)} are absent from this "
                f"model's class map {dict(names)} — keeping ALL detections "
                f"(single-class vehicle model)."
            )
            return None
        return valid

    def _preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Apply luminance-safe preprocessing if enabled.

        Only modifies the L channel in LAB space — hue and saturation
        are preserved so downstream color matching stays stable.

        When ``detector_scale`` is on, the frame is first resized to the model's
        input size, so CLAHE runs on the ~16x smaller image the detector will
        actually see rather than the full 1280x720 frame. Ultralytics would have
        performed that same downscale itself a moment later, so the detector's
        input is unchanged in size — only the order of resize-vs-CLAHE differs.

        Returns:
            (inference_frame, scale_x, scale_y) — the scale factors map a box in
            inference_frame's coordinates back onto the original frame. They are
            1.0 when no resize happened.
        """
        pp = self.preprocessing_config
        if not pp.enabled:
            return frame, 1.0, 1.0

        work = frame
        scale_x = scale_y = 1.0

        if pp.detector_scale:
            h, w = frame.shape[:2]
            ratio = min(1.0, self.imgsz / max(h, w))   # never upscale
            if ratio < 1.0:
                new_w, new_h = max(1, round(w * ratio)), max(1, round(h * ratio))
                work = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                scale_x, scale_y = w / new_w, h / new_h

        normalized = luminance_normalize_auto(
            work,
            clip_limit=pp.clip_limit,
            grid_size=pp.grid_size,
            gamma_correction=pp.gamma_correction,
            dark_threshold=pp.dark_threshold,
        )
        return normalized, scale_x, scale_y

    def _new_tracker(self):
        """Build a fresh tracker instance the way Ultralytics' track() does.

        Reads the same ``<type>.yaml`` (e.g. ``bytetrack.yaml``) the predictor
        would load, so per-camera trackers behave identically to the built-in
        one — only their *state* is isolated.
        """
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(f"{self.tracker_config.type}.yaml")))
        # Increase track buffer to handle detection gaps during slot transitions.
        # Default 30 frames @ 1.3 FPS = ~23s buffer. Doubled to ~46s to tolerate
        # brief occlusions when cars move between adjacent slots on same camera.
        cfg.track_buffer = 60
        tracker_class = _TRACKER_MAP.get(cfg.tracker_type)
        if tracker_class is None:
            raise AssertionError(
                f"Only {set(_TRACKER_MAP)} are supported, but got '{cfg.tracker_type}'"
            )
        # ``frame_rate`` is accepted by most but not all Ultralytics versions —
        # only pass it when the constructor actually takes it, so the same code
        # works across versions. When absent, the tracker uses its own default.
        kwargs = {}
        if "frame_rate" in inspect.signature(tracker_class.__init__).parameters:
            kwargs["frame_rate"] = 30
        return tracker_class(cfg, **kwargs)

    def _tracker_for(self, camera_id: str):
        """Return (lazily creating) this camera's dedicated tracker."""
        tracker = self._camera_trackers.get(camera_id)
        if tracker is None:
            tracker = self._new_tracker()
            self._camera_trackers[camera_id] = tracker
        return tracker

    def detect_and_track(
        self,
        frame: np.ndarray,
        camera_id: str = _DEFAULT_CAMERA,
    ) -> List[Detection]:
        """
        Run detection + tracking on a single frame.

        Detection uses the shared model; tracking uses this camera's own
        ByteTrack state (swapped into the predictor before the call), so IDs
        stay stable per camera even when cameras are interleaved round-robin.

        Args:
            frame: BGR image as numpy array (from cv2).
            camera_id: Identifies which per-camera tracker state to use.

        Returns:
            List of Detection objects with stable track_id values.
        """
        # Preprocess for better detection in hard lighting
        with perf_trace.stage("clahe"):
            inference_frame, scale_x, scale_y = self._preprocess_frame(frame)

        # Build tracker config filename — Ultralytics expects e.g. "bytetrack.yaml"
        tracker_cfg = f"{self.tracker_config.type}.yaml"

        # Swap in this camera's tracker before inference. On the very first
        # call the predictor doesn't exist yet; Ultralytics creates trackers[0]
        # during track(), and we adopt that instance for this camera afterwards.
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            predictor.trackers[0] = self._tracker_for(camera_id)

        # Split the "infer" wall-time into OpenVINO compute vs scheduler-wait /
        # cgroup-throttle / framework overhead. wall = whole model.track() call;
        # cpu = THIS thread's CPU time over it (wall-cpu ⇒ time off-core: pool
        # compute + preemption + CFS throttle); pre/ov/post from results.speed
        # isolate the pure forward pass; throttled-delta attributes the freeze to
        # the cgroup. All gated by PERF_TRACE (zero cost when off).
        with perf_trace.stage("infer"):
            _w0 = time.perf_counter()
            _c0 = perf_trace._thread_cpu_s() if perf_trace.enabled() else 0.0
            _t0 = perf_trace.read_cgroup_throttle() if perf_trace.enabled() else None
            results = self.model.track(
                inference_frame,
                conf=self.detector_config.confidence,
                classes=self.classes,
                imgsz=self.imgsz,
                device=self.device,
                persist=True,               # Maintain tracker state across frames
                tracker=tracker_cfg,         # e.g., "bytetrack.yaml"
                verbose=False,
            )
            if perf_trace.enabled():
                _wall = (time.perf_counter() - _w0) * 1000.0
                _cpu = (perf_trace._thread_cpu_s() - _c0) * 1000.0
                _t1 = perf_trace.read_cgroup_throttle()
                _thr_ms = ((_t1[2] - _t0[2]) / 1000.0) if (_t0 and _t1) else 0.0
                try:
                    _spd = results[0].speed  # {'preprocess','inference','postprocess'} ms
                except (IndexError, AttributeError, TypeError):
                    _spd = {}
                perf_trace.record_infer(
                    wall_ms=_wall, cpu_ms=_cpu,
                    pre_ms=float(_spd.get("preprocess", 0.0) or 0.0),
                    ov_ms=float(_spd.get("inference", 0.0) or 0.0),
                    post_ms=float(_spd.get("postprocess", 0.0) or 0.0),
                    throttled_ms=_thr_ms,
                )

        # First-call bootstrap: adopt the predictor's auto-created tracker for
        # this camera so subsequent calls swap the correct instance back in.
        predictor = getattr(self.model, "predictor", None)
        if (
            camera_id not in self._camera_trackers
            and predictor is not None
            and getattr(predictor, "trackers", None)
        ):
            self._camera_trackers[camera_id] = predictor.trackers[0]

        return self._parse_results(results, scale_x, scale_y, frame.shape)

    def _parse_results(
        self,
        results,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        frame_shape: tuple = None,
    ) -> List[Detection]:
        """Parse YOLO tracking results into Detection objects.

        Ultralytics returns boxes in the coordinates of whatever image it was
        handed. When _preprocess_frame resized the frame, those are the resized
        image's coordinates — so scale them back onto the original frame, which
        is what every downstream consumer (ROI, slots, zones, crops) expects.
        """
        detections = []

        if not results or len(results) == 0:
            return detections

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes

        # Whole-frame box guard (see DetectorConfig.max_box_area_ratio). Measured
        # against the ORIGINAL frame, so it is unaffected by _preprocess_frame's
        # resize — the coords below are already scaled back.
        max_area_ratio = getattr(self.detector_config, "max_box_area_ratio", 0.0) or 0.0
        frame_area = (
            float(frame_shape[0] * frame_shape[1])
            if max_area_ratio > 0.0 and frame_shape is not None
            else 0.0
        )

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1 = float(xyxy[0]) * scale_x, float(xyxy[1]) * scale_y
            x2, y2 = float(xyxy[2]) * scale_x, float(xyxy[3]) * scale_y

            if frame_area > 0.0:
                if ((x2 - x1) * (y2 - y1)) / frame_area > max_area_ratio:
                    continue

            class_id = int(boxes.cls[i].cpu().numpy())
            confidence = float(boxes.conf[i].cpu().numpy())

            # Track ID — may be None if tracker hasn't assigned one yet
            tid = -1
            if boxes.id is not None:
                tid = int(boxes.id[i].cpu().numpy())

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                confidence=confidence,
                track_id=tid,
            ))

        return detections
