"""
ov_infer_core.py — raw-OpenVINO detection core (AsyncInferQueue) for the
single-process migration.

WHY THIS EXISTS
---------------
The production bottleneck is not the model (isolated forward ≈ 24 ms) but the
*serial* per-process inference loop plus cross-process pool contention. The fix
(see the architecture design) is one process with a single OpenVINO
``AsyncInferQueue`` (THROUGHPUT) feeding every camera, so several inferences are
in flight at once and fill the cores instead of one-at-a-time LATENCY calls.

This module is that inference core. It deliberately reuses Ultralytics' *own*
preprocess (LetterBox) and postprocess (NMS / end2end decode + box scaling) so
detections are numerically identical to ``model.track()`` — only the forward
pass is swapped from Ultralytics' synchronous LATENCY call to our own
AsyncInferQueue. ByteTrack is then updated exactly the way Ultralytics'
``on_predict_postprocess_end`` does, using the caller-supplied per-camera tracker
so track-id state stays isolated per camera.

TWO ENTRY POINTS
----------------
* ``detect_track_sync(frame, tracker)`` — synchronous: preprocess → one OV
  forward → postprocess → tracker.update. Used by the still-serial Phase 1 path
  so preprocess/decode/ByteTrack parity can be validated before the process
  model changes.
* ``submit(frame, userdata, done)`` + the internal completion callback — the
  asynchronous path the single-process scheduler drives in Phase 2: many
  ``start_async`` in flight, each completing on an OV worker thread that runs
  postprocess + track and hands the result back via ``done``.

Both paths share ONE compiled model and ONE predictor (for pre/post), so the
parity guarantee is the same for serial and async.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

from src import perf_trace
from src.detection.detector import Detection


def _find_ir_xml(model_path: str) -> Optional[Path]:
    """The single OpenVINO ``.xml`` for an exported model dir (or the xml itself)."""
    p = Path(model_path)
    if p.is_file() and p.suffix == ".xml":
        return p
    if p.is_dir():
        xmls = sorted(p.glob("*.xml"))
        return xmls[0] if xmls else None
    return None


class OVInferCore:
    """Raw-OpenVINO detector core reusing Ultralytics pre/post for parity.

    Construct with an already-loaded Ultralytics ``YOLO`` model (so the class
    map / end2end flag / stride all come from the real export) plus the runtime
    knobs. The core warms the Ultralytics predictor once (to materialise its
    preprocess/postprocess and the class-filter args), then compiles its own
    THROUGHPUT model + AsyncInferQueue over the same IR.
    """

    def __init__(
        self,
        model,                       # ultralytics.YOLO (already loaded)
        model_path: str,
        imgsz: int,
        confidence: float,
        classes,                     # list[int] | None (class filter)
        device: str = "cpu",
        num_threads: int = 0,        # 0 = OpenVINO default
        nireq: int = 0,              # 0 = OPTIMAL_NUMBER_OF_INFER_REQUESTS
        max_box_area_ratio: float = 0.0,
    ):
        import openvino as ov

        self._ov = ov
        self.model = model
        self.imgsz = int(imgsz)
        self.max_box_area_ratio = float(max_box_area_ratio or 0.0)

        # 1) Warm the predictor so model.predictor (pre/postprocess + args) and
        #    model.predictor.model (AutoBackend: names, end2end, stride) exist and
        #    carry OUR conf/classes/imgsz. A single dummy predict is enough.
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        model.predict(
            dummy,
            conf=confidence,
            classes=classes,
            imgsz=self.imgsz,
            device=device,
            verbose=False,
        )
        self._predictor = model.predictor
        # construct_results zips orig_imgs against self.batch[0]; give it a
        # 1-element path list so a single-image postprocess never IndexErrors.
        self._predictor.batch = [["async"]]

        # 2) Compile our own THROUGHPUT copy of the IR + a persistent
        #    AsyncInferQueue. THROUGHPUT (not LATENCY) is what fills the cores with
        #    concurrent requests — the measured 2x+ over the serial LATENCY path.
        ir_xml = _find_ir_xml(model_path)
        if ir_xml is None:
            raise FileNotFoundError(
                f"OVInferCore: no OpenVINO .xml found for model_path={model_path!r}"
            )
        core = ov.Core()
        ov_model = core.read_model(str(ir_xml))
        # Static [1,3,imgsz,imgsz] export — reshape defensively so a mismatched
        # imgsz can never silently letterbox to the wrong input size.
        try:
            ov_model.reshape([1, 3, self.imgsz, self.imgsz])
        except Exception:
            pass  # already static at this shape
        # VA_OV_NUM_THREADS env wins over the config value, so the single-process
        # deploy can size the one pool to the real core count without a DB/YAML
        # change (rebuild-only constraint on prod).
        try:
            env_threads = int(os.environ.get("VA_OV_NUM_THREADS", "") or 0)
        except ValueError:
            env_threads = 0
        threads = env_threads if env_threads > 0 else int(num_threads or 0)
        config = {"PERFORMANCE_HINT": "THROUGHPUT"}
        if threads > 0:
            config["INFERENCE_NUM_THREADS"] = str(threads)
        self._compiled = core.compile_model(ov_model, "CPU", config)
        self._input_name = self._compiled.input().get_any_name()

        if nireq and nireq > 0:
            self._nireq = int(nireq)
        else:
            try:
                self._nireq = max(1, int(self._compiled.get_property(
                    "OPTIMAL_NUMBER_OF_INFER_REQUESTS")))
            except Exception:
                self._nireq = 2
        self._queue = ov.AsyncInferQueue(self._compiled, self._nireq)
        self._queue.set_callback(self._on_complete)

        # postprocess is not guaranteed thread-safe across concurrent OV
        # callbacks (it mutates predictor.results); serialize just the post step.
        self._post_lock = threading.Lock()

        print(
            f"[OVInferCore] ready: imgsz={self.imgsz} nireq={self._nireq} "
            f"threads={threads or 'default'} classes="
            f"{'ALL' if classes is None else classes} ir={ir_xml.name}"
        )

    # --- shared pre/post ---------------------------------------------------- #

    def _preprocess(self, frame_bgr: np.ndarray):
        """Ultralytics preprocess → (input_np float32 [1,3,H,W], input_tensor).

        Returns both the numpy tensor for OpenVINO and the torch tensor
        ``img`` that postprocess needs for box scaling (it reads img.shape[2:]).
        """
        im = self._predictor.preprocess([frame_bgr])   # torch [1,3,H,W], /255
        return im.cpu().numpy(), im

    def _decode(self, raw_results, im, frame_bgr: np.ndarray):
        """Ultralytics postprocess on a raw OV output → list[Results] (len 1).

        ``raw_results`` is the OV request result dict; we massage it into the
        single preds tensor Ultralytics' postprocess expects, exactly as
        AutoBackend's LATENCY branch does (``list(...).values()`` → from_numpy).
        """
        y = list(raw_results.values())
        preds = torch.from_numpy(y[0]) if len(y) == 1 else [torch.from_numpy(x) for x in y]
        # postprocess mutates predictor state (results); guard for async callers.
        with self._post_lock:
            results = self._predictor.postprocess(preds, im, [frame_bgr])
        return results

    def _tracks_to_detections(self, tracks: np.ndarray, frame_shape) -> List[Detection]:
        """Ultralytics track rows → Detection list, applying the whole-frame box
        guard exactly like TrackedDetector._parse_results.

        ``tracks`` rows are ``[x1, y1, x2, y2, track_id, conf, cls, det_idx]``
        (see on_predict_postprocess_end); boxes are already in ``frame`` coords.
        """
        detections: List[Detection] = []
        if tracks is None or len(tracks) == 0:
            return detections

        frame_area = (
            float(frame_shape[0] * frame_shape[1])
            if self.max_box_area_ratio > 0.0 and frame_shape is not None
            else 0.0
        )
        for row in tracks:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            if frame_area > 0.0:
                if ((x2 - x1) * (y2 - y1)) / frame_area > self.max_box_area_ratio:
                    continue
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=int(row[6]),
                confidence=float(row[5]),
                track_id=int(row[4]),
            ))
        return detections

    def _update_tracker(self, results, frame_bgr: np.ndarray, tracker) -> np.ndarray:
        """Feed one image's detections into a per-camera tracker, the way
        Ultralytics' on_predict_postprocess_end does. Returns the raw track rows."""
        det = results[0].boxes.cpu().numpy()
        return tracker.update(det, frame_bgr)

    # --- Phase 1: synchronous path ----------------------------------------- #

    def detect_track_sync(self, frame_bgr: np.ndarray, tracker) -> List[Detection]:
        """Preprocess → one OV forward → postprocess → tracker.update (serial).

        ``tracker`` is the caller's per-camera BYTETracker/BOTSORT instance, so
        track state stays isolated exactly as in the model.track() path.
        """
        input_np, im = self._preprocess(frame_bgr)
        raw = self._compiled({self._input_name: input_np})
        results = self._decode(raw, im, frame_bgr)
        tracks = self._update_tracker(results, frame_bgr, tracker)
        return self._tracks_to_detections(tracks, frame_bgr.shape)

    # --- Phase 2: asynchronous path ---------------------------------------- #

    def submit(
        self,
        frame_bgr: np.ndarray,
        tracker,
        done: Callable[[List[Detection], object], None],
        userdata: object = None,
    ) -> None:
        """Preprocess on the caller thread, then start an async OV forward.

        The completion callback (an OV worker thread) runs postprocess +
        tracker.update and invokes ``done(detections, userdata)``. ``start_async``
        blocks when all ``nireq`` requests are busy, which is the design's natural
        back-pressure — the scheduler paces itself against the pool.
        """
        input_np, im = self._preprocess(frame_bgr)
        self._queue.start_async(
            {self._input_name: input_np},
            userdata=(im, frame_bgr, tracker, done, userdata, time.perf_counter()),
        )

    def _on_complete(self, request, userdata) -> None:
        im, frame_bgr, tracker, done, ud, t_submit = userdata
        perf_trace.record_async_infer((time.perf_counter() - t_submit) * 1000.0)
        try:
            results = self._decode(request.results, im, frame_bgr)
            tracks = self._update_tracker(results, frame_bgr, tracker)
            detections = self._tracks_to_detections(tracks, frame_bgr.shape)
        except Exception as exc:  # never let one bad frame kill the OV worker
            print(f"[OVInferCore] completion error: {exc!r}")
            detections = []
        done(detections, ud)

    def wait_all(self) -> None:
        """Block until all in-flight async requests have completed."""
        self._queue.wait_all()

    @property
    def nireq(self) -> int:
        return self._nireq
