"""
src.ocr.plate_ocr — PaddleOCR-mobile wrapper implementing the ``PlateOCR`` ABC.

Phase 1 (Wave 1, WS-D). The cascade gates this plugin to the marginal ReID
band ``[ocr_marginal_low, ocr_marginal_high]`` (Phase 2 wiring); this module
is unaware of that gating. It only answers ``read(crop) -> (text, confidence)``.

Backend
-------
PaddleOCR-mobile (``PP-OCRv4_mobile_det`` / ``PP-OCRv5_mobile_det`` + English
recognition by default). The English mobile recogniser is the smallest one
that covers Latin alphanumerics; Saudi/UAE plates are predominantly
Latin-script + digits with an Arabic-script row beneath that is normally
masked or unreadable from floor-camera angles, so we deliberately stick to
English text-recognition to keep the model footprint and latency low. If a
deployment needs Arabic-script readings the ``lang`` constructor argument
can be set to ``"arabic"`` to load the Arabic recogniser instead.

Lazy initialisation
-------------------
Heavy model loading is deferred to the first ``read()`` call so unit tests
that don't exercise the backend stay fast, and the noop default in
``MatchDecision`` doesn't pay any cost.

Compatibility
-------------
PaddleOCR's Python API changed between 2.x and 3.x:

  * 2.x: ``PaddleOCR(use_angle_cls=True, lang='en')`` returns
    ``[[ [box, (text, conf)], ... ]]`` from ``ocr(img, cls=True)``.
  * 3.x: ``PaddleOCR(...)`` takes per-stage flags
    (``use_textline_orientation`` etc.) and ``predict(img)`` returns
    objects with ``rec_texts`` / ``rec_scores`` attributes (or dict keys).

This module handles both shapes. If neither PaddleOCR nor PaddlePaddle is
importable, ``PaddlePlateOCR()`` construction raises ``ImportError`` with
installation instructions.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import List, Optional, Tuple

import numpy as np

from src import perf_trace
from src.matching.plugins import PlateOCR

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Text helpers (no heavy deps — exported for tests and the benchmark tool)
# --------------------------------------------------------------------------- #


_PLATE_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def normalise_plate_text(raw: str) -> str:
    """
    Uppercase, strip non-alphanumerics, collapse whitespace.

    Examples
    --------
    >>> normalise_plate_text(' ab-c 123 ')
    'ABC123'
    >>> normalise_plate_text('')
    ''
    """
    if not raw:
        return ""
    upper = raw.upper()
    # Strip non-alphanumeric — handles hyphens, spaces, punctuation, and
    # incidental Arabic/Latin diacritics.
    return _PLATE_NON_ALNUM.sub("", upper)


def levenshtein_distance(a: str, b: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    A small iterative DP — no external dependency. Cost-1 for insert,
    delete, and substitute.

    Examples
    --------
    >>> levenshtein_distance('ABC123', 'ABC123')
    0
    >>> levenshtein_distance('ABC123', 'ABC124')
    1
    >>> levenshtein_distance('ABC123', 'BC123')
    1
    >>> levenshtein_distance('', 'ABC')
    3
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Rolling two-row DP — O(len(a) * len(b)) time, O(min(len)) memory.
    if len(a) < len(b):
        a, b = b, a  # ensure |a| >= |b| so the inner row is the shorter one

    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a, start=1):
        curr[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,          # deletion
                curr[j - 1] + 1,      # insertion
                prev[j - 1] + cost,   # substitution
            )
        prev, curr = curr, prev
    return prev[len(b)]


def plates_match(
    ocr_text: str,
    expected_plate: str,
    max_edit_distance: int = 1,
) -> bool:
    """
    Return True iff the normalised OCR reading is within ``max_edit_distance``
    Levenshtein edits of the normalised expected plate.

    Used by the Phase 2 ensemble rule: when the OCR reading agrees with the
    pending ANPR plate the cascade can hard-confirm even at marginal ReID
    scores; when it disagrees it can hard-reject. Empty readings always
    return False (the caller should fall through to the ensemble rule).
    """
    a = normalise_plate_text(ocr_text)
    b = normalise_plate_text(expected_plate)
    if not a or not b:
        return False
    # Cheap fast-path before computing the full DP.
    if a == b:
        return True
    return levenshtein_distance(a, b) <= max_edit_distance


# --------------------------------------------------------------------------- #
# Plugin
# --------------------------------------------------------------------------- #


_PADDLEOCR_INSTALL_HINT = (
    "PaddleOCR is required for PaddlePlateOCR. Install with:\n"
    "    pip install 'paddleocr>=2.7.0' 'paddlepaddle>=2.6,<3.4'\n"
    "On Windows, paddlepaddle's oneDNN runtime can be flaky — pass "
    "enable_mkldnn=False to the constructor or set FLAGS_use_mkldnn=0 in "
    "the environment if you see 'ConvertPirAttribute2RuntimeAttribute' "
    "errors at predict time."
)


class PaddlePlateOCR(PlateOCR):
    """
    Reads a license plate from a vehicle crop using PaddleOCR-mobile.

    Parameters
    ----------
    lang:
        Recognition language passed to PaddleOCR. Defaults to ``'en'`` (Latin
        alphanumerics). For Arabic-script plates pass ``'arabic'``. The
        detection model is always the mobile variant.
    use_angle_cls:
        Whether to run the angle/textline-orientation classifier. Off by
        default — license-plate crops are usually axis-aligned by the upstream
        detector, and the classifier adds ~10 ms.
    enable_mkldnn:
        Toggle oneDNN. Off by default because paddlepaddle 3.x oneDNN trips a
        ``ConvertPirAttribute2RuntimeAttribute`` NotImpl at predict time — on
        Windows AND Linux (verified in production 2026-07-16: every read
        failed on the Linux Xeon with paddle 3.x). Do not enable until the
        image ships a paddle build where oneDNN+PIR actually works; use
        ``cpu_threads`` for speed instead.
    model_dir:
        Optional override for the PaddleOCR home directory (where det/rec
        weights are downloaded). Defaults to ``~/.paddlex/official_models``.
    min_crop_h, min_crop_w:
        If the input crop is smaller than these dimensions it is upsampled
        bicubically to twice that size before OCR runs. Tiny crops are the
        single biggest failure mode for mobile recognisers.
    rec_score_thresh:
        Minimum per-line recognition confidence kept. Lines below this are
        discarded before aggregation.
    hud_top_mask_ratio, hud_bottom_mask_ratio:
        Fraction of the crop height to blank out (filled with black) before
        running OCR. Defaults to 0.08 each — eliminates the burned-in camera
        name overlay (top corner) and the HH:MM:SS timestamp band (bottom)
        that surveillance recorders typically render. Set both to ``0.0`` when
        feeding pre-cropped plate ROIs (otherwise you'll mask the plate). The
        defaults were chosen from a 626-crop diagnostic sweep on this facility
        where PaddleOCR was returning confident "CAM01GFFRONTR" / "103446"
        readings from the HUD instead of the actual plate text.
    plate_roi_*:
        Geometric heuristic plate-region crop applied AFTER HUD masking but
        BEFORE OCR. Vehicle plates sit in the bottom-centre of a standard
        vehicle bbox: roughly ``[plate_roi_top:plate_roi_bottom]`` rows and
        ``[plate_roi_left:plate_roi_right]`` columns. Defaults are
        ``top=0.50, bottom=0.95, left=0.20, right=0.80`` — keeps the bottom
        ~half of the crop and the central 60 % horizontally, which captures
        the plate on head-on/angled vehicle crops while dropping windshield
        stickers, branding, and the wider chassis background. Set
        ``plate_roi_enabled=False`` when feeding an already-tight plate ROI
        (e.g., from a future plate-detection model).
    """

    DEFAULT_LANG = "en"
    DEFAULT_MIN_H = 32
    DEFAULT_MIN_W = 64
    # Mobile det/rec model names — chosen to keep the on-disk footprint
    # and per-call latency small. The server variants are ~10x larger and
    # ~5x slower without an accuracy gain that matters at the marginal
    # band the cascade will gate this OCR to.
    DEFAULT_DET_V3 = "PP-OCRv5_mobile_det"
    DEFAULT_REC_V3_EN = "en_PP-OCRv5_mobile_rec"
    DEFAULT_HUD_TOP_RATIO = 0.08
    DEFAULT_HUD_BOTTOM_RATIO = 0.08
    DEFAULT_PLATE_ROI_TOP = 0.50      # plate band starts at 50% from top
    DEFAULT_PLATE_ROI_BOTTOM = 0.95   # ends at 95% (drop very-bottom shadow)
    DEFAULT_PLATE_ROI_LEFT = 0.20     # plate sits centre-horizontally
    DEFAULT_PLATE_ROI_RIGHT = 0.80

    # Retry factor applied ONLY when a read comes back empty. A blank read is not
    # proof the plate is absent — a car parked far from the camera carries a plate
    # too few pixels wide for the text DETECTOR to find, even though it is legible.
    # 1.0 (or 0) disables the retry.
    DEFAULT_EMPTY_READ_UPSCALE = 2.0
    # Never enlarge past this many pixels: beyond it the retry costs more time than
    # the recogniser gains, and a crop that big already had plenty of resolution.
    DEFAULT_MAX_UPSCALED_PIXELS = 6_000_000

    def __init__(
        self,
        lang: str = DEFAULT_LANG,
        use_angle_cls: bool = True,
        enable_mkldnn: bool = False,
        model_dir: Optional[str] = None,
        min_crop_h: int = DEFAULT_MIN_H,
        min_crop_w: int = DEFAULT_MIN_W,
        rec_score_thresh: float = 0.5,
        det_model_name: Optional[str] = None,
        rec_model_name: Optional[str] = None,
        hud_top_mask_ratio: float = DEFAULT_HUD_TOP_RATIO,
        hud_bottom_mask_ratio: float = DEFAULT_HUD_BOTTOM_RATIO,
        plate_roi_enabled: bool = True,
        plate_roi_top: float = DEFAULT_PLATE_ROI_TOP,
        plate_roi_bottom: float = DEFAULT_PLATE_ROI_BOTTOM,
        plate_roi_left: float = DEFAULT_PLATE_ROI_LEFT,
        plate_roi_right: float = DEFAULT_PLATE_ROI_RIGHT,
        empty_read_upscale: float = DEFAULT_EMPTY_READ_UPSCALE,
        max_upscaled_pixels: int = DEFAULT_MAX_UPSCALED_PIXELS,
        cpu_threads: int = 4,
        upscale_factor: float = 1.0,
        preprocessing: Optional[List[str]] = None,
    ) -> None:
        # Fail fast at construction so callers know whether the plugin is
        # usable before the first frame arrives. The actual heavy model load
        # happens lazily on the first read().
        try:
            import paddleocr  # noqa: F401
        except Exception as exc:  # pragma: no cover - exercised by skip path
            raise ImportError(_PADDLEOCR_INSTALL_HINT) from exc
        try:
            import paddle  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise ImportError(_PADDLEOCR_INSTALL_HINT) from exc

        self._lang = lang
        self._use_angle_cls = bool(use_angle_cls)
        self._enable_mkldnn = bool(enable_mkldnn)
        self._model_dir = model_dir
        self._min_crop_h = int(min_crop_h)
        self._min_crop_w = int(min_crop_w)
        self._rec_score_thresh = float(rec_score_thresh)
        self._empty_read_upscale = float(empty_read_upscale)
        self._max_upscaled_pixels = int(max_upscaled_pixels)
        # Paddle's math-library pool, set per-predictor at engine construction.
        # Deliberately decoupled from OMP_NUM_THREADS: the supervisor pins the
        # OMP env to 1 in the unpinned/quota regime (idle OMP workers spin-burn
        # the CFS budget), but a single-threaded det+rec pass on a weak server
        # core takes SECONDS and runs inline in the serial camera loop.
        self._cpu_threads = max(1, int(cpu_threads or 1))
        self._upscale_factor = float(upscale_factor or 1.0)
        self._preprocessing = list(preprocessing or [])
        # Default to PaddleOCR v5 mobile det/rec; callers can override for
        # an explicit model upgrade or downgrade.
        self._det_model_name = det_model_name or self.DEFAULT_DET_V3
        if rec_model_name is not None:
            self._rec_model_name: Optional[str] = rec_model_name
        elif lang == "en":
            self._rec_model_name = self.DEFAULT_REC_V3_EN
        else:
            # For other languages let PaddleOCR pick the matching mobile rec.
            self._rec_model_name = None

        # Clamp HUD mask ratios into [0, 0.49] so we never blank the whole crop.
        self._hud_top_mask_ratio = max(0.0, min(0.49, float(hud_top_mask_ratio)))
        self._hud_bottom_mask_ratio = max(0.0, min(0.49, float(hud_bottom_mask_ratio)))

        # Plate-ROI heuristic. Sanity-check the bounds so we never invert them.
        self._plate_roi_enabled = bool(plate_roi_enabled)
        rt = max(0.0, min(0.99, float(plate_roi_top)))
        rb = max(rt + 0.01, min(1.0, float(plate_roi_bottom)))
        rl = max(0.0, min(0.99, float(plate_roi_left)))
        rr = max(rl + 0.01, min(1.0, float(plate_roi_right)))
        self._plate_roi_top = rt
        self._plate_roi_bottom = rb
        self._plate_roi_left = rl
        self._plate_roi_right = rr

        self._engine = None
        self._api_version: Optional[int] = None  # 2 or 3, resolved at first use
        # The paddle predictor is NOT thread-safe. Slot-identify OCR runs on a
        # background worker (see engine async_slot_ocr) while the gate/entry path
        # may read on the main thread — serialise every read through one lock so a
        # single instance can be shared without corrupting the predictor state.
        self._read_lock = threading.Lock()

        # Suppress paddle's oneDNN noise unless an env var explicitly enables it.
        if not enable_mkldnn:
            os.environ.setdefault("FLAGS_use_mkldnn", "0")

    # ----- Lazy backend init ---------------------------------------------- #

    def _ensure_engine(self) -> None:
        """Construct the underlying PaddleOCR engine on first use."""
        if self._engine is not None:
            return

        from paddleocr import PaddleOCR  # type: ignore

        # PaddleOCR 3.x uses kw-only flags like ``use_textline_orientation``
        # and ``enable_mkldnn``; 2.x uses ``use_angle_cls`` / ``show_log``.
        # We try the 3.x signature first and fall back to 2.x.
        v3_kwargs: dict = {}
        # PaddleOCR 3.x emits a warning if ``lang`` is supplied alongside
        # explicit model names, since the names already pin the language.
        # Only pass ``lang`` when no explicit rec model is given.
        if not self._rec_model_name:
            v3_kwargs["lang"] = self._lang
        if self._model_dir:
            v3_kwargs["text_detection_model_dir"] = self._model_dir
            v3_kwargs["text_recognition_model_dir"] = self._model_dir
        # Force mobile det/rec by name so PaddleOCR doesn't default to the
        # heavier server variants — the server det is ~80 MB and ~5x slower.
        if self._det_model_name:
            v3_kwargs["text_detection_model_name"] = self._det_model_name
        if self._rec_model_name:
            v3_kwargs["text_recognition_model_name"] = self._rec_model_name

        # cpu_threads tried first, dropped on TypeError — never let a thread
        # knob demote us to the 2.x code path on a 3.x install.
        for extra in ({"cpu_threads": self._cpu_threads}, {}):
            try:
                self._engine = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=self._use_angle_cls,
                    enable_mkldnn=self._enable_mkldnn,
                    device="cpu",
                    **v3_kwargs,
                    **extra,
                )
                self._api_version = 3
                break
            except TypeError:
                continue
        if self._engine is None:
            # 2.x signature — fall back. Reset common_kwargs (model dirs use
            # the older ``det_model_dir`` / ``rec_model_dir`` names).
            kwargs_2x = {"lang": self._lang, "use_angle_cls": self._use_angle_cls}
            if self._model_dir:
                kwargs_2x["det_model_dir"] = self._model_dir
                kwargs_2x["rec_model_dir"] = self._model_dir
            try:
                kwargs_2x["enable_mkldnn"] = self._enable_mkldnn
                kwargs_2x["show_log"] = False
                kwargs_2x["cpu_threads"] = self._cpu_threads
                self._engine = PaddleOCR(**kwargs_2x)
            except TypeError:
                # ultra-old signature — drop the optional kwargs
                kwargs_2x.pop("enable_mkldnn", None)
                kwargs_2x.pop("show_log", None)
                kwargs_2x.pop("cpu_threads", None)
                self._engine = PaddleOCR(**kwargs_2x)
            self._api_version = 2

        logger.info(
            "[PaddlePlateOCR] initialised: api=%s lang=%s angle_cls=%s mkldnn=%s threads=%s",
            self._api_version,
            self._lang,
            self._use_angle_cls,
            self._enable_mkldnn,
            self._cpu_threads,
        )

    # ----- Preprocessing -------------------------------------------------- #

    def _mask_hud(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Blank the top/bottom HUD bands so PaddleOCR ignores burned-in text.

        Surveillance recorders typically render the camera name in the top
        corner and a HH:MM:SS timestamp in the bottom band. PaddleOCR reads
        both at 0.95+ confidence and treats them as plate text. We blank
        those bands with solid black before OCR runs.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr
        if self._hud_top_mask_ratio <= 0.0 and self._hud_bottom_mask_ratio <= 0.0:
            return crop_bgr

        out = crop_bgr.copy()
        h = out.shape[0]
        top_px = int(round(h * self._hud_top_mask_ratio))
        bottom_px = int(round(h * self._hud_bottom_mask_ratio))
        if top_px > 0:
            out[:top_px, :] = 0
        if bottom_px > 0:
            out[h - bottom_px:, :] = 0
        return out

    def _extract_plate_roi(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Crop the geometric plate region (bottom-centre band of a vehicle).

        Plates sit in the lower-middle of a standard vehicle bbox.
        Returns a *view* of the original (no copy) for performance; callers
        that mutate the result should `.copy()` it themselves.
        """
        if not self._plate_roi_enabled or crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr
        h, w = crop_bgr.shape[:2]
        y0 = max(0, int(round(h * self._plate_roi_top)))
        y1 = min(h, int(round(h * self._plate_roi_bottom)))
        x0 = max(0, int(round(w * self._plate_roi_left)))
        x1 = min(w, int(round(w * self._plate_roi_right)))
        if y1 - y0 < 8 or x1 - x0 < 16:
            # ROI degenerate (crop too small) — fall back to the full input.
            return crop_bgr
        return crop_bgr[y0:y1, x0:x1]

    def _preprocess(
        self,
        crop_bgr: np.ndarray,
        apply_roi: bool = True,
        apply_hud_mask: bool = True,
    ) -> np.ndarray:
        """Mask HUD bands, crop the geometric plate ROI, then upsample tiny crops.

        ``apply_roi=False`` skips the bottom-centre plate-ROI crop. That crop
        assumes a head-on GATE view where the plate sits at 50-95% of the box
        height; on a ceiling-mounted SLOT camera looking DOWN, the plate sits at
        the very bottom of the box and the 50-95% band lands on the grille (BMW
        kidney / Lexus spindle slats read as garbage). Slot reads pass False and
        let PaddleOCR's own text detector localise the plate in the full car crop.

        ``apply_hud_mask=False`` skips the HUD bands. THIS IS REQUIRED WHEN THE
        INPUT IS ALREADY A TIGHT PLATE CROP. The HUD ratios are fractions of the
        input's HEIGHT, so they only mean "the recorder's overlay strip" on a
        whole-frame-ish vehicle crop. On a ~130x35 plate box from the region
        detector, 0.08 top + 0.08 bottom blanks ~3px at each edge — ~17% of the
        plate, and precisely the rows carrying glyph ascenders/descenders. Left on,
        it silently degrades exactly the reads this path exists to improve.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr

        # 1. Mask HUD bands FIRST so the ROI crop doesn't pull in timestamp
        #    pixels from a corner.
        if apply_hud_mask:
            crop_bgr = self._mask_hud(crop_bgr)

        # 2. Geometric plate-ROI crop. Drops windshield stickers, branding,
        #    wide chassis background — leaves the bottom-centre region where
        #    license plates live on head-on / angled vehicle bboxes. Skipped for
        #    slot cameras, whose downward geometry puts the plate below this band.
        if apply_roi:
            crop_bgr = self._extract_plate_roi(crop_bgr)

        h, w = crop_bgr.shape[:2]

        # 3. Scale uniformly to lift the short side to roughly twice the floor if tiny
        if h < self._min_crop_h or w < self._min_crop_w:
            scale_h = (2 * self._min_crop_h) / h if h < self._min_crop_h else 1.0
            scale_w = (2 * self._min_crop_w) / w if w < self._min_crop_w else 1.0
            scale = max(scale_h, scale_w)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            try:
                import cv2
                crop_bgr = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            except Exception as exc:  # pragma: no cover
                logger.debug("[PaddlePlateOCR] fallback resize failed: %r", exc)

        # 4. Custom upscale / Super-resolution
        if hasattr(self, "_upscale_factor") and self._upscale_factor > 1.0:
            try:
                import cv2
                h, w = crop_bgr.shape[:2]
                new_w = max(1, int(round(w * self._upscale_factor)))
                new_h = max(1, int(round(h * self._upscale_factor)))
                crop_bgr = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            except Exception as exc:
                logger.debug("[PaddlePlateOCR] custom upscale failed: %r", exc)

        # 5. Configurable image enhancements
        if hasattr(self, "_preprocessing") and self._preprocessing:
            try:
                import cv2
                for step in self._preprocessing:
                    if step == "clahe":
                        if len(crop_bgr.shape) == 3 and crop_bgr.shape[2] == 3:
                            lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
                            l_channel, a_channel, b_channel = cv2.split(lab)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            cl = clahe.apply(l_channel)
                            limg = cv2.merge((cl, a_channel, b_channel))
                            crop_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                        else:
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            crop_bgr = clahe.apply(crop_bgr)
                    elif step == "sharpen":
                        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                        crop_bgr = cv2.filter2D(crop_bgr, -1, kernel)
                    elif step == "denoise":
                        # Bilateral filter is edge-preserving, which is ideal for text
                        crop_bgr = cv2.bilateralFilter(crop_bgr, 5, 50, 50)
                    elif step == "threshold":
                        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
                        thresh = cv2.adaptiveThreshold(
                            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                        )
                        if len(crop_bgr.shape) == 3:
                            crop_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                        else:
                            crop_bgr = thresh
                    else:
                        # Never silently ignore a step the operator asked for — a
                        # typo'd name would otherwise look exactly like "this
                        # enhancement does nothing", which is unfalsifiable from
                        # the outside. Warned once per read; the list is tiny.
                        logger.warning(
                            "[PaddlePlateOCR] unknown slot_ocr_preprocessing step "
                            "%r — skipped (valid: clahe, sharpen, denoise, threshold)",
                            step,
                        )
            except Exception as exc:
                logger.debug("[PaddlePlateOCR] custom preprocessing step failed: %r", exc)

        return crop_bgr

    # ----- Result parsing ------------------------------------------------- #

    @staticmethod
    def _parse_result_v3(res) -> List[Tuple[str, float]]:
        """Extract (text, confidence) pairs from a PaddleOCR 3.x result."""
        out: List[Tuple[str, float]] = []
        if not res:
            return out
        for item in res:
            # 3.x items expose .json / dict-like access; fall back to attribute.
            rec_texts = None
            rec_scores = None
            if isinstance(item, dict):
                rec_texts = item.get("rec_texts")
                rec_scores = item.get("rec_scores")
            else:
                rec_texts = getattr(item, "rec_texts", None)
                rec_scores = getattr(item, "rec_scores", None)
                # PaddleOCR 3.x results expose .json["res"] as well; try that.
                if rec_texts is None:
                    json_blob = getattr(item, "json", None)
                    if isinstance(json_blob, dict):
                        res_blob = json_blob.get("res") or json_blob
                        if isinstance(res_blob, dict):
                            rec_texts = res_blob.get("rec_texts")
                            rec_scores = res_blob.get("rec_scores")

            if not rec_texts:
                continue
            scores = list(rec_scores) if rec_scores is not None else [
                1.0
            ] * len(rec_texts)
            for text, score in zip(rec_texts, scores):
                if text is None:
                    continue
                try:
                    sc = float(score)
                except (TypeError, ValueError):
                    sc = 0.0
                out.append((str(text), sc))
        return out

    @staticmethod
    def _parse_result_v2(res) -> List[Tuple[str, float]]:
        """Extract (text, confidence) pairs from a PaddleOCR 2.x result.

        The 2.x ``ocr`` method returns ``[[ [box, (text, conf)], ... ]]`` —
        one outer list per image. We always pass a single image so we index
        ``res[0]``.
        """
        out: List[Tuple[str, float]] = []
        if not res:
            return out
        page = res[0] if isinstance(res, list) else res
        if not page:
            return out
        for line in page:
            try:
                # line is typically [box, (text, conf)]
                payload = line[1]
                if isinstance(payload, (tuple, list)) and len(payload) >= 2:
                    text, conf = payload[0], payload[1]
                    out.append((str(text), float(conf)))
                elif isinstance(payload, dict):
                    text = payload.get("text") or payload.get("rec_text")
                    conf = payload.get("confidence") or payload.get("rec_score") or 0.0
                    if text is not None:
                        out.append((str(text), float(conf)))
            except (IndexError, TypeError, ValueError):
                continue
        return out

    # ----- PlateOCR ABC --------------------------------------------------- #

    def read(
        self,
        crop_bgr: np.ndarray,
        *,
        allow_retry: bool = True,
        apply_plate_roi: bool = True,
        apply_hud_mask: bool = True,
    ) -> Tuple[str, float]:
        """
        Run OCR on a BGR crop and return ``(normalised_text, mean_conf)``.

        ``apply_hud_mask=False`` MUST be passed when ``crop_bgr`` is already a tight
        plate ROI (e.g. from the plate-region detector) — see :meth:`_preprocess`.

        ``allow_retry`` gates the enlarged second pass (see below). The enlarged pass is
        what rescues a distant plate, but on a slot whose plate is genuinely out of frame
        it finds spurious text regions, runs the recogniser on them, discards the lot,
        and costs ~670ms for nothing. Callers with an attempt budget should spend the
        retry on the first few attempts only.

        * Normalisation: uppercase, strip non-alphanumerics.
        * Fragments are concatenated in PaddleOCR's natural detection order
          (top-to-bottom, left-to-right). The plan's brief calls for sorting
          by descending confidence; we keep detection order because for a
          standard single-line plate that preserves the reading order ("ABC"
          + "1234" -> "ABC1234"), whereas confidence-sort can reverse it.
          Low-confidence fragments (below ``rec_score_thresh``) are dropped
          before concatenation.
        * Returns ``("", 0.0)`` on empty crops, OCR errors, or no detections
          above the per-line score threshold.
        """
        if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
            return ("", 0.0)

        try:
            self._ensure_engine()
        except Exception as exc:
            logger.warning("[PaddlePlateOCR] engine init failed: %r", exc)
            return ("", 0.0)

        # PERF_TRACE stage: OCR runs INLINE in the engine's serial camera loop
        # (inside the "slot" stage — so ocr is a subset of slot, not additive).
        # It is the dominant per-frame cost when armed slots read empty: the
        # enlarged retry alone measured 199-759ms on a fast dev core, seconds
        # on the production Xeon.
        with perf_trace.stage("ocr"):
            with self._read_lock:
                return self._read_timed(
                    crop_bgr, allow_retry, apply_plate_roi, apply_hud_mask
                )

    def _read_timed(
        self,
        crop_bgr: np.ndarray,
        allow_retry: bool,
        apply_plate_roi: bool,
        apply_hud_mask: bool = True,
    ) -> Tuple[str, float]:
        prepped = self._preprocess(
            crop_bgr, apply_roi=apply_plate_roi, apply_hud_mask=apply_hud_mask
        )
        text, conf = self._infer(prepped)
        if text:
            return (text, conf)

        if not allow_retry:
            # Caller has spent its retry budget for this slot (see
            # ocr_upscale_retry_max_attempts). A slot whose plate is genuinely not in
            # frame would otherwise pay the enlarged pass on all 12 attempts.
            return ("", 0.0)

        # Empty read — retry once on the ENLARGED plate region.
        #
        # A blank read is NOT the same as "no plate in frame". The crop is a whole
        # vehicle, so a car parked far from the camera can carry a plate only ~120px
        # wide inside a 540px crop, and the text DETECTOR never finds it — even though
        # the plate is perfectly legible to a human. _preprocess only upsamples crops
        # below a tiny 32x64 floor, so a large-but-distant crop like that is never
        # enlarged and reads as ''.
        #
        # Measured 2026-07-13 on slot B13 (CAM-24, a C-level slot dark for 60 attempts):
        #     native 540x562  -> ''          (what production did)
        #     retry at x2     -> '9990BHD'   -> matches BHD-9990
        #
        # Enlarge the ALREADY-PREPROCESSED region, not the raw crop: it is ~3x smaller
        # (HUD masked, cropped to the plate ROI), so the retry costs 199ms instead of
        # 759ms on a slot whose plate is genuinely absent — and that is the common case,
        # paid on every one of the 12 attempts. Same read either way.
        scale = float(self._empty_read_upscale or 0.0)
        if scale <= 1.0 or prepped is None or getattr(prepped, "size", 0) == 0:
            return ("", 0.0)
        try:
            import cv2

            h, w = prepped.shape[:2]
            if h * w * scale * scale > self._max_upscaled_pixels:
                return ("", 0.0)  # already huge; enlarging further just costs time
            bigger = cv2.resize(
                prepped, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
            )
        except Exception:
            return ("", 0.0)
        # Feed the enlarged region STRAIGHT to the engine. Running _preprocess again
        # would re-apply the plate-ROI crop to an image that is already the ROI, cutting
        # into the plate itself.
        return self._infer(bigger)

    def _infer(self, prepped: np.ndarray) -> Tuple[str, float]:
        """OCR an ALREADY-preprocessed image. ``("", 0.0)`` when nothing is read."""
        if prepped is None or getattr(prepped, "size", 0) == 0:
            return ("", 0.0)

        try:
            if self._api_version == 3:
                # PaddleOCR 3.x: predict() returns a list of result objects.
                res = self._engine.predict(prepped)
                pairs = self._parse_result_v3(res)
            else:
                # PaddleOCR 2.x: ocr() returns a nested list.
                res = self._engine.ocr(prepped, cls=self._use_angle_cls)
                pairs = self._parse_result_v2(res)
        except Exception as exc:
            logger.warning("[PaddlePlateOCR] read failed: %r", exc)
            return ("", 0.0)

        if not pairs:
            return ("", 0.0)

        # Drop low-confidence reads.
        kept = [
            (text, conf)
            for text, conf in pairs
            if conf >= self._rec_score_thresh and text
        ]
        if not kept:
            return ("", 0.0)

        # Preserve PaddleOCR's natural detection order — top-to-bottom,
        # left-to-right — so a multi-fragment plate ("ABC" + "1234") joins
        # as the human would read it ("ABC1234"). Sorting by confidence
        # could invert the order on equally-confident fragments.
        concatenated = "".join(text for text, _ in kept)
        normalised = normalise_plate_text(concatenated)

        if not normalised:
            return ("", 0.0)

        mean_conf = float(np.mean([conf for _, conf in kept]))
        return (normalised, mean_conf)
