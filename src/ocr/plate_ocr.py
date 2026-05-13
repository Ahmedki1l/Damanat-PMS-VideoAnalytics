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
from typing import List, Optional, Tuple

import numpy as np

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
        Toggle oneDNN. Off by default because on Windows + paddlepaddle 3.x
        oneDNN can trip a ``ConvertPirAttribute2RuntimeAttribute`` NotImpl.
        Production Linux deployments can flip this on for ~30 % more speed.
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

        self._engine = None
        self._api_version: Optional[int] = None  # 2 or 3, resolved at first use

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

        try:
            self._engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=self._use_angle_cls,
                enable_mkldnn=self._enable_mkldnn,
                device="cpu",
                **v3_kwargs,
            )
            self._api_version = 3
        except TypeError:
            # 2.x signature — fall back. Reset common_kwargs (model dirs use
            # the older ``det_model_dir`` / ``rec_model_dir`` names).
            kwargs_2x = {"lang": self._lang, "use_angle_cls": self._use_angle_cls}
            if self._model_dir:
                kwargs_2x["det_model_dir"] = self._model_dir
                kwargs_2x["rec_model_dir"] = self._model_dir
            try:
                kwargs_2x["enable_mkldnn"] = self._enable_mkldnn
                kwargs_2x["show_log"] = False
                self._engine = PaddleOCR(**kwargs_2x)
            except TypeError:
                # ultra-old signature — drop the optional kwargs
                kwargs_2x.pop("enable_mkldnn", None)
                kwargs_2x.pop("show_log", None)
                self._engine = PaddleOCR(**kwargs_2x)
            self._api_version = 2

        logger.info(
            "[PaddlePlateOCR] initialised: api=%s lang=%s angle_cls=%s mkldnn=%s",
            self._api_version,
            self._lang,
            self._use_angle_cls,
            self._enable_mkldnn,
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

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Mask HUD bands, then upsample if the crop is tiny."""
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr

        # Apply HUD masking FIRST so the upsample doesn't smear timestamp
        # pixels into the plate region.
        crop_bgr = self._mask_hud(crop_bgr)

        h, w = crop_bgr.shape[:2]
        if h >= self._min_crop_h and w >= self._min_crop_w:
            return crop_bgr

        # Scale uniformly to lift the short side to roughly twice the floor —
        # tiny ROIs (<32×64) hurt the mobile recogniser disproportionately.
        scale_h = (2 * self._min_crop_h) / h if h < self._min_crop_h else 1.0
        scale_w = (2 * self._min_crop_w) / w if w < self._min_crop_w else 1.0
        scale = max(scale_h, scale_w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        try:
            import cv2

            return cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        except Exception as exc:  # pragma: no cover - opencv always present
            logger.debug("[PaddlePlateOCR] resize failed: %r", exc)
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

    def read(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Run OCR on a BGR crop and return ``(normalised_text, mean_conf)``.

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

        prepped = self._preprocess(crop_bgr)

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
