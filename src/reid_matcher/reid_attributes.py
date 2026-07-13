from typing import Optional, Tuple

import cv2
import numpy as np


def dominant_color_hsv(bgr: Optional[np.ndarray]) -> Optional[Tuple[float, float, float]]:
    """Return mean HSV of the center crop to avoid background/window bias."""
    if bgr is None or bgr.size == 0:
        return None

    h, w = bgr.shape[:2]
    center = bgr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    if center.size == 0:
        return None

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    return tuple(float(v) for v in np.mean(hsv.reshape(-1, 3), axis=0))


def color_compatible(hsv_a, hsv_b, h_tol=25, s_tol=80, v_tol=80) -> bool:
    """Loose compatibility check that only rejects obvious color mismatches."""
    if hsv_a is None or hsv_b is None:
        return True

    dh = min(abs(hsv_a[0] - hsv_b[0]), 180 - abs(hsv_a[0] - hsv_b[0]))
    ds = abs(hsv_a[1] - hsv_b[1])
    dv = abs(hsv_a[2] - hsv_b[2])
    return dh < h_tol and ds < s_tol and dv < v_tol


# Hue is unstable/meaningless when saturation is low (grey/silver/white) OR when
# value is low (dark/black): such a crop is "achromatic", distinguished only by
# brightness. A dark charcoal car reads s~40-50, so key off value too.
_S_ACHROMATIC = 50
_V_ACHROMATIC = 70


def _is_achromatic(s, v) -> bool:
    return s < _S_ACHROMATIC or v < _V_ACHROMATIC


def body_colour_compatible(hsv_a, hsv_b, v_tol=90) -> bool:
    """Saturation-aware body-colour veto — the one used to reject a wrong-car
    crop/query against a car's ground-truth colour.

    Unlike :func:`color_compatible`, this does NOT compare hue for achromatic
    colours, where hue is pure noise: most cars are black / grey / silver /
    white, and comparing their unstable hue wrongly flags the same car under
    different lighting. Rules:

      * both achromatic (low saturation) → compatible iff brightness is close
        (separates black from white, tolerates lighting on the same grey);
      * one achromatic, one chromatic → incompatible only when the chromatic
        side is vividly saturated (a clearly-coloured car vs a neutral one);
        borderline saturations fail-safe to compatible;
      * both chromatic → hue must be close (and brightness within tolerance).

    Fail-open when either colour is missing. Deliberately biased toward
    "compatible" so a legitimate reference is never dropped on a marginal call —
    the gross mismatches (white-vs-dark, red-vs-blue) are what this catches."""
    if hsv_a is None or hsv_b is None:
        return True
    ha, sa, va = hsv_a
    hb, sb, vb = hsv_b
    a_achro = _is_achromatic(sa, va)
    b_achro = _is_achromatic(sb, vb)
    if a_achro and b_achro:
        return abs(va - vb) < v_tol
    if a_achro != b_achro:
        chroma_s = sb if a_achro else sa
        if chroma_s >= 90:
            return False  # vivid colour vs neutral -> incompatible
        # A muted-chroma car (champagne / beige / tan) escapes the saturation
        # test above, but it is still a DIFFERENT car from a dark neutral when
        # brightness is far apart — a bright tan Lexus vs a black sedan. The
        # saturation test alone missed this (the leak that poisoned a dark car's
        # gallery with a champagne one); value catches it. Same v_tol as the
        # both-achromatic branch, so genuine lighting variation still passes.
        return abs(va - vb) < v_tol
    # Both chromatic. Use a wide hue tolerance: the gate (daylight) and the
    # garage floor cameras (artificial light) shift a car's apparent hue, so a
    # moderate difference is lighting, not a different car. Only a large hue gap
    # (e.g. teal vs orange) is a genuine colour mismatch.
    dh = min(abs(ha - hb), 180 - abs(ha - hb))
    return dh < 40 and abs(va - vb) < v_tol
