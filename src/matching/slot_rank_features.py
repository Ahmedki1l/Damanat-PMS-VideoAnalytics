"""Feature vector for one (parked-car query, candidate) pair — the slot ranker's input.

THE ONE PLACE the feature order is defined. Training and inference both import
``build_features`` from here; if they ever compute features differently the model silently
degrades and nothing fails loudly, which is the failure mode that quietly kills a ranker.

Why a ranker at all: measured 2026-07-13, ReID puts the right car in the top-5 ~98% of the
time but FIRST only ~88% (cold). The gap is not a missing signal, it is a combination
problem — and the signals that break the ties (a same-camera parked pose, a colour that
does not match, a car already locked in another slot) are all cheap and already computed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Order is the contract. Append only — never reorder or delete, or an existing model's
# column meanings shift under it.
FEATURE_NAMES: List[str] = [
    "reid_score",            # production-weighted best similarity (what ranking sorts on)
    "same_view_score",       # RAW similarity to a ref from THIS camera; 0 if never parked here
    "cross_view_score",      # weighted similarity to everything else (gate/other cameras)
    "warm",                  # 1 when a same-camera ref exists. Warm top-1 errs 2%, cold 25%.
    "margin_to_next",        # this candidate's score minus the NEXT candidate's
    "margin_to_top1",        # 0 for the leader; how far behind everyone else is
    "candidate_rank",        # 1-based position in the ReID ordering
    "active_candidates",     # how many cars were in contention at all
    "n_refs",                # gallery depth for this candidate
    "n_same_view_refs",      # how many times we have seen it parked at THIS camera
    "best_ref_is_gate",      # 1 when its best-matching ref is a gate photo (cross-view)
    "color_match",           # 1 when query and candidate colour labels agree
    "color_conf_query",      # colour classifier confidence on the live crop
    "color_conf_cand",       # ...and on the candidate's reference crop
    "type_match",            # 1 when vehicle-type labels agree
    "type_conf_query",
    "crop_sharpness",        # query crop quality — a blurred crop makes every score noisy
    "crop_area",
    "crop_aspect",
    "locked_elsewhere",      # 1 when this car is already parked in a DIFFERENT slot
]

N_FEATURES = len(FEATURE_NAMES)


@dataclass
class CandidateSignals:
    """Everything known about one candidate at one decision moment.

    Populated identically by the offline dataset builder and the live engine — that
    symmetry is the whole point of this module.
    """

    plate: str
    reid_score: float
    same_view_score: float
    cross_view_score: float
    warm: bool
    rank: int
    n_refs: int = 0
    n_same_view_refs: int = 0
    best_ref_is_gate: bool = True
    color_match: Optional[bool] = None
    color_conf_query: float = 0.0
    color_conf_cand: float = 0.0
    type_match: Optional[bool] = None
    type_conf_query: float = 0.0
    locked_elsewhere: bool = False


@dataclass
class QuerySignals:
    """Everything known about the crop we are trying to identify."""

    crop_sharpness: float = 0.0
    crop_area: float = 0.0
    crop_aspect: float = 0.0
    active_candidates: int = 0


def _tri(x: Optional[bool]) -> float:
    """Tri-state -> float. Unknown is NOT False: a colour classifier that could not run
    must not look like a colour that disagreed. LightGBM handles NaN natively as
    'missing', which is exactly the right semantics."""
    if x is None:
        return float("nan")
    return 1.0 if x else 0.0


def build_features(
    candidates: Sequence[CandidateSignals], query: QuerySignals
) -> np.ndarray:
    """(n_candidates, N_FEATURES) matrix, rows aligned with ``candidates``."""
    scores = [c.reid_score for c in candidates]
    top1 = max(scores) if scores else 0.0

    rows: List[List[float]] = []
    for i, c in enumerate(candidates):
        nxt = scores[i + 1] if i + 1 < len(scores) else 0.0
        rows.append(
            [
                c.reid_score,
                c.same_view_score,
                c.cross_view_score,
                1.0 if c.warm else 0.0,
                c.reid_score - nxt,
                top1 - c.reid_score,
                float(c.rank),
                float(query.active_candidates),
                float(c.n_refs),
                float(c.n_same_view_refs),
                1.0 if c.best_ref_is_gate else 0.0,
                _tri(c.color_match),
                c.color_conf_query,
                c.color_conf_cand,
                _tri(c.type_match),
                c.type_conf_query,
                query.crop_sharpness,
                query.crop_area,
                query.crop_aspect,
                1.0 if c.locked_elsewhere else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)
