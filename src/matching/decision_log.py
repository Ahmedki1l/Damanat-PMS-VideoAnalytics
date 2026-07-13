"""Append-only JSONL log of every slot-identity decision.

WHY THIS EXISTS
    A ranker trained offline on the gallery alone does NOT beat plain ReID — measured
    2026-07-13, cold rank-1 fell 74.9% -> 71.1% and the trainer refused to ship it. The
    reason is not the model, it is the features: the signals that actually break ties
    only exist at the moment of the decision and cannot be reconstructed afterwards.

        locked_elsewhere   a car cannot be in two slots (offline this is unknowable)
        ocr_confirms       the second witness
        detector conf, view quality, sharpness   when to distrust the crop
        time since entry, area reachability      route constraints

    So we record them as they happen. Every OCR-confirmed bind yields one positive and
    up to k-1 hard negatives — the other cars ReID thought were plausible at that exact
    instant. That is the training set, and it is generated for free by traffic that is
    already flowing.

DESIGN CONSTRAINTS
    * NEVER block the frame loop. A bounded queue plus a daemon writer thread; when the
      queue is full we drop and count rather than stall a camera group.
    * ONE FILE PER PROCESS. The supervisor runs several workers, and zoning_trace.log
      already demonstrates what happens when they share a handle: interleaved, corrupt
      lines. Suffix by worker/pid.
    * The record's ``features`` block is built by src.matching.slot_rank_features, the
      same module the trainer imports. If the two ever diverge the model degrades
      silently, so there is exactly one definition.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _default(o: Any):
    if isinstance(o, datetime):
        return o.isoformat()
    if hasattr(o, "item"):          # numpy scalar
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


class DecisionLog:
    """Per-process JSONL writer. Construct once; call :meth:`emit` from the frame loop."""

    def __init__(
        self,
        directory: str = "logs/decisions",
        worker: str = "",
        max_queue: int = 2000,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self._dir = directory
        self._worker = worker or f"pid{os.getpid()}"
        self._q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=max_queue)
        self._dropped = 0
        self._written = 0
        self._day: Optional[str] = None
        self._fh = None
        self._thread: Optional[threading.Thread] = None
        if self.enabled:
            os.makedirs(self._dir, exist_ok=True)
            self._thread = threading.Thread(
                target=self._run, name="decision-log", daemon=True
            )
            self._thread.start()
            atexit.register(self.close)

    # ---- producer side (frame loop) ------------------------------------ #
    def emit(self, record: Dict[str, Any]) -> None:
        """Queue a record. Never raises, never blocks."""
        if not self.enabled:
            return
        record.setdefault("v", SCHEMA_VERSION)
        record.setdefault("worker", self._worker)
        record.setdefault("ts", datetime.now(timezone.utc).isoformat())
        try:
            self._q.put_nowait(record)
        except queue.Full:
            # Dropping a training row is a far smaller problem than stalling a camera
            # group behind a disk write. Counted, and reported at shutdown.
            self._dropped += 1

    # ---- consumer side (daemon thread) --------------------------------- #
    def _file_for_today(self):
        day = datetime.now().strftime("%Y-%m-%d")
        if day != self._day or self._fh is None:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
            path = os.path.join(self._dir, f"decisions_{self._worker}_{day}.jsonl")
            self._fh = open(path, "a", encoding="utf-8")
            self._day = day
        return self._fh

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                fh = self._file_for_today()
                fh.write(json.dumps(item, default=_default, ensure_ascii=False) + "\n")
                fh.flush()          # flush, but never fsync — durability is not worth the stall
                self._written += 1
            except Exception as exc:
                logger.debug("[decision-log] write failed: %r", exc)

    def close(self) -> None:
        if not self.enabled or self._thread is None:
            return
        try:
            self._q.put_nowait(None)
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        if self._dropped:
            logger.warning(
                "[decision-log] wrote %d records, DROPPED %d (queue full — raise "
                "decision_log_queue_max if this is not rare)",
                self._written, self._dropped,
            )
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass


def build_record(
    *,
    slot_id: str,
    camera_id: str,
    floor: Optional[str],
    area: Optional[str],
    is_reserved: bool,
    reserved_for: Optional[str],
    attempt: int,
    max_attempts: int,
    query: Any,                 # QuerySignals
    candidates: List[Any],      # List[CandidateSignals], ranked
    feature_matrix: Any,        # np.ndarray (n_candidates, N_FEATURES)
    feature_names: List[str],
    rejected: List[Any],        # List[RejectedCandidate]
    ocr_text: str,
    ocr_conf: float,
    decision: str,              # "BIND" | "ABSTAIN_NO_MATCH" | "ABSTAIN_NO_READ" | ...
    bound_plate: Optional[str],
    label_source: Optional[str],  # "ocr" when bound_plate came from a read
) -> Dict[str, Any]:
    """Assemble one decision record.

    ``bound_plate`` IS THE LABEL. Every other candidate in ``candidates`` is a hard
    negative — a car ReID genuinely believed was plausible at this instant, which is far
    more valuable than a random negative.
    """
    rows = []
    for i, c in enumerate(candidates):
        feats = {
            name: (None if feature_matrix[i][j] != feature_matrix[i][j]  # NaN
                   else float(feature_matrix[i][j]))
            for j, name in enumerate(feature_names)
        }
        rows.append(
            {
                "plate": c.plate,
                "rank": c.rank,
                "is_label": bool(bound_plate and c.plate == bound_plate),
                "features": feats,
            }
        )

    return {
        "event": "slot_identity_attempt",
        "slot": {"slot_id": slot_id, "is_reserved": is_reserved,
                 "reserved_for": reserved_for},
        "camera": {"camera_id": camera_id, "floor": floor, "area": area},
        "attempt": attempt,
        "max_attempts": max_attempts,
        "query": {
            "crop_sharpness": float(query.crop_sharpness),
            "crop_area": float(query.crop_area),
            "crop_aspect": float(query.crop_aspect),
            "active_candidates": int(query.active_candidates),
        },
        "candidates": rows,
        "rejected": [
            {"plate": r.plate, "raw_rank": r.raw_rank, "score": r.score,
             "reason": r.reason, "detail": r.detail}
            for r in (rejected or [])
        ],
        "ocr": {"text": ocr_text or "", "conf": float(ocr_conf or 0.0)},
        "decision": decision,
        "label": {"plate": bound_plate, "source": label_source} if bound_plate else None,
    }
