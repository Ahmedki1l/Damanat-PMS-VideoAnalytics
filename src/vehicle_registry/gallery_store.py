"""
src.vehicle_registry.gallery_store — per-car persistent multi-shot ReID gallery.

One folder per plate under ``<base_dir>/gallery/<safe_plate>/`` holding a small,
quality-gated set of reference crops and their embeddings, so a car's appearance
profile survives a restart and warm-starts the car on a return visit. This is
the durable backing for :class:`VehicleSession`'s in-memory
``reference_feature_vectors`` — the crops are the source of truth, the vectors a
model-tagged cache (re-embedded from the crops when the ReID model changes).

Folder layout::

    <base_dir>/gallery/<safe_plate>/
        meta.json                 # plate, model_tag, timestamps, per-ref records
        crop_<ts>_<tok>.jpg       # reference crop (BGR)
        crop_<ts>_<tok>.npy       # its L2-normed float32 embedding

All methods are best-effort and swallow I/O errors (logged) so gallery
persistence can never crash the per-frame pipeline. Thread-safety is provided by
the caller (``VehicleRegistry`` holds its RLock); this class does no locking of
its own beyond being stateless per call.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_META = "meta.json"
_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_plate(plate: str) -> str:
    """Filesystem-safe folder name for a plate (keeps A-Z0-9 and ``-``/``_``)."""
    return _SAFE.sub("_", (plate or "").strip()).strip("._") or "UNKNOWN"


class VehicleGalleryStore:
    def __init__(self, base_dir: str, model_tag: str, max_refs: int = 10):
        self._root = os.path.join(base_dir or "vehicle_images", "gallery")
        self._model_tag = model_tag or "unknown"
        self._max_refs = max(1, int(max_refs))

    # ------------------------------------------------------------------ #
    # Paths / meta
    # ------------------------------------------------------------------ #
    def _plate_dir(self, plate: str) -> str:
        return os.path.join(self._root, safe_plate(plate))

    def has(self, plate: str) -> bool:
        return os.path.isfile(os.path.join(self._plate_dir(plate), _META))

    def _read_meta(self, plate: str) -> Optional[dict]:
        path = os.path.join(self._plate_dir(plate), _META)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def _write_meta(self, plate: str, meta: dict) -> None:
        d = self._plate_dir(plate)
        try:
            os.makedirs(d, exist_ok=True)
            tmp = os.path.join(d, _META + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
            os.replace(tmp, os.path.join(d, _META))
        except OSError as exc:
            logger.warning("[gallery] meta write failed for %s: %r", plate, exc)

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    def save_ref(
        self,
        plate: str,
        crop_bgr: np.ndarray,
        vector: np.ndarray,
        quality: float,
        camera_id: str = "",
        timestamp: Optional[datetime] = None,
        gate_only: bool = False,
    ) -> Optional[str]:
        """Persist one reference (crop + embedding), append to meta, prune to cap.

        Returns the crop filename on success, else None. Eviction keeps the
        highest-``quality`` refs when the folder exceeds ``max_refs``.

        ``gate_only`` marks the wide gate-camera ANPR shot: the folder (and the
        entry photo) exist on disk from the moment the car enters, but the ref
        is excluded from :meth:`load_vectors` / :meth:`load_crops` so a
        warm-start can never ReID-match against the untrustworthy gate view.
        """
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0 or vector is None:
            return None
        now = timestamp or datetime.now()
        d = self._plate_dir(plate)
        stem = f"crop_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        try:
            os.makedirs(d, exist_ok=True)
            if not cv2.imwrite(os.path.join(d, stem + ".jpg"), crop_bgr):
                raise OSError("imwrite returned False")
            np.save(os.path.join(d, stem + ".npy"), np.asarray(vector, dtype=np.float32))
        except OSError as exc:
            logger.warning("[gallery] save_ref failed for %s: %r", plate, exc)
            return None

        meta = self._read_meta(plate) or {
            "plate": plate,
            "created_at": now.isoformat(),
            "refs": [],
        }
        meta["model_tag"] = self._model_tag
        meta["updated_at"] = now.isoformat()
        if camera_id:
            meta.setdefault("last_camera", camera_id)
        ref = {
            "crop": stem + ".jpg",
            "vec": stem + ".npy",
            "quality": float(quality),
            "camera": camera_id,
            "ts": now.isoformat(),
        }
        if gate_only:
            ref["gate"] = True
        meta.setdefault("refs", []).append(ref)
        self._prune_meta_inplace(d, meta)
        self._write_meta(plate, meta)
        return stem + ".jpg"

    def _prune_meta_inplace(self, plate_dir: str, meta: dict) -> None:
        """Keep the top ``max_refs`` refs by quality; delete evicted files."""
        refs = meta.get("refs", [])
        if len(refs) <= self._max_refs:
            return
        refs.sort(key=lambda r: r.get("quality", 0.0), reverse=True)
        keep, drop = refs[: self._max_refs], refs[self._max_refs :]
        for r in drop:
            for key in ("crop", "vec"):
                fn = r.get(key)
                if fn:
                    try:
                        os.remove(os.path.join(plate_dir, fn))
                    except OSError:
                        pass
        meta["refs"] = keep

    def stamp_exit(self, plate: str, timestamp: Optional[datetime] = None) -> None:
        """Record the car's exit time so the TTL GC ages the folder from here."""
        meta = self._read_meta(plate)
        if meta is None:
            return
        meta["last_exit"] = (timestamp or datetime.now()).isoformat()
        self._write_meta(plate, meta)

    def delete(self, plate: str) -> None:
        shutil.rmtree(self._plate_dir(plate), ignore_errors=True)

    @staticmethod
    def clear_all(base_dir: str) -> int:
        """Delete every per-plate gallery folder under ``<base_dir>/gallery``.

        Used by the ``--reset-plates`` command so a reset also forgets the
        persisted appearance galleries — otherwise a returning car would
        warm-start its wiped plate identity straight back from disk. Returns the
        number of plate folders removed. Static so callers (e.g. main.py before
        the registry is built) don't need a live store."""
        root = os.path.join(base_dir or "vehicle_images", "gallery")
        if not os.path.isdir(root):
            return 0
        removed = 0
        try:
            for entry in os.scandir(root):
                if entry.is_dir():
                    shutil.rmtree(entry.path, ignore_errors=True)
                    removed += 1
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("[gallery] clear_all failed under %s: %r", root, exc)
        return removed

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #
    def load_vectors(self, plate: str) -> Tuple[List[np.ndarray], Optional[str]]:
        """Return (vectors, stored_model_tag). Empty list when absent/unreadable.

        Vectors are returned primary-first (highest quality first) so the caller
        can use ``vectors[0]`` as the session's primary feature. Gate-only refs
        are excluded — they exist for the folder/photo guarantee, never for
        matching (see :meth:`save_ref`).
        """
        meta = self._read_meta(plate)
        if not meta:
            return [], None
        d = self._plate_dir(plate)
        refs = sorted(
            (r for r in meta.get("refs", []) if not r.get("gate")),
            key=lambda r: r.get("quality", 0.0),
            reverse=True,
        )
        vectors: List[np.ndarray] = []
        for r in refs:
            fn = r.get("vec")
            if not fn:
                continue
            try:
                vectors.append(np.load(os.path.join(d, fn)).astype(np.float32))
            except (OSError, ValueError):
                continue
        return vectors, meta.get("model_tag")

    def load_crops(self, plate: str) -> List[np.ndarray]:
        """Load the reference crop images (highest quality first) for re-embedding
        when the stored ``model_tag`` no longer matches the running model.
        Gate-only refs are excluded, mirroring :meth:`load_vectors`."""
        meta = self._read_meta(plate)
        if not meta:
            return []
        d = self._plate_dir(plate)
        refs = sorted(
            (r for r in meta.get("refs", []) if not r.get("gate")),
            key=lambda r: r.get("quality", 0.0),
            reverse=True,
        )
        crops: List[np.ndarray] = []
        for r in refs:
            fn = r.get("crop")
            if not fn:
                continue
            img = cv2.imread(os.path.join(d, fn))
            if img is not None and img.size > 0:
                crops.append(img)
        return crops

    def ref_count(self, plate: str) -> int:
        meta = self._read_meta(plate)
        return len(meta.get("refs", [])) if meta else 0

    # ------------------------------------------------------------------ #
    # Garbage collection
    # ------------------------------------------------------------------ #
    def gc(self, retention_days: float) -> int:
        """Delete plate folders whose newest activity is older than the TTL.

        Newest activity = max of the folder's file mtimes (covers both the last
        ref write and the ``stamp_exit`` meta rewrite). Returns folders removed.
        """
        if retention_days <= 0 or not os.path.isdir(self._root):
            return 0
        cutoff = time.time() - retention_days * 86400.0
        removed = 0
        try:
            entries = list(os.scandir(self._root))
        except OSError:
            return 0
        for entry in entries:
            if not entry.is_dir():
                continue
            try:
                newest = entry.stat().st_mtime
                for f in os.scandir(entry.path):
                    newest = max(newest, f.stat().st_mtime)
            except OSError:
                continue
            if newest < cutoff:
                shutil.rmtree(entry.path, ignore_errors=True)
                removed += 1
        if removed:
            logger.info("[gallery] GC removed %d folder(s) idle > %.1fd", removed, retention_days)
        return removed
