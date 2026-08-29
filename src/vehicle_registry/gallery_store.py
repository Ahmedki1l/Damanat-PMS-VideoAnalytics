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

I/O failures are reported as unsuccessful writes instead of escaping into the
per-frame pipeline. Individual file/metadata replacements are atomic and
guarded by an internal lock. In strict mode only references carrying a verified
admission proof are matchable; older unverified records remain on disk and use
a restart-recoverable two-phase quarantine transaction.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_META = "meta.json"
_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def select_diverse_indices(
    vectors: List[np.ndarray], cap: int, seed_index: int = 0
) -> List[int]:
    """Greedy farthest-point selection of the ``cap`` most mutually-dissimilar
    vectors — the subset that maximally spans the appearance/viewpoint space.

    Used to prune a per-car ReID gallery for *coverage* instead of by
    sharpness/recency (which collapse it to one dominant viewpoint and make a
    query from a different camera/angle unmatchable). Vectors are assumed
    L2-normalised, so cosine similarity = dot product and distance = 1 - dot.

    Returns the kept indices (order not significant). When ``len(vectors) <=
    cap`` every index is returned.
    """
    n = len(vectors)
    if cap <= 0:
        return []
    if n <= cap:
        return list(range(n))
    mat = np.stack([np.asarray(v, dtype=np.float32).ravel() for v in vectors])
    seed = seed_index if 0 <= seed_index < n else 0
    kept = [seed]
    # nearest[i] = cosine similarity of vector i to its closest kept vector.
    nearest = mat @ mat[seed]
    while len(kept) < cap:
        masked = nearest.copy()
        masked[kept] = np.inf  # never re-pick a kept vector
        nxt = int(np.argmin(masked))  # farthest from the kept set
        kept.append(nxt)
        nearest = np.maximum(nearest, mat @ mat[nxt])
    return kept


def safe_plate(plate: str) -> str:
    """Filesystem-safe folder name for a plate (keeps A-Z0-9 and ``-``/``_``)."""
    return _SAFE.sub("_", (plate or "").strip()).strip("._") or "UNKNOWN"


class VehicleGalleryStore:
    def __init__(
        self,
        base_dir: str,
        model_tag: str,
        max_refs: int = 10,
        *,
        require_verified_admission: bool = False,
    ):
        self._root = os.path.join(base_dir or "vehicle_images", "gallery")
        self._quarantine_root = os.path.join(
            base_dir or "vehicle_images",
            "gallery_quarantine",
        )
        self._model_tag = model_tag or "unknown"
        self._max_refs = max(1, int(max_refs))
        self._require_verified_admission = bool(require_verified_admission)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Paths / meta
    # ------------------------------------------------------------------ #
    def _plate_dir(self, plate: str) -> str:
        return os.path.join(self._root, safe_plate(plate))

    def has(self, plate: str) -> bool:
        meta = self._read_meta(plate)
        return bool(
            meta
            and self._plate_key(meta.get("plate")) == self._plate_key(plate)
        )

    def all_plates(self) -> List[str]:
        """Every plate with a gallery on disk, INCLUDING cars with no open session.

        The rest of the system reaches the gallery only through a live session:
        ``_restore_vehicle_galleries`` hydrates the plates in ``parking_sessions``
        and nothing else. That makes the gallery unreachable for exactly the car
        that needs it most — one parked inside with no session, whose references
        are sitting on disk while ReID reports it has never seen the car.
        (Measured 2026-07-30: BHD-9990 parked in B13, OCR reading ``BHD`` on 6 of 6
        frames, and every read discarded as "confirms none of ReID's top-5"
        because its gallery was never loaded.)

        Reads the plate from each folder's ``meta.json`` rather than trusting the
        directory name, which is ``safe_plate()``-mangled and not reversible.
        Returns [] on any filesystem error — a recovery path must never be able
        to take the engine down.
        """
        if not os.path.isdir(self._root):
            return []
        out: List[str] = []
        try:
            for entry in os.scandir(self._root):
                if not entry.is_dir():
                    continue
                meta = None
                try:
                    with open(os.path.join(entry.path, "meta.json"), encoding="utf-8") as fh:
                        meta = json.load(fh)
                except (OSError, ValueError):
                    continue
                plate = (meta or {}).get("plate")
                if plate:
                    out.append(str(plate))
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("[gallery] all_plates failed under %s: %r", self._root, exc)
            return []
        return out

    def _read_meta(self, plate: str) -> Optional[dict]:
        path = os.path.join(self._plate_dir(plate), _META)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def _write_meta(self, plate: str, meta: dict) -> bool:
        d = self._plate_dir(plate)
        try:
            os.makedirs(d, exist_ok=True)
            tmp = os.path.join(d, _META + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, os.path.join(d, _META))
            self._fsync_directory(d)
            return True
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("[gallery] meta write failed for %s: %r", plate, exc)
            try:
                os.remove(os.path.join(d, _META + ".tmp"))
            except OSError:
                pass
            return False

    @staticmethod
    def _fsync_directory(directory: str) -> None:
        try:
            descriptor = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @staticmethod
    def _plate_key(value: object) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())

    @classmethod
    def _strict_admission_valid(
        cls,
        plate: str,
        evidence_id: Optional[str],
        admission: Optional[dict],
        reference_camera: Optional[str] = None,
    ) -> bool:
        if not evidence_id or not isinstance(admission, dict):
            return False
        if (
            admission.get("verified") is not True
            or admission.get("policy") != "entry_v2_parked_v1"
        ):
            return False
        entry_proof = admission.get("entry_proof")
        if not isinstance(entry_proof, dict):
            return False
        decision_id = str(admission.get("entry_decision_id") or "")
        slot_id = str(admission.get("parked_slot") or "")
        camera_id = str(admission.get("parked_camera") or "")
        if (
            not decision_id
            or not slot_id
            or not camera_id
            or evidence_id != f"parked:{decision_id}:{slot_id}:{camera_id}"
        ):
            return False
        if (
            reference_camera is not None
            and cls._plate_key(reference_camera) != cls._plate_key(camera_id)
        ):
            return False
        if (
            entry_proof.get("policy") != "entry_v2_gallery_v1"
            or entry_proof.get("decision_id") != decision_id
            or entry_proof.get("requires_parked_ocr") is not True
            or cls._plate_key(entry_proof.get("canonical_plate"))
            != cls._plate_key(plate)
        ):
            return False
        role = str(entry_proof.get("crossing_role") or "")
        source = str(entry_proof.get("ocr_source") or "")
        crossing_camera = cls._plate_key(entry_proof.get("crossing_camera_id"))
        # The CAMERA must still be an approved one for its role, but the plate
        # no longer comes from that camera: it is decided by consensus across
        # ANPR, HikCentral and our own OCR, so every authorized proof reports
        # "consensus" whichever camera produced the crop.
        if (role, crossing_camera) not in {
            ("primary", "CAM23"),
            ("fallback", "CAM03"),
        } or source != "consensus":
            return False
        entry_numeric_floors = (
            (entry_proof.get("reid_score"), 0.85, 1.0),
            (entry_proof.get("reid_row_margin"), 0.12, 2.0),
            (entry_proof.get("reid_column_margin"), 0.12, 2.0),
            (entry_proof.get("ocr_confidence"), 0.90, 1.0),
        )
        for value, minimum, maximum in entry_numeric_floors:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(number) or not minimum <= number <= maximum:
                return False
        if cls._plate_key(entry_proof.get("ocr_text")) != cls._plate_key(plate):
            return False
        evidence_ids = entry_proof.get("ocr_evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            return False
        authorization_path = entry_proof.get("authorization_path")
        entry_reason = admission.get("entry_reason")
        if authorization_path == "exact_plate":
            try:
                reported_confidence = float(entry_proof.get("reported_confidence"))
            except (TypeError, ValueError):
                return False
            if (
                entry_reason != "reid_and_plate_consensus"
                or entry_proof.get("corrected") is not False
                or not math.isfinite(reported_confidence)
                or reported_confidence < 0.90
                or reported_confidence > 1.0
                or cls._plate_key(entry_proof.get("reported_plate"))
                != cls._plate_key(plate)
            ):
                return False
        elif authorization_path == "corrected_plate":
            if (
                entry_reason != "reid_and_independent_ocr_correction"
                or entry_proof.get("corrected") is not True
                or len(set(str(item) for item in evidence_ids)) < 2
            ):
                return False
        else:
            return False
        numeric_ranges = (
            (admission.get("parked_ocr_confidence"), 0.90, 1.0),
            (admission.get("parked_reid_score"), 0.70, 1.0),
            (admission.get("parked_reid_margin"), 0.15, 2.0),
            (admission.get("parked_view_quality"), 0.90, 1.0),
            (admission.get("parked_neighbour_clearance"), 0.90, 1.0),
        )
        for value, minimum, maximum in numeric_ranges:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(number) or not minimum <= number <= maximum:
                return False
        rank = admission.get("parked_reid_rank")
        return bool(
            not isinstance(rank, bool)
            and isinstance(rank, int)
            and rank == 1
            and cls._plate_key(admission.get("parked_ocr_text"))
            == cls._plate_key(plate)
        )

    def _is_verified_ref(self, plate: str, ref: dict) -> bool:
        return self._strict_admission_valid(
            plate,
            ref.get("evidence_id"),
            ref.get("admission"),
            ref.get("camera"),
        )

    def _active_refs(self, meta: dict) -> List[dict]:
        refs = list(meta.get("refs", []))
        if not self._require_verified_admission:
            return refs
        plate = str(meta.get("plate") or "")
        return [ref for ref in refs if self._is_verified_ref(plate, ref)]

    def _crop_integrity_ok(self, directory: str, ref: dict) -> bool:
        """Strict-mode backstop against partial or corrupted crop files."""
        if not self._require_verified_admission:
            return True
        filename = ref.get("crop")
        expected = ref.get("sha256")
        if not filename or not expected:
            return False
        try:
            digest = hashlib.sha256()
            with open(os.path.join(directory, filename), "rb") as fh:
                for chunk in iter(lambda: fh.read(64 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest() == expected
        except OSError:
            return False

    def _vector_integrity_ok(self, directory: str, ref: dict) -> bool:
        """Strict-mode backstop against a mismatched/corrupted embedding."""
        if not self._require_verified_admission:
            return True
        filename = ref.get("vec")
        expected = ref.get("vector_sha256")
        if not filename or not expected:
            return False
        try:
            digest = hashlib.sha256()
            with open(os.path.join(directory, filename), "rb") as fh:
                for chunk in iter(lambda: fh.read(64 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest() == expected
        except OSError:
            return False

    def _prepare_quarantine_records(
        self,
        plate: str,
        refs: List[dict],
    ) -> List[dict]:
        """Describe unverified files before moving any of them.

        The returned records are written to ``meta.json`` with the matching
        ``refs`` removed before :meth:`_move_quarantine_records` runs.  A
        process or power failure can therefore leave either source or
        destination files behind, but never an unverified, matchable record or
        an unrecorded quarantine move.  The source/destination names make the
        operation restart-recoverable.
        """
        if not refs:
            return []
        source_dir = self._plate_dir(plate)
        quarantine_dir = os.path.join(self._quarantine_root, safe_plate(plate))
        prepared: List[dict] = []
        for item in refs:
            record = dict(item)
            quarantine_id = uuid.uuid4().hex
            record["quarantine_id"] = quarantine_id
            record.setdefault("quarantine_reason", "unverified")
            record["quarantine_dir"] = os.path.relpath(
                quarantine_dir,
                os.path.dirname(self._root),
            )
            record["quarantine_state"] = "pending"
            record["quarantine_started_at"] = (
                datetime.now().astimezone().isoformat()
            )
            source_files: dict[str, str] = {}
            for key in ("crop", "vec"):
                filename = item.get(key)
                if not filename:
                    continue
                filename = str(filename)
                source_files[key] = filename
                source = os.path.join(source_dir, filename)
                destination = os.path.join(quarantine_dir, filename)
                # A missing source plus an existing same-name destination can
                # be the residue of the pre-transaction implementation.  Keep
                # that destination so the new transaction can adopt it.
                if os.path.isfile(source) and os.path.exists(destination):
                    stem, extension = os.path.splitext(filename)
                    record[key] = f"{stem}_{quarantine_id[:6]}{extension}"
                else:
                    record[key] = filename
            record["quarantine_source_files"] = source_files
            prepared.append(record)
        return prepared

    def _move_quarantine_records(
        self,
        plate: str,
        records: List[dict],
    ) -> List[dict]:
        """Complete or resume durable quarantine records.

        Each file is moved independently and idempotently.  A destination that
        exists while its source is absent is treated as an already-completed
        move from an interrupted prior attempt.  Incomplete records stay
        ``pending`` with their last error in metadata and are retried by the
        next strict save.
        """
        if not records:
            return []
        source_dir = self._plate_dir(plate)
        quarantine_dir = os.path.join(self._quarantine_root, safe_plate(plate))
        directory_error: Optional[str] = None
        try:
            os.makedirs(quarantine_dir, exist_ok=True)
        except OSError as exc:
            directory_error = repr(exc)

        reconciled: List[dict] = []
        for item in records:
            record = dict(item)
            previous_error = str(record.get("quarantine_last_error") or "")
            errors: List[str] = []
            source_files = record.get("quarantine_source_files")
            if not isinstance(source_files, dict):
                source_files = {}

            if directory_error is not None:
                errors.append(f"quarantine directory: {directory_error}")
            else:
                for key in ("crop", "vec"):
                    source_name = source_files.get(key)
                    destination_name = record.get(key)
                    if not source_name and not destination_name:
                        continue
                    if not source_name or not destination_name:
                        errors.append(f"{key}: incomplete file metadata")
                        continue
                    source = os.path.join(source_dir, str(source_name))
                    destination = os.path.join(
                        quarantine_dir,
                        str(destination_name),
                    )
                    source_exists = os.path.isfile(source)
                    destination_exists = os.path.isfile(destination)
                    if destination_exists and not source_exists:
                        continue
                    if not source_exists and not destination_exists:
                        errors.append(
                            f"{key}: source and destination are both missing"
                        )
                        continue
                    if source_exists and destination_exists:
                        errors.append(f"{key}: destination collision")
                        continue
                    try:
                        os.replace(source, destination)
                    except OSError as exc:
                        errors.append(f"{key}: {exc!r}")

            if errors:
                error_text = "; ".join(errors)
                record["quarantine_state"] = "pending"
                record["quarantine_move_failed"] = True
                record["quarantine_last_error"] = error_text
                record["quarantine_attempts"] = int(
                    record.get("quarantine_attempts") or 0
                ) + 1
                # Preserve diagnostics in metadata but do not emit the same
                # warning on every parked observation/retry.
                if error_text != previous_error:
                    logger.warning(
                        "[gallery] quarantine incomplete for %s: %s",
                        plate,
                        error_text,
                    )
            else:
                record["quarantine_state"] = "complete"
                record["quarantined_at"] = (
                    datetime.now().astimezone().isoformat()
                )
                record.pop("quarantine_move_failed", None)
                record.pop("quarantine_last_error", None)
                record.pop("quarantine_attempts", None)
            reconciled.append(record)
        self._fsync_directory(source_dir)
        if directory_error is None:
            self._fsync_directory(quarantine_dir)
        return reconciled

    @staticmethod
    def _replace_quarantine_records(
        all_records: List[dict],
        replacements: List[dict],
    ) -> List[dict]:
        """Replace transaction records without disturbing older audit rows."""
        by_id = {
            str(item.get("quarantine_id")): item
            for item in replacements
            if item.get("quarantine_id")
        }
        return [
            by_id.get(str(item.get("quarantine_id")), item)
            for item in all_records
        ]

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
        evidence_id: Optional[str] = None,
        admission: Optional[dict] = None,
    ) -> Optional[str]:
        """Persist one crop/vector reference and its admission proof.

        File replacements are atomic and metadata publication is fail-closed:
        an interruption may leave an unreferenced file, never a matchable record
        whose required files or verified proof were not published.

        Returns the crop filename on success, else None. Eviction keeps the
        highest-``quality`` refs when the folder exceeds ``max_refs``.

        ``evidence_id`` is an optional idempotency key. Replaying the same key
        and bytes returns the existing filename; reusing it for different bytes
        fails closed. ``admission`` stores the scalar proof that authorized a
        strict gallery write, making every learned image auditable.

        ``gate_only`` is retained for legacy non-strict stores and excludes a
        reference from :meth:`load_vectors` / :meth:`load_crops`. Production
        strict persistence never writes gate/entry images; it admits only the
        independently verified parked crop.
        """
        if self._require_verified_admission and not self._strict_admission_valid(
            plate,
            evidence_id,
            admission,
            camera_id,
        ):
            logger.warning(
                "[gallery] refused unverified durable reference for %s", plate
            )
            return None
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0 or vector is None:
            return None
        try:
            array = np.asarray(vector, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(array))
            if array.size == 0 or norm <= 0.0 or not np.isfinite(array).all():
                return None
            array = array / norm
            encoded_ok, encoded = cv2.imencode(".jpg", crop_bgr)
            if not encoded_ok:
                return None
            jpeg_bytes = encoded.tobytes()
            vector_buffer = io.BytesIO()
            np.save(vector_buffer, array, allow_pickle=False)
            vector_bytes = vector_buffer.getvalue()
        except Exception:
            return None

        now = timestamp or datetime.now()
        d = self._plate_dir(plate)
        stem = f"crop_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        crop_name = stem + ".jpg"
        vec_name = stem + ".npy"
        crop_path = os.path.join(d, crop_name)
        vec_path = os.path.join(d, vec_name)
        crop_tmp = crop_path + ".tmp"
        vec_tmp = vec_path + ".tmp"
        digest = hashlib.sha256(jpeg_bytes).hexdigest()
        vector_digest = hashlib.sha256(vector_bytes).hexdigest()

        with self._lock:
            meta = self._read_meta(plate) or {
                "plate": plate,
                "created_at": now.isoformat(),
                "refs": [],
            }
            if meta.get("plate") not in (None, plate):
                logger.error(
                    "[gallery] plate-folder collision: requested=%s stored=%s",
                    plate,
                    meta.get("plate"),
                )
                return None

            # Older metadata carried one folder-level tag. Preserve that tag on
            # every existing ref before writing a vector from a newer model; otherwise
            # the folder-level overwrite would mislabel old vectors as current and a
            # restart could compare embeddings from incompatible model spaces.
            previous_model_tag = str(meta.get("model_tag") or "unknown")
            for existing_ref in meta.get("refs", []):
                existing_ref.setdefault("model_tag", previous_model_tag)

            if self._require_verified_admission:
                # Existing galleries pre-date strict evidence metadata. Keep
                # their files for operator review. First persist a fail-closed
                # transaction that removes their records from the matchable set,
                # then move/reconcile the files and persist the completed audit
                # state before adding a verified ref.
                verified_refs = []
                quarantined_refs = list(meta.get("quarantined_refs", []))
                newly_unverified = []
                for item in meta.get("refs", []):
                    if self._is_verified_ref(plate, item):
                        verified_refs.append(item)
                    else:
                        newly_unverified.append(item)
                meta["refs"] = verified_refs
                if newly_unverified:
                    quarantined_refs.extend(
                        self._prepare_quarantine_records(
                            plate,
                            newly_unverified,
                        )
                    )
                if quarantined_refs:
                    meta["quarantined_refs"] = quarantined_refs
                if newly_unverified and not self._write_meta(plate, meta):
                    # Nothing has moved until the pending audit transaction is
                    # durable. The strict loader still excludes the legacy refs
                    # in the unchanged metadata, so this remains fail-closed.
                    return None

                pending_quarantine = [
                    item
                    for item in quarantined_refs
                    if item.get("quarantine_state") == "pending"
                ]
                if pending_quarantine:
                    reconciled = self._move_quarantine_records(
                        plate,
                        pending_quarantine,
                    )
                    meta["quarantined_refs"] = self._replace_quarantine_records(
                        quarantined_refs,
                        reconciled,
                    )
                    if not self._write_meta(plate, meta):
                        # The already-durable pending records describe the
                        # exact source and destination names, allowing the next
                        # process to finish reconciliation without ambiguity.
                        return None

            if evidence_id:
                existing = next(
                    (
                        item
                        for item in meta.get("refs", [])
                        if item.get("evidence_id") == evidence_id
                    ),
                    None,
                )
                if existing is not None:
                    if (
                        existing.get("sha256") != digest
                        or existing.get("vector_sha256") != vector_digest
                    ):
                        logger.error(
                            "[gallery] evidence id reused with different reference: %s",
                            evidence_id,
                        )
                        return None
                    existing_crop = existing.get("crop")
                    existing_vec = existing.get("vec")
                    if (
                        existing_crop
                        and existing_vec
                        and os.path.isfile(os.path.join(d, existing_crop))
                        and os.path.isfile(os.path.join(d, existing_vec))
                        and self._crop_integrity_ok(d, existing)
                        and self._vector_integrity_ok(d, existing)
                    ):
                        return str(existing_crop)
                    meta["refs"].remove(existing)
                    for filename in (existing_crop, existing_vec):
                        if filename:
                            try:
                                os.remove(os.path.join(d, filename))
                            except OSError:
                                pass

            try:
                os.makedirs(d, exist_ok=True)
                with open(crop_tmp, "wb") as fh:
                    fh.write(jpeg_bytes)
                    fh.flush()
                    os.fsync(fh.fileno())
                with open(vec_tmp, "wb") as fh:
                    fh.write(vector_bytes)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(crop_tmp, crop_path)
                os.replace(vec_tmp, vec_path)
                self._fsync_directory(d)
            except OSError as exc:
                logger.warning("[gallery] save_ref failed for %s: %r", plate, exc)
                for path in (crop_tmp, vec_tmp, crop_path, vec_path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                return None

            meta["model_tag"] = self._model_tag
            meta["updated_at"] = now.isoformat()
            if camera_id:
                meta["last_camera"] = camera_id
            ref = {
                "crop": crop_name,
                "vec": vec_name,
                "model_tag": self._model_tag,
                "quality": float(quality),
                "camera": camera_id,
                "ts": now.isoformat(),
                "sha256": digest,
                "vector_sha256": vector_digest,
            }
            if evidence_id:
                ref["evidence_id"] = str(evidence_id)
            if admission is not None:
                ref["admission"] = dict(admission)
            if gate_only:
                ref["gate"] = True
            meta.setdefault("refs", []).append(ref)
            removed = self._prune_meta_inplace(meta)
            if not self._write_meta(plate, meta):
                for path in (crop_path, vec_path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                return None
            for item in removed:
                for key in ("crop", "vec"):
                    filename = item.get(key)
                    if filename:
                        try:
                            os.remove(os.path.join(d, filename))
                        except OSError:
                            pass
            return crop_name

    def _prune_meta_inplace(self, meta: dict) -> List[dict]:
        """Prune to ``max_refs``, keeping a spread across cameras/viewpoints.

        Keeping the globally-sharpest refs collapses the gallery onto one
        camera's viewpoint (a parked car in front of one camera generates the
        sharpest, most numerous crops), which makes a returning car's query from
        a different camera/angle unmatchable. Instead we round-robin across
        cameras — each camera is a distinct position — taking the highest-quality
        crop from each in turn until the cap is filled. Legacy gate-only refs are
        kept preferentially. Within a single camera this still falls back to
        quality order.
        """
        refs = meta.get("refs", [])
        if len(refs) <= self._max_refs:
            return []

        gate_refs = [r for r in refs if r.get("gate")]
        normal_refs = [r for r in refs if not r.get("gate")]

        by_cam: dict = {}
        for r in normal_refs:
            by_cam.setdefault(r.get("camera", "") or "", []).append(r)
        for cam_refs in by_cam.values():
            cam_refs.sort(key=lambda r: r.get("quality", 0.0), reverse=True)

        keep: list = list(gate_refs[: self._max_refs])
        # Round-robin across cameras so every viewpoint is represented before
        # any single camera contributes a second crop.
        queues = [list(v) for v in by_cam.values()]
        while len(keep) < self._max_refs and any(queues):
            for q in queues:
                if q:
                    keep.append(q.pop(0))
                    if len(keep) >= self._max_refs:
                        break

        keep_ids = {id(r) for r in keep}
        removed = [r for r in refs if id(r) not in keep_ids]
        meta["refs"] = keep
        return removed

    def stamp_exit(self, plate: str, timestamp: Optional[datetime] = None) -> None:
        """Record the car's exit time so the TTL GC ages the folder from here."""
        with self._lock:
            meta = self._read_meta(plate)
            if meta is None:
                return
            meta["last_exit"] = (timestamp or datetime.now()).isoformat()
            self._write_meta(plate, meta)

    def delete(self, plate: str) -> None:
        with self._lock:
            shutil.rmtree(self._plate_dir(plate), ignore_errors=True)
            self._fsync_directory(self._root)

    def rename(self, old_plate: str, new_plate: str) -> bool:
        """Move a car's gallery to the plate it should have been filed under.

        PMS-AI corrects a stay whose ENTRY plate was misread. Without this the
        crops stay filed under the misread and VA keeps re-minting it, so the
        correction survives in PMS and dies here.

        Three cases, and none of them may lose a reference:

        * Nothing on disk for the misread — nothing to move, returns False.
        * The target has no gallery — one ``os.replace`` of the folder. Atomic,
          so a crash leaves the refs under exactly one of the two names.
        * The target ALREADY has a gallery, because the car has been here
          before. The two sets are merged and pruned to ``max_refs`` rather than
          either being dropped: the misread folder holds this stay's crops, the
          target folder holds previous stays', and both are the same car.

        Filenames carry a per-crop token, so merging two folders cannot collide.
        Never raises — a correction must not fail because a disk did.
        """
        with self._lock:
            src = self._plate_dir(old_plate)
            dst = self._plate_dir(new_plate)
            if not os.path.isdir(src):
                return False

            meta = self._read_meta(old_plate) or {}
            if os.path.abspath(src) == os.path.abspath(dst):
                # Different strings, same folder (safe_plate collapses both).
                meta["plate"] = new_plate
                return self._write_meta(new_plate, meta)

            try:
                if not os.path.exists(dst):
                    os.replace(src, dst)
                    meta["plate"] = new_plate
                    ok = self._write_meta(new_plate, meta)
                    self._fsync_directory(self._root)
                    logger.info(
                        "[gallery] renamed %s -> %s (%d refs)",
                        old_plate, new_plate, len(meta.get("refs", [])),
                    )
                    return ok
                return self._merge_into(src, dst, meta, old_plate, new_plate)
            except OSError as exc:
                logger.warning(
                    "[gallery] rename %s -> %s failed: %r",
                    old_plate, new_plate, exc,
                )
                return False

    def _merge_into(
        self, src: str, dst: str, meta: dict, old_plate: str, new_plate: str
    ) -> bool:
        """Fold the misread plate's refs into an existing gallery, then prune.

        A ref whose files cannot be moved is dropped from the merge rather than
        carried into the target's meta — a record pointing at a file that is not
        there reads as a corrupt gallery, which is worse than one fewer crop.
        """
        target = self._read_meta(new_plate) or {"plate": new_plate}
        moved = []
        for ref in meta.get("refs", []):
            names = [ref.get(key) for key in ("crop", "vec") if ref.get(key)]
            try:
                for name in names:
                    os.replace(os.path.join(src, name), os.path.join(dst, name))
            except OSError as exc:
                logger.warning(
                    "[gallery] merge %s -> %s dropped a ref: %r",
                    old_plate, new_plate, exc,
                )
                continue
            moved.append(ref)

        target["plate"] = new_plate
        target.setdefault("refs", []).extend(moved)
        removed = self._prune_meta_inplace(target)
        ok = self._write_meta(new_plate, target)
        if ok:
            for item in removed:
                for key in ("crop", "vec"):
                    name = item.get(key)
                    if name:
                        try:
                            os.remove(os.path.join(dst, name))
                        except OSError:
                            pass
            shutil.rmtree(src, ignore_errors=True)
            self._fsync_directory(self._root)
        logger.info(
            "[gallery] merged %s into existing %s: %d refs moved, %d pruned",
            old_plate, new_plate, len(moved), len(removed),
        )
        return ok

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
        quarantine_root = os.path.join(
            base_dir or "vehicle_images",
            "gallery_quarantine",
        )
        shutil.rmtree(quarantine_root, ignore_errors=True)
        return removed

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #
    def load_vectors(
        self, plate: str, *, current_tag_only: bool = False
    ) -> Tuple[List[np.ndarray], Optional[str], List[str]]:
        """Return (vectors, stored_model_tag, source_cameras). Empty when
        absent/unreadable.

        Vectors are returned primary-first (highest quality first) so the caller
        can use ``vectors[0]`` as the session's primary feature. ``source_cameras``
        is index-aligned with ``vectors`` (the camera that captured each ref, for
        match-time trust weighting). Legacy gate-only refs are excluded and are
        never used for matching (see :meth:`save_ref`).

        ``current_tag_only`` drops references embedded under a DIFFERENT model
        contract than the one now running. A folder can hold a mix — BHD-9990 in
        production carries 4 refs tagged ``openvino:PS_matcher V2`` alongside 16
        tagged ``openvino:PS_matcher V2:13683459c498b2dfc01e`` — and a cosine
        distance between vectors from two different models means nothing.

        The default is False because the existing caller
        (``build_session_from_gallery``) inspects the returned tag itself and
        re-embeds every crop when it mismatches. A caller that scores vectors
        directly, without that re-embed, must pass True or it will silently
        compare across contracts.
        """
        meta = self._read_meta(plate)
        if not meta or self._plate_key(meta.get("plate")) != self._plate_key(plate):
            return [], None, []
        d = self._plate_dir(plate)
        refs = sorted(
            (r for r in self._active_refs(meta) if not r.get("gate")),
            key=lambda r: r.get("quality", 0.0),
            reverse=True,
        )
        vectors: List[np.ndarray] = []
        cameras: List[str] = []
        model_tags: List[str] = []
        for r in refs:
            fn = r.get("vec")
            if (
                not fn
                or not self._crop_integrity_ok(d, r)
                or not self._vector_integrity_ok(d, r)
            ):
                continue
            if current_tag_only:
                ref_tag = str(r.get("model_tag") or meta.get("model_tag") or "unknown")
                if ref_tag != self._model_tag:
                    continue
            try:
                vector = (
                    np.load(os.path.join(d, fn), allow_pickle=False)
                    .astype(np.float32)
                    .reshape(-1)
                )
            except (OSError, ValueError):
                continue
            norm = float(np.linalg.norm(vector))
            if vector.size == 0 or norm <= 0.0 or not np.isfinite(vector).all():
                continue
            vectors.append(vector / norm)
            cameras.append(r.get("camera", "") or "")
            model_tags.append(str(r.get("model_tag") or meta.get("model_tag") or "unknown"))
        unique_tags = set(model_tags)
        if len(unique_tags) == 1:
            stored_tag: Optional[str] = next(iter(unique_tags))
        elif unique_tags:
            stored_tag = "mixed:" + ",".join(sorted(unique_tags))
        else:
            stored_tag = meta.get("model_tag")
        return vectors, stored_tag, cameras

    def load_crops(self, plate: str) -> Tuple[List[np.ndarray], List[str]]:
        """Load the reference crop images (highest quality first) for re-embedding
        when the stored ``model_tag`` no longer matches the running model, with an
        index-aligned list of their source cameras. Gate-only refs are excluded,
        mirroring :meth:`load_vectors`."""
        meta = self._read_meta(plate)
        if not meta or self._plate_key(meta.get("plate")) != self._plate_key(plate):
            return [], []
        d = self._plate_dir(plate)
        refs = sorted(
            (r for r in self._active_refs(meta) if not r.get("gate")),
            key=lambda r: r.get("quality", 0.0),
            reverse=True,
        )
        crops: List[np.ndarray] = []
        cameras: List[str] = []
        for r in refs:
            fn = r.get("crop")
            if (
                not fn
                or not self._crop_integrity_ok(d, r)
                or not self._vector_integrity_ok(d, r)
            ):
                continue
            img = cv2.imread(os.path.join(d, fn))
            if img is not None and img.size > 0:
                crops.append(img)
                cameras.append(r.get("camera", "") or "")
        return crops, cameras

    def ref_count(self, plate: str) -> int:
        meta = self._read_meta(plate)
        if not meta or self._plate_key(meta.get("plate")) != self._plate_key(plate):
            return 0
        return len(self._active_refs(meta))

    # ------------------------------------------------------------------ #
    # Garbage collection
    # ------------------------------------------------------------------ #
    def gc(self, retention_days: float) -> int:
        """Delete plate folders whose newest activity is older than the TTL.

        Newest activity = max of the folder's file mtimes (covers both the last
        ref write and the ``stamp_exit`` meta rewrite). Returns folders removed.
        """
        with self._lock:
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
                self._fsync_directory(self._root)
        if removed:
            logger.info("[gallery] GC removed %d folder(s) idle > %.1fd", removed, retention_days)
        return removed
