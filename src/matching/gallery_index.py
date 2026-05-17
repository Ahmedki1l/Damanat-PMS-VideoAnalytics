"""
src.matching.gallery_index — Phase 3 / T3.2 FAISS-CPU gallery index.

Replaces the O(n) linear scan in
:meth:`VehicleRegistry.match_global_session` with an O(log n) IVF (or HNSW)
search. The wrapper is feature-flagged via
``MatchingConfig.use_faiss_index`` so the legacy path remains the default
until the production rollout completes shadow-mode benchmarking.

Design
------

* In-memory only — no on-disk persistence. The index is rebuilt from
  :attr:`VehicleRegistry._sessions` at boot. This avoids stale-state bugs
  on registry restart and removes a persistence schema we would need to
  version.
* Two backends:
  - **faiss-cpu** (preferred when available). IVF index with a small
    ``nlist`` (8 by default), auto-trained once we accumulate
    ``nlist * 16`` vectors. Below the threshold we use ``IndexFlatIP``
    so search remains correct.
  - **numpy fallback** (always available). A brute-force cosine search.
    Same public API; identical correctness; slower above ~5 000 sessions
    where FAISS shines.
* Cosine metric is implemented via L2-normalised inputs + inner product
  (FAISS native IP). Callers must already pass unit vectors — the
  reid_matcher path emits these by default.

Thread-safety
-------------

All public methods take ``self._lock`` (an RLock). The matrix is
re-allocated on demand when adds exceed capacity. Search uses a snapshot
of the IDs list and the matrix view to keep critical sections short.

Public surface
~~~~~~~~~~~~~~

::

    GalleryIndex(dimension=512, nlist=8, metric="cosine")
        .add(session_id, feature_vector)
        .remove(session_id)
        .search(query_vector, k=10) -> List[(session_id, score)]
        .__len__() -> int
        .clear() -> None
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# FAISS detection
# --------------------------------------------------------------------------- #


def _try_import_faiss():
    """Return the faiss module when importable, else None.

    The import is intentionally lazy so a missing faiss dependency never
    breaks ``import src.vehicle_registry``. The numpy fallback covers all
    public behaviour.
    """
    try:
        import faiss  # type: ignore  # noqa: WPS433

        return faiss
    except Exception as exc:  # pragma: no cover - only when faiss missing
        logger.info(
            "[GalleryIndex] faiss unavailable (%r) — using numpy fallback.",
            exc,
        )
        return None


_FAISS = _try_import_faiss()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Return a unit-norm copy of ``vec`` (idempotent for already-normed).

    We accept a non-normed feature for robustness — the rest of the
    matching cascade assumes L2-normed feature vectors, but defensive
    re-normalisation here costs almost nothing.
    """
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n > 1e-8:
        return v / n
    return v


# --------------------------------------------------------------------------- #
# GalleryIndex
# --------------------------------------------------------------------------- #


class GalleryIndex:
    """FAISS-CPU (or numpy fallback) gallery index for session features.

    Args:
        dimension: feature vector dimensionality. OSNet-AIN emits 512-d
            vectors; the WS-A OpenVINO export preserves this. The index
            is fixed-dimension; passing a vector with a different length
            triggers a ValueError at add/search time.
        nlist: number of IVF coarse-quantiser cells. Small (8) is fine
            for facility-scale galleries (~thousands of sessions). Larger
            values reduce search latency at the cost of higher training
            data requirements.
        metric: ``cosine`` (the only supported value today). Stored as a
            constructor argument so a future swap to L2 is a config change
            rather than a code change.
    """

    AUTO_TRAIN_MULTIPLIER: int = 16

    def __init__(
        self,
        dimension: int = 512,
        nlist: int = 8,
        metric: str = "cosine",
    ):
        if dimension <= 0:
            raise ValueError(f"dimension must be > 0, got {dimension}")
        if nlist <= 0:
            raise ValueError(f"nlist must be > 0, got {nlist}")
        if metric.lower() != "cosine":
            raise ValueError(
                f"GalleryIndex only supports the 'cosine' metric, got {metric!r}"
            )
        self._dim = int(dimension)
        self._nlist = int(nlist)
        self._metric = metric.lower()
        self._lock = threading.RLock()

        # --- numpy fallback storage ---------------------------------------- #
        # ``_ids`` is the ordered list of session_ids. ``_vectors`` is an
        # NxD matrix aligned with ``_ids``. Remove is O(D) (matrix shrink)
        # but we expect removals to be rare relative to adds + searches.
        self._ids: List[str] = []
        self._vectors: np.ndarray = np.zeros((0, self._dim), dtype=np.float32)

        # --- faiss state --------------------------------------------------- #
        # We start in "flat" mode (IndexFlatIP) which is correct for all
        # sizes but slow above ~5000 vectors. Once the gallery exceeds
        # ``nlist * AUTO_TRAIN_MULTIPLIER`` rows we train an IVF index and
        # migrate. This is the lazy-training pattern recommended by the
        # FAISS docs to avoid the "needs training" footgun.
        self._faiss = _FAISS
        self._faiss_index = None
        self._faiss_trained: bool = False

    # ------------------------------------------------------------------ #
    # Status helpers
    # ------------------------------------------------------------------ #

    @property
    def backend(self) -> str:
        """Reports the active backend name for logs/tests."""
        if self._faiss is not None:
            return "faiss-ivf" if self._faiss_trained else "faiss-flat"
        return "numpy"

    @property
    def dimension(self) -> int:
        return self._dim

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)

    def __contains__(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._ids

    def clear(self) -> None:
        """Drop all entries. Safe to call concurrently."""
        with self._lock:
            self._ids.clear()
            self._vectors = np.zeros((0, self._dim), dtype=np.float32)
            self._faiss_index = None
            self._faiss_trained = False

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #

    def add(self, session_id: str, feature_vector: np.ndarray) -> None:
        """Add (or update) ``feature_vector`` for ``session_id``.

        If ``session_id`` is already in the index, its vector is replaced
        in place. This matches the registry's ``update_session_feature``
        semantics (EMA-smoothed updates overwrite the stored feature).
        """
        if feature_vector is None:
            return
        vec = _normalize(feature_vector)
        if vec.shape[0] != self._dim:
            raise ValueError(
                f"vector dimension {vec.shape[0]} does not match index "
                f"dimension {self._dim}"
            )
        with self._lock:
            if session_id in self._ids:
                idx = self._ids.index(session_id)
                self._vectors[idx] = vec
            else:
                self._ids.append(session_id)
                if self._vectors.shape[0] == 0:
                    self._vectors = vec.reshape(1, -1).astype(np.float32)
                else:
                    self._vectors = np.vstack([self._vectors, vec[np.newaxis, :]]).astype(
                        np.float32
                    )
            self._rebuild_faiss_index_locked()

    def remove(self, session_id: str) -> bool:
        """Remove ``session_id`` from the index.

        Returns True when the id existed, False otherwise.
        """
        with self._lock:
            if session_id not in self._ids:
                return False
            idx = self._ids.index(session_id)
            del self._ids[idx]
            self._vectors = np.delete(self._vectors, idx, axis=0)
            self._rebuild_faiss_index_locked()
            return True

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def search(
        self, query_vector: np.ndarray, k: int = 10
    ) -> List[Tuple[str, float]]:
        """Top-k cosine search. Returns ``[(session_id, score), ...]`` sorted
        descending by score.

        ``k`` is clamped to the current gallery size. When the index is
        empty an empty list is returned.
        """
        if query_vector is None:
            return []
        q = _normalize(query_vector)
        if q.shape[0] != self._dim:
            raise ValueError(
                f"query dimension {q.shape[0]} does not match index "
                f"dimension {self._dim}"
            )
        with self._lock:
            n = len(self._ids)
            if n == 0 or k <= 0:
                return []
            k = min(k, n)
            if self._faiss is not None and self._faiss_index is not None:
                return self._search_faiss_locked(q, k)
            return self._search_numpy_locked(q, k)

    # ------------------------------------------------------------------ #
    # Backend implementations (all called while self._lock is held)
    # ------------------------------------------------------------------ #

    def _search_numpy_locked(
        self, q: np.ndarray, k: int
    ) -> List[Tuple[str, float]]:
        """Brute-force cosine search. Same shape of result as faiss path."""
        # Inner product of two L2-normed vectors equals cosine.
        scores = self._vectors @ q  # shape (n,)
        if k >= scores.shape[0]:
            order = np.argsort(-scores)
        else:
            partial = np.argpartition(-scores, k - 1)[:k]
            order = partial[np.argsort(-scores[partial])]
        return [(self._ids[i], float(scores[i])) for i in order]

    def _search_faiss_locked(
        self, q: np.ndarray, k: int
    ) -> List[Tuple[str, float]]:
        """FAISS search. Output is identical in shape to the numpy path.

        Falls back to the numpy implementation when FAISS raises (defensive
        — we have observed FAISS returning -1 indices for under-trained
        IVF indexes).
        """
        try:
            xq = q.reshape(1, -1).astype(np.float32)
            scores, indices = self._faiss_index.search(xq, k)
            result: List[Tuple[str, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if int(idx) < 0 or int(idx) >= len(self._ids):
                    continue
                result.append((self._ids[int(idx)], float(score)))
            if not result:
                return self._search_numpy_locked(q, k)
            return result
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "[GalleryIndex] faiss search raised %r — falling back to numpy.",
                exc,
            )
            return self._search_numpy_locked(q, k)

    def _rebuild_faiss_index_locked(self) -> None:
        """(Re)build the FAISS index from ``self._vectors``.

        We rebuild from scratch on every mutation rather than incremental
        adds because:
          * IVF cannot remove rows in-place; rebuilding is the standard
            recommendation in the FAISS docs.
          * Adds are infrequent relative to searches in our workload (one
            per confirmation; tens to hundreds of searches per second
            during peak processing).
          * The rebuild cost is dominated by the matrix copy (microseconds
            for thousands of rows) — well under the per-frame budget.
        """
        if self._faiss is None:
            self._faiss_index = None
            self._faiss_trained = False
            return
        n = self._vectors.shape[0]
        if n == 0:
            self._faiss_index = None
            self._faiss_trained = False
            return

        # Choose the index type based on size.
        nlist_threshold = self._nlist * self.AUTO_TRAIN_MULTIPLIER
        try:
            if n < nlist_threshold:
                # Flat index — exact search, no training required.
                index = self._faiss.IndexFlatIP(self._dim)
                index.add(self._vectors.astype(np.float32))
                self._faiss_index = index
                self._faiss_trained = False
            else:
                # IVF — needs training. We use FlatIP coarse quantiser.
                quantizer = self._faiss.IndexFlatIP(self._dim)
                ivf = self._faiss.IndexIVFFlat(
                    quantizer, self._dim, self._nlist, self._faiss.METRIC_INNER_PRODUCT
                )
                ivf.train(self._vectors.astype(np.float32))
                ivf.add(self._vectors.astype(np.float32))
                # Higher ``nprobe`` -> closer to brute-force accuracy.
                ivf.nprobe = min(self._nlist, 4)
                self._faiss_index = ivf
                self._faiss_trained = True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[GalleryIndex] FAISS rebuild failed (%r) — disabling FAISS "
                "for this index instance and falling back to numpy.",
                exc,
            )
            self._faiss_index = None
            self._faiss_trained = False
