"""
tests/test_gallery_index.py — Phase 3 / T3.2 GalleryIndex regression tests.

Covers:

* Empty / single-element edge cases.
* Add / remove correctness.
* Search top-K agrees with brute-force cosine on a 100-vector dataset.
* Concurrent add + search smoke test.
* Idempotent re-add (same session_id replaces its vector).
* Dimension mismatch raises ValueError.

The tests are FAISS-agnostic — when ``faiss-cpu`` is installed the index
runs through the FAISS backend, otherwise the numpy fallback. The
assertions are tolerant to a small drift (1e-2) caused by IVF approximate
search.
"""

from __future__ import annotations

import threading
import unittest
from typing import List, Tuple

import numpy as np

from src.matching.gallery_index import GalleryIndex


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Return ``n`` L2-normalised random vectors of dimension ``dim``."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return vecs / norms


def _brute_force_topk(
    query: np.ndarray, vectors: np.ndarray, ids: List[str], k: int
) -> List[Tuple[str, float]]:
    """Reference brute-force cosine top-K used as the oracle."""
    scores = vectors @ query
    order = np.argsort(-scores)[:k]
    return [(ids[i], float(scores[i])) for i in order]


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


class TestEmptyAndSingleElement(unittest.TestCase):
    def test_empty_search_returns_empty_list(self):
        idx = GalleryIndex(dimension=8)
        self.assertEqual(idx.search(np.ones(8, dtype=np.float32), k=5), [])
        self.assertEqual(len(idx), 0)

    def test_single_element(self):
        idx = GalleryIndex(dimension=4)
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        idx.add("only", vec)
        self.assertEqual(len(idx), 1)
        results = idx.search(vec, k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "only")
        # Cosine of identical unit vectors is 1.
        self.assertAlmostEqual(results[0][1], 1.0, places=4)

    def test_search_with_k_zero(self):
        idx = GalleryIndex(dimension=4)
        idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
        self.assertEqual(idx.search(np.ones(4, dtype=np.float32), k=0), [])

    def test_none_query_returns_empty(self):
        idx = GalleryIndex(dimension=4)
        idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
        self.assertEqual(idx.search(None, k=1), [])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class TestValidation(unittest.TestCase):
    def test_rejects_bad_dimension(self):
        with self.assertRaises(ValueError):
            GalleryIndex(dimension=0)
        with self.assertRaises(ValueError):
            GalleryIndex(dimension=-3)

    def test_rejects_bad_nlist(self):
        with self.assertRaises(ValueError):
            GalleryIndex(dimension=4, nlist=0)

    def test_rejects_non_cosine_metric(self):
        with self.assertRaises(ValueError):
            GalleryIndex(dimension=4, metric="l2")

    def test_add_wrong_dimension_raises(self):
        idx = GalleryIndex(dimension=4)
        with self.assertRaises(ValueError):
            idx.add("a", np.ones(8, dtype=np.float32))

    def test_search_wrong_dimension_raises(self):
        idx = GalleryIndex(dimension=4)
        idx.add("a", np.ones(4, dtype=np.float32))
        with self.assertRaises(ValueError):
            idx.search(np.ones(8, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Add / remove
# --------------------------------------------------------------------------- #


class TestAddRemove(unittest.TestCase):
    def test_add_then_remove(self):
        idx = GalleryIndex(dimension=4)
        idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
        idx.add("b", np.array([0, 1, 0, 0], dtype=np.float32))
        self.assertEqual(len(idx), 2)
        self.assertIn("a", idx)
        self.assertIn("b", idx)
        self.assertTrue(idx.remove("a"))
        self.assertEqual(len(idx), 1)
        self.assertNotIn("a", idx)

    def test_remove_unknown_returns_false(self):
        idx = GalleryIndex(dimension=4)
        idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
        self.assertFalse(idx.remove("unknown"))

    def test_readd_updates_vector(self):
        idx = GalleryIndex(dimension=4)
        original = np.array([1, 0, 0, 0], dtype=np.float32)
        updated = np.array([0, 1, 0, 0], dtype=np.float32)
        idx.add("a", original)
        idx.add("a", updated)
        self.assertEqual(len(idx), 1)
        # The stored vector should now match ``updated``, not ``original``.
        results = idx.search(updated, k=1)
        self.assertEqual(results[0][0], "a")
        self.assertAlmostEqual(results[0][1], 1.0, places=4)
        results_old = idx.search(original, k=1)
        # Cosine with the orthogonal original should be ~0.
        self.assertLess(results_old[0][1], 0.1)

    def test_clear(self):
        idx = GalleryIndex(dimension=4)
        for i in range(5):
            v = np.zeros(4, dtype=np.float32)
            v[i % 4] = 1.0
            idx.add(f"sess-{i}", v)
        self.assertEqual(len(idx), 5)
        idx.clear()
        self.assertEqual(len(idx), 0)
        self.assertEqual(idx.search(np.ones(4, dtype=np.float32), k=5), [])

    def test_unnormalised_input_is_normalised(self):
        # Add a vector with norm != 1 — it should be normalised internally
        # so cosine search still returns a value of 1.0 for the same vec.
        idx = GalleryIndex(dimension=4)
        v = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)  # norm 5
        idx.add("a", v)
        results = idx.search(v, k=1)
        self.assertAlmostEqual(results[0][1], 1.0, places=4)


# --------------------------------------------------------------------------- #
# Search correctness vs brute-force
# --------------------------------------------------------------------------- #


class TestSearchVsBruteForce(unittest.TestCase):
    def test_top_k_matches_brute_force_on_100_vectors(self):
        n = 100
        dim = 64
        k = 10
        vecs = _make_unit_vectors(n, dim, seed=42)
        ids = [f"sess-{i:03d}" for i in range(n)]
        idx = GalleryIndex(dimension=dim)
        for sid, v in zip(ids, vecs):
            idx.add(sid, v)
        self.assertEqual(len(idx), n)

        # Use a known vector as the query — its self should be #1.
        query = vecs[7]
        results = idx.search(query, k=k)
        oracle = _brute_force_topk(query, vecs, ids, k)

        self.assertEqual(len(results), k)
        # Top-1 must always match (cosine 1.0 against itself).
        self.assertEqual(results[0][0], ids[7])
        self.assertEqual(oracle[0][0], ids[7])

        # The full set of returned IDs should match (order may differ
        # slightly because of ties + IVF approximations). Allow a tiny
        # overlap drift to accommodate FAISS-IVF's approximate search.
        returned_ids = {r[0] for r in results}
        oracle_ids = {o[0] for o in oracle}
        overlap = len(returned_ids & oracle_ids)
        # Brute-force backend gets perfect overlap; the FAISS path may
        # drop the bottom-most candidate when it lands across cells.
        self.assertGreaterEqual(overlap, k - 1)

        # Score for the top-1 must be within 0.01 of the brute-force.
        self.assertAlmostEqual(results[0][1], oracle[0][1], delta=0.01)

    def test_search_score_drift_within_tolerance(self):
        # Stress test: 200 vectors, query a random new vector. Compare
        # the top-1 score against the brute-force oracle.
        n = 200
        dim = 32
        vecs = _make_unit_vectors(n, dim, seed=99)
        ids = [f"sess-{i:03d}" for i in range(n)]
        idx = GalleryIndex(dimension=dim, nlist=4)
        for sid, v in zip(ids, vecs):
            idx.add(sid, v)
        # New random query (not in the gallery)
        query = _make_unit_vectors(1, dim, seed=7)[0]
        oracle = _brute_force_topk(query, vecs, ids, k=1)
        results = idx.search(query, k=1)
        # Cosine drift between FAISS IVF and brute force must stay
        # under 0.01 in the worst case.
        self.assertAlmostEqual(results[0][1], oracle[0][1], delta=0.01)


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


class TestConcurrency(unittest.TestCase):
    def test_concurrent_add_and_search(self):
        """Two threads adding while one is searching must not crash and the
        final length must equal the total inserted."""
        idx = GalleryIndex(dimension=16)
        n_per_thread = 50
        errors: List[Exception] = []

        def adder(start: int) -> None:
            try:
                for i in range(start, start + n_per_thread):
                    v = np.zeros(16, dtype=np.float32)
                    v[i % 16] = 1.0
                    idx.add(f"t-{i}", v)
            except Exception as exc:  # pragma: no cover - debug aid
                errors.append(exc)

        def searcher() -> None:
            try:
                q = np.ones(16, dtype=np.float32) / np.sqrt(16)
                for _ in range(20):
                    idx.search(q, k=5)
            except Exception as exc:  # pragma: no cover - debug aid
                errors.append(exc)

        threads = [
            threading.Thread(target=adder, args=(0,)),
            threading.Thread(target=adder, args=(n_per_thread,)),
            threading.Thread(target=searcher),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(idx), 2 * n_per_thread)


# --------------------------------------------------------------------------- #
# Backend probe
# --------------------------------------------------------------------------- #


class TestBackendReporting(unittest.TestCase):
    def test_backend_attribute_is_well_known(self):
        idx = GalleryIndex(dimension=8)
        self.assertIn(idx.backend, {"faiss-flat", "faiss-ivf", "numpy"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
