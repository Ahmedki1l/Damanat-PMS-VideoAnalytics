"""Recovering a car that is parked inside with no open session.

The circularity this breaks: `_restore_vehicle_galleries` hydrates the appearance
gallery only for plates that HAVE an open `parking_session`, so ReID holds no
vectors for a car with no session — and slot OCR then discards its own correct
read for naming a car "ReID has never seen". Measured live on 2026-07-30: B13
returned `BHD` on 6 of 6 frames and stayed NULL, because BHD-9990 appeared
nowhere in the running process.

`offsession_gallery_candidates` adds those plates as CANDIDATES. It never names
anything by itself, and these tests pin that as hard as the successes: on the
first live run ReID ranked RZD-4976 above the true ZZR-1372, and only the OCR
witness stopped a stranger's plate landing on B7_CHRO.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.config import MatchingConfig
from src.vehicle_registry.gallery_store import VehicleGalleryStore
from src.vehicle_registry.vehicle_registry_identity import VehicleRegistryIdentityMixin


def _vec(*head) -> np.ndarray:
    v = np.zeros(64, dtype=np.float32)
    for i, x in enumerate(head):
        v[i] = x
    return v / (np.linalg.norm(v) or 1.0)


class _Store:
    """Gallery on disk, with no dependence on sessions — the point of the fix.

    ``mapping`` is {plate: [vec, ...]} or {plate: ([vec, ...], [camera, ...])} when a
    test needs to say which camera taught each reference.
    """

    def __init__(self, mapping):
        self._m = mapping

    def all_plates(self):
        return list(self._m)

    def load_vectors(self, plate, *, current_tag_only=False):
        entry = self._m.get(plate, [])
        if isinstance(entry, tuple):
            vecs, cams = entry
            return list(vecs), "tag", list(cams)
        return list(entry), "tag", []

    def load_crops(self, plate):
        # Only reached when load_vectors returns nothing; these stubs always have
        # vectors, so the re-embed fallback must not fire.
        return [], []


class _Reg(VehicleRegistryIdentityMixin):
    def __init__(self, gallery, sessions=(), locked=()):
        import threading
        self._lock = threading.RLock()
        self._matching_config = MatchingConfig()
        self._matching_config.slot_recovery_enabled = True
        self._matching_config.slot_recovery_min_score = 0.55
        self._matching_config.slot_recovery_min_margin = 0.10
        self._matching_config.slot_recovery_top_k = 5
        self._gallery_store = _Store(gallery)
        self._sessions = {}
        for p in sessions:
            s = MagicMock()
            s.plate = p
            self._sessions[p] = s
        self._locked = set(locked)

    @property
    def gallery_store(self):
        return self._gallery_store

    def _is_plate_locked_elsewhere(self, plate, slot_id):
        return plate in self._locked


# Distinguishable cars (dot ~0.71, so the winner clears the 0.10 margin), plus a
# stranger orthogonal to every one of them.
CAR_A = _vec(1.0, 0.0)
CAR_B = _vec(1.0, 1.0)
STRANGER = _vec(0.0, 0.0, 1.0)
GALLERY = {"AAA-1111": [CAR_A], "BBB-2222": [CAR_B], "CCC-3333": [_vec(0.0, 1.0)]}


class TestOffSessionCandidates(unittest.TestCase):
    def test_car_with_no_session_becomes_a_candidate(self):
        """The whole point: reid_rank cannot see this car, this path can."""
        r = _Reg(GALLERY)
        got = r.offsession_gallery_candidates(CAR_A, slot_id="B13")
        self.assertIn("AAA-1111", got)

    def test_cars_that_already_have_a_session_are_skipped(self):
        """reid_rank already ranks those — duplicating them widens the pool for free."""
        r = _Reg(GALLERY, sessions=["AAA-1111"])
        self.assertNotIn("AAA-1111", r.offsession_gallery_candidates(CAR_A, slot_id="B13"))

    def test_plate_locked_to_another_slot_is_skipped(self):
        """A car cannot be in two slots — the same rule reid_rank applies."""
        r = _Reg(GALLERY, locked=["AAA-1111"])
        self.assertNotIn("AAA-1111", r.offsession_gallery_candidates(CAR_A, slot_id="B13"))

    def test_stranger_gets_no_candidates(self):
        """Open-set guard: a car in nobody's gallery must return NOTHING.

        This is the failure that matters. An unenrolled car scores 0.218-0.339
        against the whole gallery (measured), so the floor is what refuses it —
        without it the nearest of a bad bunch would be nominated.
        """
        r = _Reg(GALLERY)
        self.assertEqual(r.offsession_gallery_candidates(STRANGER, slot_id="B13"), [])

    def test_flat_field_is_refused_even_when_the_score_is_high(self):
        """Two near-identical cars => no margin => refuse.

        A high score with a flat margin means several cars explain the crop equally
        well. Binding the top one there is how appearance stamps the wrong plate.
        """
        twins = {"AAA-1111": [CAR_A], "AAB-1112": [CAR_A]}
        r = _Reg(twins)
        self.assertEqual(r.offsession_gallery_candidates(CAR_A, slot_id="B13"), [])

    def test_disabled_by_default(self):
        r = _Reg(GALLERY)
        r._matching_config.slot_recovery_enabled = False
        self.assertEqual(r.offsession_gallery_candidates(CAR_A, slot_id="B13"), [])
        self.assertFalse(MatchingConfig().slot_recovery_enabled,
                         "recovery must be opt-in: it widens the candidate pool to "
                         "every car ever seen")

    def test_no_query_vector_is_a_no_op(self):
        self.assertEqual(_Reg(GALLERY).offsession_gallery_candidates(None, slot_id="B13"), [])

    def test_gallery_scan_failure_degrades_silently(self):
        class Broken(_Store):
            def all_plates(self):
                raise OSError("disk gone")

        r = _Reg(GALLERY)
        r._gallery_store = Broken({})
        self.assertEqual(r.offsession_gallery_candidates(CAR_A, slot_id="B13"), [])

    def test_vectors_are_cached_between_calls(self):
        """The cold scan is ~4.6s for 38 plates; it must not run per frame."""
        calls = []

        class Counting(_Store):
            def all_plates(self):
                calls.append(1)
                return super().all_plates()

        r = _Reg(GALLERY)
        r._gallery_store = Counting(GALLERY)
        for _ in range(5):
            r.offsession_gallery_candidates(CAR_A, slot_id="B13")
        self.assertEqual(len(calls), 1, "gallery rescanned on every call")


class TestSameViewIsPrimary(unittest.TestCase):
    """A parked-pose reference from THIS camera is the strong evidence.

    `save_parked_reference` writes a crop of the car standing in the slot once OCR
    has named it. On the car's next visit that reference is a SAME-VIEW match
    (rank-1 0.976) rather than a cross-view one against a gate photo (0.736, and
    per-car INVERTED: 0.583 for the right car vs 0.634 for a different one).
    Ranking must reflect that, or a warm car loses to a stranger's lucky gate score.
    """

    def test_warm_car_outranks_a_higher_scoring_cold_one(self):
        gallery = {
            # taught at the slot camera on a previous visit, slightly lower score
            "WARM-1111": ([_vec(0.98, 0.20)], ["CAM-24"]),
            # gate photo only, scores higher on raw cosine
            "COLD-2222": ([_vec(1.0, 0.0)], ["ANPR"]),
        }
        r = _Reg({})
        r._gallery_store = _Store(gallery)
        got = r.offsession_gallery_candidates(
            _vec(1.0, 0.0), slot_id="B13", slot_camera="CAM-24")
        self.assertEqual(got[0], "WARM-1111",
                         "a parked-pose match at this camera must outrank a gate photo")

    def test_warm_car_is_scored_on_its_parked_pose_only(self):
        """Its own gate photos must not drag a same-view match down."""
        gallery = {"WARM-1111": (
            [_vec(1.0, 0.0), _vec(0.0, 1.0)], ["CAM-24", "ANPR"])}
        r = _Reg({})
        r._gallery_store = _Store(gallery)
        self.assertIn("WARM-1111", r.offsession_gallery_candidates(
            _vec(1.0, 0.0), slot_id="B13", slot_camera="CAM-24"))

    def test_no_slot_camera_falls_back_to_best_of_all_references(self):
        gallery = {"AAA-1111": ([_vec(1.0, 0.0)], ["ANPR"]),
                   "BBB-2222": ([_vec(0.0, 1.0)], ["ANPR"])}
        r = _Reg({})
        r._gallery_store = _Store(gallery)
        self.assertIn("AAA-1111", r.offsession_gallery_candidates(
            _vec(1.0, 0.0), slot_id="B13", slot_camera=None))

    def test_margin_compares_like_with_like(self):
        """A warm winner is measured against other WARM cars, not against cold ones.

        Comparing warm 0.97 to cold 0.70 would invent a 0.27 margin out of the
        viewpoint gap rather than out of any difference between the cars.
        """
        gallery = {
            "WARM-A": ([_vec(1.0, 0.0)], ["CAM-24"]),
            "WARM-B": ([_vec(1.0, 0.02)], ["CAM-24"]),   # near-identical warm peer
            "COLD-C": ([_vec(0.0, 1.0)], ["ANPR"]),
        }
        r = _Reg({})
        r._gallery_store = _Store(gallery)
        self.assertEqual(
            r.offsession_gallery_candidates(_vec(1.0, 0.0), slot_id="B13",
                                            slot_camera="CAM-24"),
            [], "two warm cars this close must not produce a confident answer")


class TestAllPlates(unittest.TestCase):
    def test_reads_the_plate_from_meta_not_the_folder_name(self):
        """Folder names are safe_plate()-mangled and not reversible."""
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "gallery", "BHD_9990")
            os.makedirs(root)
            with open(os.path.join(root, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"plate": "BHD-9990", "refs": []}, fh)
            self.assertEqual(VehicleGalleryStore(d, "tag").all_plates(), ["BHD-9990"])

    def test_missing_root_and_junk_folders_are_survivable(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(VehicleGalleryStore(d, "tag").all_plates(), [])
            os.makedirs(os.path.join(d, "gallery", "no_meta_here"))
            self.assertEqual(VehicleGalleryStore(d, "tag").all_plates(), [])


if __name__ == "__main__":
    unittest.main()


class TestSilentNoOpGuard(unittest.TestCase):
    """Folders on disk but no usable vectors must SAY so.

    Strict gallery admission drops every reference lacking an `entry_v2_parked_v1`
    proof. Measured on the live gallery 2026-07-30: 38 folders, 38 usable with the
    flag off, ZERO with it on. Recovery then silently never fires, which reads as
    "ReID isn't good enough" rather than "the gallery was shut" — the same
    misdiagnosis that cost a day on 2026-07-27.
    """

    def _reg_with_empty_vectors(self):
        class Shut(_Store):
            def load_vectors(self, plate):
                return [], "tag", []      # strict admission filtered them all

        r = _Reg(GALLERY)
        r._gallery_store = Shut(GALLERY)
        return r

    def test_warns_when_folders_exist_but_yield_nothing(self):
        r = self._reg_with_empty_vectors()
        with self.assertLogs("src.vehicle_registry.vehicle_registry_identity",
                             level="WARNING") as cm:
            r._offsession_gallery_vectors()
        self.assertTrue(
            any("gallery_strict_admission_enabled" in m for m in cm.output),
            f"expected a strict-admission hint, got {cm.output}")

    def test_warns_only_once(self):
        r = self._reg_with_empty_vectors()
        with self.assertLogs("src.vehicle_registry.vehicle_registry_identity",
                             level="WARNING") as cm:
            for _ in range(4):
                r._offsession_cache_until = 0.0     # force a rescan each time
                r._offsession_gallery_vectors()
        hits = [m for m in cm.output if "gallery_strict_admission_enabled" in m]
        self.assertEqual(len(hits), 1, "must not warn on every rescan")

    def test_no_warning_when_the_gallery_is_simply_empty(self):
        """Nothing on disk is a normal new deployment, not a misconfiguration."""
        r = _Reg({})
        r._gallery_store = _Store({})
        with self.assertNoLogs("src.vehicle_registry.vehicle_registry_identity",
                               level="WARNING"):
            r._offsession_gallery_vectors()


class TestSoloTier(unittest.TestCase):
    """ReID alone, for slots where OCR can never corroborate.

    Bar set from leave-one-out over the 50 production identities / 782 refs,
    cross-view: 100% precision at margin >=0.35 (15.5% recall) against 97.2% at
    the ordinary 0.10 gate. The one wrong answer seen live — B7_CHRO ranked
    RZD-4976 at margin 0.324 — sits just under it and must stay refused.
    """

    def _reg(self):
        # Well-separated field: the winner clears 0.35 over the runner-up. The
        # default CAR_A/CAR_B pair sits at 0.293, which this tier refuses on
        # purpose — see test_margin_just_below_the_bar_is_refused.
        gallery = {"AAA-1111": [_vec(1.0, 0.0)],
                   "BBB-2222": [_vec(1.0, 1.5)],
                   "CCC-3333": [_vec(0.0, 1.0)]}
        r = _Reg(gallery)
        r._matching_config.slot_recovery_solo_enabled = True
        r._matching_config.slot_recovery_solo_min_margin = 0.35
        return r

    def test_confident_match_is_admitted_alone(self):
        r = self._reg()
        got = r.offsession_solo_candidate(CAR_A, slot_id="B13")
        self.assertIsNotNone(got)
        self.assertEqual(got[0], "AAA-1111")
        self.assertGreaterEqual(got[2], 0.35)

    def test_margin_just_below_the_bar_is_refused(self):
        """The live B7_CHRO miss scored 0.324. It must not be admitted."""
        r = self._reg()
        r._matching_config.slot_recovery_solo_min_margin = 0.35
        # two cars close enough that the winner's margin lands under the bar
        r._gallery_store = _Store({"AAA-1111": [CAR_A], "AAB-1112": [_vec(1.0, 0.55)]})
        r._offsession_cache_until = 0.0
        self.assertIsNone(r.offsession_solo_candidate(CAR_A, slot_id="B13"))

    def test_off_by_default(self):
        r = _Reg(GALLERY)          # solo not enabled
        self.assertIsNone(r.offsession_solo_candidate(CAR_A, slot_id="B13"))
        self.assertFalse(MatchingConfig().slot_recovery_solo_enabled,
                         "appearance-alone recovery must be opt-in")

    def test_stranger_is_refused(self):
        r = self._reg()
        self.assertIsNone(r.offsession_solo_candidate(STRANGER, slot_id="B13"))
