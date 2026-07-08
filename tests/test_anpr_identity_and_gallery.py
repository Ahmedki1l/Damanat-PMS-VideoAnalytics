"""Tests for the ANPR identity-binding (C1) and gallery-seed-at-confirm (A1)
changes.

C1 — the ANPR-image entry path must bind a candidate to the SPECIFIC plate whose
image it is (by event_id), never the FIFO-oldest pending entry, so two cars
entering close together cannot swap plates.

A1 — a car must get a durable per-plate gallery folder
(vehicle_images/gallery/<plate>/) the moment it enters the gate, so EVERY
entering car has a folder even if the CAM-03 B-entry reference never arrives.
The wide gate-only ANPR shot is persisted but flagged (gate_only) so it is
excluded from warm-start matching — the folder exists, yet the gate shot can
never false-match against a parked car. The authoritative CAM-03 B-entry
reference is what adds the first *matchable* vector.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np

from src.config import MatchingConfig
from src.vehicle_registry.vehicle_registry import VehicleRegistry
from src.vehicle_registry.gallery_store import safe_plate


class FakeMatcher:
    """Deterministic stand-in for the ReID model — feature encodes mean pixel."""

    backend = "fake"

    def extract_feature(self, image):
        value = float(np.mean(image) / 255.0)
        return np.array([value, 1.0 - value], dtype=np.float32)

    def extract_features_batch(self, images):
        return [self.extract_feature(image) for image in images]

    @staticmethod
    def compute_similarity(feat1, feat2):
        if feat1 is None or feat2 is None:
            return 0.0
        return float(np.dot(feat1, feat2))


class FakeImageMatcher:
    @staticmethod
    def _compare_dominant_colors(_a, _b):
        return 1.0

    @staticmethod
    def compare(_a, _b):
        return 0.0


def _make_registry(gallery=False):
    image_dir = tempfile.mkdtemp(prefix="pms_anpr_test_")
    cfg = MatchingConfig()
    if gallery:
        cfg.gallery_persist_enabled = True
        cfg.reid_openvino_model_dir = ""  # tag → "…:default"
    registry = VehicleRegistry(image_dir=image_dir, matching_config=cfg)
    registry._reid_matcher = FakeMatcher()
    registry._matcher = FakeImageMatcher()
    return registry, image_dir


def _image(value):
    return np.full((24, 24, 3), value, dtype=np.uint8)


# ── C1 — deterministic ANPR-image → event binding ─────────────────────────────
class TestAnprEventBinding(unittest.TestCase):
    def test_binds_the_specific_event_not_fifo_oldest(self):
        """With two pending entries, binding by event_id must return THAT plate,
        not the FIFO-oldest — the core anti-swap guarantee."""
        registry, _ = _make_registry()
        now = datetime.now()
        first = registry.register_anpr_event("AAA-111", "entry", timestamp=now)
        second = registry.register_anpr_event(
            "BBB-222", "entry", timestamp=now + timedelta(seconds=1)
        )
        # sanity: FIFO would pick the older AAA-111
        cand_fifo = registry.open_park_entry_candidate("ANPR", -1)
        self.assertEqual(
            registry.bind_next_pending_anpr_to_candidate(cand_fifo.candidate_id),
            "AAA-111",
        )
        # event-specific bind of the SECOND plate returns BBB-222
        cand = registry.open_park_entry_candidate("ANPR", -2)
        bound = registry.bind_anpr_event_to_candidate(
            cand.candidate_id, second.event_id
        )
        self.assertEqual(bound, "BBB-222")
        self.assertEqual(
            registry._pending_events[second.event_id].candidate_id,
            cand.candidate_id,
        )

    def test_rebind_retires_the_stale_candidate(self):
        """A burst re-read binds a fresh candidate to an already-provisional
        event; the previously-bound candidate must be retired (dropped) so both
        don't compete at B1."""
        registry, _ = _make_registry()
        event = registry.register_anpr_event("CCC-333", "entry")
        c1 = registry.open_park_entry_candidate("ANPR", -1)
        self.assertEqual(
            registry.bind_anpr_event_to_candidate(c1.candidate_id, event.event_id),
            "CCC-333",
        )
        c2 = registry.open_park_entry_candidate("ANPR", -2)
        self.assertEqual(
            registry.bind_anpr_event_to_candidate(c2.candidate_id, event.event_id),
            "CCC-333",
        )
        # old candidate retired, event now points at the new one
        self.assertEqual(registry._park_entry_candidates[c1.candidate_id].status, "dropped")
        self.assertIsNone(registry._park_entry_candidates[c1.candidate_id].bound_event_id)
        self.assertEqual(
            registry._pending_events[event.event_id].candidate_id, c2.candidate_id
        )

    def test_returns_none_for_unknown_or_expired_event(self):
        """Caller falls back to FIFO when the event is not bindable."""
        registry, _ = _make_registry()
        cand = registry.open_park_entry_candidate("ANPR", -1)
        self.assertIsNone(
            registry.bind_anpr_event_to_candidate(cand.candidate_id, "does-not-exist")
        )
        old = registry.register_anpr_event(
            "DDD-444", "entry",
            timestamp=datetime.now() - timedelta(
                seconds=registry.PENDING_ANPR_EXPIRY_SECONDS + 5
            ),
        )
        self.assertIsNone(
            registry.bind_anpr_event_to_candidate(cand.candidate_id, old.event_id)
        )


# ── A1 — gallery folder seeded at confirmation ────────────────────────────────
class TestGallerySeedAtConfirm(unittest.TestCase):
    def _plate_dir(self, image_dir, plate):
        return os.path.join(image_dir, "gallery", safe_plate(plate))

    def test_gate_entry_seeds_gate_only_folder(self):
        registry, image_dir = _make_registry(gallery=True)
        plate = "SEED-01"
        # The ANPR front crop seeds the durable folder the moment the car passes
        # the gate — so EVERY entering car has a folder — as a MATCHABLE
        # ground-truth reference (the canonical front viewpoint). Live entry-time
        # parked-car latching is prevented by the separate session-level
        # gate_reference_only guard, not by excluding the crop from the gallery.
        sid = registry.confirm_anpr_session_directly(
            plate=plate,
            image=_image(120),
            event_id="evt-seed",
            candidate_id="cand-seed",
            gate_snapshot_paths=[],
        )
        self.assertIsNotNone(sid)
        pdir = self._plate_dir(image_dir, plate)
        self.assertTrue(
            os.path.isdir(pdir),
            "gate ANPR entry must seed the durable folder",
        )
        self.assertTrue(os.path.isfile(os.path.join(pdir, "meta.json")))
        self.assertTrue([f for f in os.listdir(pdir) if f.endswith(".jpg")])
        # The ANPR front shot is a matchable ground-truth reference: a warm-start
        # vector exists immediately.
        vectors, _, _ = registry.gallery_store.load_vectors(plate)
        self.assertTrue(
            vectors, "ANPR front shot must be a matchable ground-truth reference"
        )
        # The authoritative CAM-03 B-entry reference adds another matchable ref.
        registry.confirm_b1_entrance_by_plate(plate, _image(200))
        matchable, _, _ = registry.gallery_store.load_vectors(plate)
        self.assertTrue(matchable, "B-entry must add a matchable reference vector")

    def test_park_entry_seed_is_noop_when_gate_already_seeded(self):
        registry, image_dir = _make_registry(gallery=True)
        plate = "SEED-PE"
        # Gate entry already seeds the folder with the matchable ANPR front ref.
        # When the car later reaches the CAM-23 Park_Entry zone, the seed is a
        # no-op — the folder is already there.
        registry.register_anpr_event(plate, "entry")
        registry.confirm_anpr_session_directly(
            plate=plate, image=_image(120), event_id="evt-pe",
            candidate_id="cand-pe", gate_snapshot_paths=[],
        )
        pdir = self._plate_dir(image_dir, plate)
        self.assertTrue(os.path.isdir(pdir), "gate entry seeds the folder")

        candidate = registry.open_park_entry_candidate("CAM-23", 7)
        registry.update_park_entry_candidate_snapshot(
            candidate.candidate_id, _image(120), quality_score=5.0
        )
        bound = registry.bind_next_pending_anpr_to_candidate(candidate.candidate_id)
        self.assertEqual(bound, plate)
        # Folder already exists → Park_Entry seed no-ops (returns False).
        self.assertFalse(
            registry.seed_gallery_from_park_entry(candidate.candidate_id, plate)
        )
        vectors, _, _ = registry.gallery_store.load_vectors(plate)
        self.assertTrue(
            vectors, "ANPR front seed is a matchable ground-truth reference"
        )
        # The authoritative CAM-03 B-entry reference adds another matchable ref.
        registry.confirm_b1_entrance_by_plate(plate, _image(200))
        matchable, _, _ = registry.gallery_store.load_vectors(plate)
        self.assertTrue(matchable, "B-entry must add a matchable reference vector")

    def test_park_entry_seeds_folder_when_gate_did_not(self):
        registry, image_dir = _make_registry(gallery=True)
        plate = "SEED-PE-FB"
        # Fallback path: no gate confirm seeded the folder (e.g. the gate image
        # never yielded a feature), so Park_Entry is the seeder — the folder still
        # exists the moment the car reaches the CAM-23 Park_Entry zone.
        registry.register_anpr_event(plate, "entry")
        candidate = registry.open_park_entry_candidate("CAM-23", 8)
        registry.update_park_entry_candidate_snapshot(
            candidate.candidate_id, _image(120), quality_score=5.0
        )
        self.assertEqual(
            registry.bind_next_pending_anpr_to_candidate(candidate.candidate_id),
            plate,
        )
        self.assertTrue(
            registry.seed_gallery_from_park_entry(candidate.candidate_id, plate)
        )
        pdir = self._plate_dir(image_dir, plate)
        self.assertTrue(os.path.isdir(pdir), "Park_Entry must create gallery/<plate>/")
        self.assertTrue(os.path.isfile(os.path.join(pdir, "meta.json")))
        vectors, _, _ = registry.gallery_store.load_vectors(plate)
        self.assertTrue(
            vectors, "CAM-23 Park_Entry seed is a matchable ground-truth reference"
        )

    def test_park_entry_seed_is_idempotent_per_visit(self):
        registry, image_dir = _make_registry(gallery=True)
        plate = "SEED-PE2"
        registry.register_anpr_event(plate, "entry")
        candidate = registry.open_park_entry_candidate("CAM-23", 9)
        registry.update_park_entry_candidate_snapshot(
            candidate.candidate_id, _image(120), quality_score=5.0
        )
        registry.bind_next_pending_anpr_to_candidate(candidate.candidate_id)
        self.assertTrue(
            registry.seed_gallery_from_park_entry(candidate.candidate_id, plate)
        )
        # A second call (a later frame) must not re-seed the already-created folder.
        self.assertFalse(
            registry.seed_gallery_from_park_entry(candidate.candidate_id, plate)
        )

    def test_no_folder_when_persistence_disabled(self):
        registry, image_dir = _make_registry(gallery=False)
        plate = "SEED-02"
        registry.confirm_anpr_session_directly(
            plate=plate, image=_image(120), event_id="e", candidate_id="c",
            gate_snapshot_paths=[],
        )
        registry.confirm_b1_entrance_by_plate(plate, _image(200))
        self.assertFalse(os.path.isdir(self._plate_dir(image_dir, plate)))

    def test_exit_only_creates_no_session_and_no_folder(self):
        registry, image_dir = _make_registry(gallery=True)
        plate = "EXIT-99"
        registry.register_anpr_event(plate, "exit")
        self.assertFalse(os.path.isdir(self._plate_dir(image_dir, plate)))
        self.assertEqual(
            [s for s in registry._sessions.values() if s.plate == plate], []
        )


if __name__ == "__main__":
    unittest.main()
