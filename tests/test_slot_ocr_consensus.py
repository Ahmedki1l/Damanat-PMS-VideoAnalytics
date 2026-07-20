"""Multi-frame OCR consensus for the parked-slot identify path.

Guards the fallback that binds a car ReID cannot shortlist (no parked-pose gallery
reference) by accumulating repeated reads — WITHOUT loosening the correct-or-null
contract. Models the real B13/CAM-24 case: DJS-7842 read at 1080p as a clean digit run
with mush letters, invisible to cross-view ReID.
"""

import types
import unittest

import src.vehicle_registry.vehicle_registry_identity as vid


def _mixin():
    for name in dir(vid):
        obj = getattr(vid, name)
        if isinstance(obj, type) and hasattr(obj, "confirm_slot_ocr_consensus"):
            return obj
    raise AssertionError("confirm_slot_ocr_consensus not found on any class")


_Mixin = _mixin()


class _FakeRegistry(_Mixin):
    """Minimal stand-in exposing only what confirm_slot_ocr_consensus touches."""

    def __init__(self, inside, *, enabled=True, min_agreement=3, margin=2):
        self._matching_config = types.SimpleNamespace(
            slot_ocr_consensus_enabled=enabled,
            slot_ocr_consensus_min_agreement=min_agreement,
            slot_ocr_consensus_margin=margin,
        )
        self._inside = list(inside)
        self._locked = set()

    def plates_inside(self, require_appearance=None):
        return list(self._inside)

    def _is_plate_locked_elsewhere(self, plate, slot_id):
        return plate in self._locked


def _feed(reg, reads, slot="B13", token="t"):
    """Feed reads until one binds; return (frame_index, plate) or (None, None).

    Mirrors the engine: once a plate binds the slot disarms, so no further reads run.
    """
    for i, r in enumerate(reads, 1):
        bound = reg.confirm_slot_ocr_consensus(slot, r, gen_token=token)
        if bound:
            return i, bound
    return None, None


# Production-style 1080p reads of DJS-7842: digit run 7842 exact, letters mush.
DJS_READS = ["7842015", "78420JS", "78420J8", "A78420J5"]


class TestSlotOcrConsensus(unittest.TestCase):
    def test_binds_the_unique_inside_car_on_agreement(self):
        i, plate = _feed(_FakeRegistry(["DJS-7842", "AXR-1120"]), DJS_READS)
        self.assertEqual(plate, "DJS-7842")
        self.assertEqual(i, 3)  # min_agreement=3

    def test_abstains_on_digit_run_collision(self):
        # A phantom sharing the 7842 run makes every read ambiguous -> never a vote.
        i, plate = _feed(_FakeRegistry(["DJS-7842", "BJA-7842"]), DJS_READS * 4)
        self.assertIsNone(plate)

    def test_two_reads_are_below_the_agreement_floor(self):
        i, plate = _feed(_FakeRegistry(["DJS-7842"]), DJS_READS[:2])
        self.assertIsNone(plate)

    def test_never_invents_a_plate_not_believed_inside(self):
        i, plate = _feed(_FakeRegistry(["AXR-1120"]), ["7842015"] * 6)
        self.assertIsNone(plate)

    def test_locked_elsewhere_plate_is_excluded(self):
        reg = _FakeRegistry(["DJS-7842"])
        reg._locked.add("DJS-7842")  # already parked+locked in another slot
        i, plate = _feed(reg, DJS_READS * 3)
        self.assertIsNone(plate)

    def test_generation_token_change_resets_votes(self):
        reg = _FakeRegistry(["DJS-7842"])
        self.assertIsNone(reg.confirm_slot_ocr_consensus("B13", "7842015", gen_token="A"))
        self.assertIsNone(reg.confirm_slot_ocr_consensus("B13", "78420JS", gen_token="A"))
        # New occupant (token B): the two prior votes must not carry over.
        self.assertIsNone(reg.confirm_slot_ocr_consensus("B13", "78420J8", gen_token="B"))

    def test_disabled_switch_never_binds(self):
        reg = _FakeRegistry(["DJS-7842"], enabled=False)
        i, plate = _feed(reg, DJS_READS * 3)
        self.assertIsNone(plate)

    def test_margin_blocks_a_split_vote(self):
        # Two inside cars with DISTINCT runs; reads alternate so neither leads by margin.
        reg = _FakeRegistry(["DJS-7842", "RDJ-9640"], min_agreement=3, margin=2)
        reads = ["7842015", "9640RDJ", "7842015", "9640RDJ", "7842015", "9640RDJ"]
        i, plate = _feed(reg, reads)
        self.assertIsNone(plate)

    def test_clear_votes_is_safe_when_empty(self):
        reg = _FakeRegistry(["DJS-7842"])
        reg.clear_slot_ocr_votes("nonexistent")  # must not raise


if __name__ == "__main__":
    unittest.main()
