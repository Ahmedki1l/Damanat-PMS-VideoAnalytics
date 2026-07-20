"""Implausible plates must never reach a slot, and so never an alert.

`confirm_plate` can only ever return a member of the candidate set, so filtering
the candidate set is what makes "current_plate is CORRECT or NULL, never wrong"
structurally true rather than merely intended.

Vectors are the real 2026-07-20 entry feed: the gate minted 36663XN, 77842SJ,
66466RA, 87281EJ and 666EIAI as vehicles (owner "Unknown", one sighting each).
36663XN reached slot B2 and would have been the plate printed on that slot's
intrusion alert.
"""

import unittest

from src.matching.plate_ocr_match import confirm_plate, is_plausible_plate

# Seen on CAM-ENTRY, 2026-07-12..20. None is a real vehicle.
ARTEFACTS = ["36663XN", "77842SJ", "66466RA", "66466XA", "87281EJ", "666EIAI"]

# Real plates from the facility. HDU-7 has a single digit; ZVH-337 three.
REAL = ["RGR-6466", "ZVH-337", "BHD-9990", "HBR-4920", "XRD-6663",
        "EEB-80", "HVA-77", "LNV-94", "RZG-136", "HDU-7"]


class TestIsPlausiblePlate(unittest.TestCase):
    def test_artefacts_rejected(self):
        for p in ARTEFACTS:
            with self.subTest(plate=p):
                self.assertFalse(is_plausible_plate(p))

    def test_real_plates_accepted(self):
        for p in REAL:
            with self.subTest(plate=p):
                self.assertTrue(is_plausible_plate(p))

    def test_unknown_but_well_formed_visitor_accepted(self):
        """A first-time visitor is not an artefact — shape only, no registry."""
        for p in ["XYZ-1234", "BAS-6646", "ED-6644", "AB-12"]:
            with self.subTest(plate=p):
                self.assertTrue(is_plausible_plate(p))

    def test_arrangement_matters_not_just_run_lengths(self):
        """666EIAI has 4 letters and 3 digits — plausible by counts alone, but
        digits-first, which no stored plate is."""
        self.assertFalse(is_plausible_plate("666EIAI"))
        self.assertTrue(is_plausible_plate("EIAI-666"))

    def test_empty_and_junk(self):
        for p in [None, "", "   ", "-", "1234", "ABCD"]:
            with self.subTest(plate=p):
                self.assertFalse(is_plausible_plate(p))

    def test_case_and_whitespace_tolerated(self):
        self.assertTrue(is_plausible_plate("  rgr-6466 "))


class TestArtefactCannotBeConfirmed(unittest.TestCase):
    def test_filtered_candidate_set_cannot_yield_an_artefact(self):
        """The B2 case: an OCR read that would have matched the phantom binds
        nothing once the phantom is out of the candidate set."""
        candidates = [p for p in ["36663XN", "XRD-6663"] if is_plausible_plate(p)]
        self.assertEqual(candidates, ["XRD-6663"])
        # a read of the real car still confirms it
        self.assertEqual(confirm_plate("6663XRD", candidates), "XRD-6663")


if __name__ == "__main__":
    unittest.main()
