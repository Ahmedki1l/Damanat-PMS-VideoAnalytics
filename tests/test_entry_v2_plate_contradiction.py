"""Digit-run plate comparison — the replacement for exact-key inequality.

Every case here is taken from real traffic in the 2026-08-30/31 shadow window.
Exact keys called all of them contradictions, and each one would have WITHHELD
A CORRECT ENTRY: the producer-family gate refused one real crossing thirteen
times on the 30th on exactly this basis.
"""

import unittest

from src.entry.domain import plate_digit_run, plates_contradict


class DigitRunTests(unittest.TestCase):
    def test_order_does_not_matter(self):
        # The UI renders plates digits-first, the DB letters-first.
        self.assertEqual("7286", plate_digit_run("7286EED"))
        self.assertEqual("7286", plate_digit_run("EED-7286"))

    def test_separators_are_ignored(self):
        self.assertEqual("6951", plate_digit_run("GBA-6951"))


class NotAContradictionTests(unittest.TestCase):
    """Measured pairs that name ONE car."""

    def test_digits_first_versus_letters_first(self):
        self.assertFalse(plates_contradict("7286EED", "EED7286"))

    def test_hallucinated_letter_prefix(self):
        # Two reads of the same car in one window, both above 0.75 confidence.
        self.assertFalse(plates_contradict("7383HAS", "AATEIGH7383HAS"))

    def test_a_leading_digit_invented_by_the_ocr(self):
        # The 6951 burst: one car read as GBA-6951, 56951GB and 66951DB.
        self.assertFalse(plates_contradict("GBA-6951", "56951GB"))
        self.assertFalse(plates_contradict("GBA-6951", "66951DB"))

    def test_digits_lost_off_the_end(self):
        self.assertFalse(plates_contradict("KKR-6294", "KKR-629"))

    def test_identical_reads(self):
        self.assertFalse(plates_contradict("KKR-6294", "6294KKR"))

    def test_a_read_with_no_digits_contradicts_nothing(self):
        # Nothing to compare is not evidence of disagreement.
        self.assertFalse(plates_contradict("HAS", "7383HAS"))

    def test_empty_reads_are_not_contradictions(self):
        self.assertFalse(plates_contradict("", "7383HAS"))
        self.assertFalse(plates_contradict("7383HAS", ""))


class GenuineContradictionTests(unittest.TestCase):
    """The gate must still fire when the cars really are different."""

    def test_different_digits_entirely(self):
        self.assertTrue(plates_contradict("KKR-6294", "ZZR-1372"))

    def test_one_digit_apart_is_two_cars(self):
        # KKR-6294 and KKR-6295 are two vehicles, not one misread.
        self.assertTrue(plates_contradict("KKR-6294", "KKR-6295"))

    def test_a_middle_substring_is_not_a_truncation(self):
        # No camera drops digits off BOTH ends; 29 inside 6294 is a different
        # plate, not a shorter read of the same one.
        self.assertTrue(plates_contradict("KKR-6294", "KKR-29"))

    def test_two_confirmed_cars_from_the_same_day(self):
        self.assertTrue(plates_contradict("EED-7286", "EEB-80"))
