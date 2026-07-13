"""
current_plate must be CORRECT or NULL — never wrong.

Every real OCR read taken off the real parked cars on 2026-07-11 at 1080p:

    RDJ-9640  ->  '9640RDJ5'    letters clean
    RDJ-9640  ->  '9640R0J'     the D came through as a ZERO
    RDJ-9640  ->  'A9640R0J'    ...plus a phantom 'A'
    DJS-7842  ->  '78420JS'     only 'JS' of 'DJS'
    DJS-7842  ->  '78420J8'     the S came through as an EIGHT
    DJS-7842  ->  '78420J5'     ...or as a FIVE
    a Nissan Sunny (unknown)  ->  'L918410' / 'UEL918410' / 'SUNNY9190'
    a side-profile slot       ->  ''        (no plate in frame at all)

The digits are exact in all six reads of the two known cars. The letters are mush.
An earlier version of this matcher required the letters to agree and therefore REJECTED
correct reads — it refused '9640R0J' for RDJ-9640 because 'RJ' is not 'RDJ'.

So the digits carry the identification, and safety comes from UNIQUENESS rather than
from the letters. The failure being guarded against is not "we missed a plate" — it is
"we wrote a stranger's plate into a customer's slot", which is exactly what the
acquisition-by-elimination path did earlier that day (DJS-7842 onto that parked Sunny).
"""

import unittest

from src.matching.plate_ocr_match import confirm_plate, normalize, read_matches_plate


class TestReadMatchesPlate(unittest.TestCase):
    def test_every_real_read_of_the_two_known_cars(self):
        """All six. Digits exact; letters are D->0, S->8, S->5, plus a phantom 'A'."""
        for read in ("78420JS", "78420J8", "78420J5"):
            self.assertTrue(read_matches_plate(read, "DJS-7842"), read)
        for read in ("9640RDJ5", "9640R0J", "A9640R0J"):
            self.assertTrue(read_matches_plate(read, "RDJ-9640"), read)

    def test_separators_and_case_are_irrelevant(self):
        self.assertTrue(read_matches_plate("rdj 9640", "RDJ-9640"))
        self.assertTrue(read_matches_plate("9640", "rdj_9640"))

    def test_extra_ocr_noise_is_forgiven(self):
        """Badge text, phantom characters and partial Arabic are noise. The plate's
        digit run is still there, in order."""
        self.assertTrue(read_matches_plate("SUNNYRDJ96400X", "RDJ-9640"))

    # ---- the abstentions: these are what keep the column trustworthy ----

    def test_a_MISSING_digit_is_fatal(self):
        """'964' is not '9640'. The digits ARE the identification, so a partial digit
        run confirms nothing."""
        self.assertFalse(read_matches_plate("964RDJ", "RDJ-9640"))

    def test_the_wrong_cars_digits_are_fatal(self):
        """DJS-7842's read must never confirm RDJ-9640, however good the letters look."""
        self.assertFalse(read_matches_plate("78420JS", "RDJ-9640"))
        self.assertFalse(read_matches_plate("9640RDJ5", "DJS-7842"))

    def test_the_nissan_sunny_matches_no_car_inside(self):
        """THE WRONG BIND THAT ACTUALLY HAPPENED. Its real reads carry digits no car
        inside has, so its slot stays NULL — correct, because VA genuinely does not
        know that car."""
        for read in ("L918410", "UEL918410", "SUNNY9190"):
            for car in ("RDJ-9640", "DJS-7842", "HGD-2926"):
                self.assertFalse(read_matches_plate(read, car), f"{read} vs {car}")

    def test_garbage_and_empty_reads_confirm_nothing(self):
        for junk in ("CA", "", None, "   ", "!!", "RDJ"):  # 'RDJ' = letters, no digits
            self.assertFalse(read_matches_plate(junk, "RDJ-9640"))

    def test_a_side_profile_slot_reads_nothing_and_so_confirms_nothing(self):
        """CAM-21 frames B1_CRO in pure side profile — no plate in frame at all. NULL is
        the correct answer, and NULL is what it gives."""
        self.assertFalse(read_matches_plate("", "DJS-7842"))

    def test_a_plate_with_too_few_digits_is_refused(self):
        """Below two digits, noise starts matching by chance."""
        self.assertFalse(read_matches_plate("XYZ7ABC", "AB-7"))


class TestConfirmPlate(unittest.TestCase):
    """Uniqueness is where the safety lives, now that the letters cannot be trusted."""

    CARS = ["DJS-7842", "RDJ-9640", "HGD-2926"]

    def test_it_picks_the_one_car_the_read_confirms(self):
        self.assertEqual(confirm_plate("9640R0J", self.CARS), "RDJ-9640")
        self.assertEqual(confirm_plate("78420J8", self.CARS), "DJS-7842")

    def test_no_read_binds_nothing(self):
        self.assertIsNone(confirm_plate("", self.CARS))
        self.assertIsNone(confirm_plate(None, self.CARS))

    def test_a_read_matching_no_car_inside_binds_nothing(self):
        """The Sunny again — the exact slot the elimination path corrupted."""
        self.assertIsNone(confirm_plate("UEL918410", self.CARS))

    def test_TWO_matching_cars_ABSTAIN(self):
        """Ambiguity is never resolved by picking a winner — that is precisely how a
        stranger's plate ends up in a customer's slot. If two cars inside share a digit
        run, OCR cannot separate them, so it binds neither."""
        cars = ["RDJ-9640", "ABC-9640"]
        self.assertIsNone(confirm_plate("9640R0J", cars))
        # ...and each ALONE is still confirmable: the abstention is about the collision.
        self.assertEqual(confirm_plate("9640R0J", ["RDJ-9640"]), "RDJ-9640")

    def test_normalize(self):
        self.assertEqual(normalize("rdj-9640"), "RDJ9640")
        self.assertEqual(normalize(None), "")


if __name__ == "__main__":
    unittest.main()


class TestShortPlatesCannotMatchByChance(unittest.TestCase):
    """2026-07-12, live: a Nissan Sunny read 'SANNYL318AV0' and the matcher CONFIRMED
    EEB-80, stamping a stranger's plate into B25.

    'EEB-80' has only the digits '80' — and '80' is a substring of '3180'. A two-digit
    run is ~1-in-100 and collides with almost anything. Four digits is ~1-in-10,000 and
    is safe alone; anything shorter must be corroborated by the letters.
    """

    CARS = ["EEB-80", "LNV-94", "HVA-77", "ZVH-337", "RDJ-9640", "DJS-7842"]

    def test_the_sunny_no_longer_steals_a_short_plate(self):
        self.assertIsNone(confirm_plate("SANNYL318AV0", self.CARS))
        self.assertIsNone(confirm_plate("L918410", self.CARS))

    def test_a_two_digit_plate_is_never_confirmed_on_digits_alone(self):
        # '80' really is inside '3180' — the digit test passes and must not be enough.
        self.assertTrue("80" in "3180")
        self.assertFalse(read_matches_plate("3180", "EEB-80"))

    def test_a_short_plate_IS_confirmable_when_the_letters_agree(self):
        """The rule tightens short plates, it does not ban them."""
        self.assertTrue(read_matches_plate("EEB80", "EEB-80"))
        self.assertTrue(read_matches_plate("XEEB80X", "EEB-80"))

    def test_a_short_plate_with_CONTRADICTORY_letters_is_refused(self):
        self.assertFalse(read_matches_plate("SANNY80", "EEB-80"))

    def test_four_digit_plates_still_bind_on_digits_alone(self):
        """The mush-letter reads must keep working — that was the whole point."""
        self.assertEqual(confirm_plate("9640R0J", self.CARS), "RDJ-9640")
        self.assertEqual(confirm_plate("78420J8", self.CARS), "DJS-7842")
