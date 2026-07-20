"""
Confirm a parked car's plate by READING it, not by inferring it.

Why this exists
---------------
At the slot, appearance-based ReID cannot identify a car. Measured on 2026-07-11:

    RDJ-9640's own parked view vs its OWN gate references : 0.583
    ...vs a DIFFERENT car's references                    : 0.634   <- ranks FIRST

The ranking is INVERTED, so no threshold recovers the plate — a looser bar binds the
WRONG one. And the non-appearance fallback (acquisition by elimination) bound DJS-7842
to a stranger's Nissan Sunny that had merely been parked, unidentified, for hours.

OCR does not have this problem: it reads the characters off the car. The 1080p stream
switch made that viable — at 720p the plate was ~50x18px and OCR returned nothing; at
1080p it returns::

    RDJ-9640 parked in B10_CTO  ->  '9640RDJ5'

Noisy and re-ordered, but the plate is unmistakably there.

The contract
------------
``current_plate`` must be CORRECT or NULL — never wrong. A slot reading "(none)" is a
visible gap; a slot reading a stranger's plate is a liability that propagates silently
into billing and occupancy. So every rule below is written to ABSTAIN when unsure:

* the candidate's letters must appear CONTIGUOUSLY in the OCR letters, and its digits
  CONTIGUOUSLY in the OCR digits (order within each run is preserved, so 'RDJ'+'9640'
  matches '9640RDJ5' even though OCR shuffled the runs);
* extra OCR characters are tolerated (badges like 'SUNNY', partial Arabic glyphs,
  stray '5's) — they are noise, not evidence against;
* MISSING characters are fatal: every letter and every digit of the plate must be read;
* if TWO candidate plates match the same read, we abstain. A near-miss is not a vote.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

# A plate needs enough signal to be worth trusting. Two letters and two digits is the
# floor; below that, random badge text starts matching by chance.
_MIN_LETTERS = 2
_MIN_DIGITS = 2
# At/above this many digits, the digit run alone identifies the car (~1-in-10,000 among
# the handful of cars inside). BELOW it, the letters must corroborate: 'EEB-80' has only
# the digits '80', and '80' sits inside '3180' � which is how a Nissan Sunny's read
# ('SANNYL318AV0') "confirmed" EEB-80 and put a stranger's plate in B25 on 2026-07-12.
_STRONG_DIGITS = 4

# A STORED plate at this facility: 2-4 letters, then 1-4 digits, dash optional.
# Anchors the ARRANGEMENT, not just the run lengths — see is_plausible_plate.
_CANONICAL_PLATE = re.compile(r"^[A-Z]{2,4}-?[0-9]{1,4}$")


def normalize(text: Optional[str]) -> str:
    """Upper-case, strip everything that is not A-Z0-9 (separators, Arabic glyphs)."""
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def split_runs(text: str) -> Tuple[str, str]:
    """(letters, digits) of a normalized string, each in original order."""
    norm = normalize(text)
    return (
        "".join(ch for ch in norm if ch.isalpha()),
        "".join(ch for ch in norm if ch.isdigit()),
    )


def is_plausible_plate(plate: Optional[str]) -> bool:
    """Could this string be a real vehicle plate at this facility?

    A CANDIDATE FILTER, not a reader. Saudi civilian plates are 2-4 letters and
    2-4 digits; anything else is an ANPR artefact, and an artefact in the
    candidate set can only ever subtract. Measured on the 2026-07-20 entry feed,
    the gate minted these as *vehicles* — owner "Unknown", one sighting each:

        36663XN   77842SJ   66466RA   87281EJ   666EIAI

    ``77842SJ`` is a misread of DJS-7842 and ``36663XN`` of XRD-6663; ``36663XN``
    was bound to slot B2 and would have been the plate shown on that slot's
    intrusion alert. Since ``confirm_plate`` can only ever return a member of the
    candidate set, dropping implausible members here makes it structurally
    impossible for one to reach ``current_plate`` — which is the "CORRECT or
    NULL, never wrong" contract the slot identity path is built on.

    Deliberately shape-only: NO registry lookup, so a genuinely new visitor with
    a well-formed plate passes untouched. Validated against the live facility
    2026-07-20 — of 55 registered vehicles this rejects 0, and of the 15
    unknown-owner rows it keeps the 12 that are real unregistered visitors
    (BAS-6646, DJS-7842, ...) and drops only the artefacts.

    The order matters, not just the counts: a stored plate is letters-then-digits
    (``RGR-6466``, ``HDU-7``), while every artefact we have seen is digits-first
    (``36663XN``, ``77842SJ``, ``666EIAI``). Counting runs alone would admit
    ``666EIAI``; anchoring the arrangement does not.
    """
    if not plate:
        return False
    return bool(_CANONICAL_PLATE.match(str(plate).strip().upper()))


def read_matches_plate(ocr_text: Optional[str], plate: Optional[str]) -> bool:
    """Does this OCR read CONFIRM this plate?

    Calibrated against what the OCR actually produces on Saudi plates at 1080p, where the
    Latin characters sit small beneath the Arabic. Every read of the same two cars::

        RDJ-9640  ->  '9640RDJ5'   letters clean
        RDJ-9640  ->  '9640R0J'    the D came through as a ZERO
        RDJ-9640  ->  'A9640R0J'   ...plus a phantom 'A'
        DJS-7842  ->  '78420JS'    only 'JS' of 'DJS'
        DJS-7842  ->  '78420J8'    the S came through as an EIGHT
        DJS-7842  ->  '78420J5'    ...or as a FIVE

    The digits are exact in all six. The letters are mush: D->0, S->8, S->5, phantom
    prefixes. So an earlier version of this function that required the letters to agree
    REJECTED correct reads — it refused '9640R0J' for RDJ-9640 because 'RJ' is not
    'RDJ'. Letters cannot carry the decision.

    So: THE DIGITS ARE THE IDENTIFICATION. The plate's whole digit run must be read, in
    order, exactly. Safety comes not from the letters but from UNIQUENESS — see
    :func:`confirm_plate`. A 4-digit run is ~1-in-10,000, and it is matched only against
    the handful of cars VA believes are inside; if two of them could explain the read, we
    bind neither. The Nissan Sunny that the elimination path wrongly claimed reads
    '918410' — no car inside has those digits, so OCR leaves its slot NULL, correctly.
    """
    p_letters, p_digits = split_runs(plate)
    o_letters, o_digits = split_runs(ocr_text)
    if not o_digits:
        return False
    # Absolute floor: a one-digit plate cannot be identified by reading, full stop.
    if len(p_digits) < _MIN_DIGITS:
        return False

    # THE DIGITS ARE THE IDENTIFICATION. They must be read in order and in full.
    if p_digits not in o_digits:
        return False

    # A SHORT digit run is not enough on its own. 'EEB-80' has just TWO digits, and '80'
    # is a substring of '3180' — so on 2026-07-12 the read 'SANNYL318AV0' off a Nissan
    # Sunny "confirmed" EEB-80 and stamped a stranger's plate into B25. Four digits is
    # ~1-in-10,000 and safe alone; anything shorter must be CORROBORATED by the letters.
    if len(p_digits) >= _STRONG_DIGITS:
        return True

    # Short plate: the letters must also agree. A partial read corroborates ('JS' of
    # 'DJS'); a different one refutes. This keeps short plates confirmable where the
    # letters are legible, and NULL where they are not — never wrong.
    if not o_letters or len(p_letters) < _MIN_LETTERS:
        return False
    return (o_letters in p_letters) or (p_letters in o_letters)


def confirm_plate(
    ocr_text: Optional[str],
    candidates: Iterable[Optional[str]],
) -> Optional[str]:
    """The ONE candidate plate this read confirms, or None.

    None on: no read, no match, or MORE THAN ONE match. Ambiguity is never resolved by
    picking a winner — that is precisely how the elimination path put DJS-7842 on a
    Nissan Sunny. If two plates could explain the read, we do not know which car this is.
    """
    if not normalize(ocr_text):
        return None

    hits: List[str] = []
    for plate in candidates:
        if plate and read_matches_plate(ocr_text, plate) and plate not in hits:
            hits.append(plate)

    return hits[0] if len(hits) == 1 else None
