# Entry OCR advisory monitoring

## Current policy

VA OCR cannot override the ANPR plate during entry confirmation. HikCentral
retains its existing independent-source behavior, including the existing
Hik-only fallback. OCR from vehicle images associated with the attempt or
evaluated crossing family is diagnostic only: it cannot block a session,
rename its plate, or correct ANPR output.

The physical-witness requirement, source-time causality, ReID score and
uniqueness margins, and ANPR/HikCentral disagreement handling are unchanged.

## Decision-log evidence

Every `plate_consensus` record uses `record_v: 2` and includes an
`ocr_advisory` block:

- `anpr` contains the plate and reported confidence being monitored.
- `reads` lists every OCR frame from the causally eligible attempt and the
  evaluated producer family. It preserves the raw OCR text, exact confidence,
  state, camera, source role, evidence ID, and whether it differs from ANPR.
- `correction_candidates` is the subset of readable OCR reads that differs
  from ANPR. It is a list for offline review, not a decision input.
- `correction_applied` is always `false` in this monitoring release.

No high-confidence threshold has been introduced. A later correction policy
requires labelled facility evidence and a separately reviewed decision.

## Current limitation

This policy removes the former ramp-observation OCR veto. If an earlier pending
ANPR identity passes the existing ReID and witness checks for a visually similar
car, VA can confirm it before the later car's correct ANPR attempt arrives. OCR
will record the disagreement but will not retain the crossing for that later
attempt. The regression fixture covers a CAM-03 score of `0.92` against that
fixture's configured `0.75` ReID threshold. This is the explicit tradeoff of non-blocking
OCR monitoring; no replacement matching gate is introduced here.
