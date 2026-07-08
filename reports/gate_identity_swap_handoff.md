# Hand-off: night ANPR misread → gallery contamination → wrong Re-ID match

**Branch:** `feat/decouple-detection-percamera-tracker` · pushed through `ceae199`

## The incident
A car lingered at the B1 ANPR gate at night; the ANPR server misread its plate
twice, so one physical car "entered with two plates". Because gate identity was
decided by **plate string + arrival order only — with no appearance/ReID gate on
binding or gallery admission**, the two misreads became two identities/galleries,
and the **next** car FIFO-inherited a stale misread plate and had its crops merged
into that plate's gallery → it matched the wrong plate.

**Master root cause:** nothing in the entry/gallery pipeline enforced "this crop
looks like the identity it's joining." Fixes below add that guardrail (open-set
novelty rejection — the standard ANPR/Re-ID technique) plus source- and
window-side defenses.

## What shipped (5 commits, all with regression tests; 284 tests green)

| Commit | Fix | Closes |
|---|---|---|
| `4668002` | Slot probe point `(y1+y2)/1.5 → y2`; overlap divisor → det-box | Occupancy mis-attribution (separate audit) |
| `c59f5a7` | Gallery viewpoint-diversity + MERGE; **appearance-consistency gate** (in-mem + on-disk); ensemble ReID floor; single-candidate floor; type-classifier OV2026 | Contamination; appearance-blind bind; viewpoint-thin galleries |
| `49cc3ee` | FIFO pending-plate **bind TTL 30s → 10s** | Stale-plate inheritance window |
| `5511740` | **ANPR entry OCR-confidence gate** | Two-plate genesis (source side) |
| `ceae199` | **Cross-session identity reconciliation** (one car ↔ two plates) | Duplicate identities |

## ⚠️ DEPLOYMENT — you MUST mirror config into the client box
`config.yaml` is **gitignored**, so none of the operative values below shipped in
the commits. The committed **code defaults the new guards to OFF (0.0)** for
safety. Set these in the client deployment config (and the tracked
`config.example.yaml` if you use it as the template):

| Key | Code default | Recommended prod | Notes |
|---|---|---|---|
| **Matching thresholds** (`b1_zone`, `b1_anpr`, `global_*`, `reattach_*`, `reid_solo_confirm`, `ocr_marginal_*`, `lock_confidence`) | old values | **recalibrated set** (see config.yaml) | Recalibrated for `models/PS_carMatching`. Cross-camera bars kept low (0.43) for handover. |
| `single_candidate_min_reid` | 0.0 (off) | **0.35** | Rejects appearance-blind lone-candidate binds. |
| `gallery_min_identity_similarity` | 0.0 (off) | **0.35** | Rejects foreign crops from an identity's gallery (contamination gate). |
| `use_multishot` / `multishot_ref_top_k` | false / 3 | **true / 3** | Multi-view gallery references on every camera. |
| `anpr_min_accept_confidence` | 0.0 (off) | **0** until ANPR server sends confidence, then ~0.8–0.9 | Holds low-confidence night misreads. |
| `identity_reconcile_min_similarity` | 0.0 (off) | **0** until validated, then ~0.75 | Merges two-plate duplicates. Enabling closes real sessions — validate first. |
| `identity_reconcile_window_seconds` | 60 | 60 | Gate-dwell window for the above. |

Also note: a **DB Config row can override `config.yaml`** — verify the client DB
isn't pinning old threshold values.

## Calibration TODO (the research's key caveat)
The two `0.35` floors and the `0.75` reconcile floor are **conservative defaults,
not tuned values**. Calibrate on a labelled **FAR/DIR** curve for the gate
cross-view: too high rejects legitimate cross-view matches; too low lets
contamination/false-merges through. Cross-view same-car scores are inherently low
(different camera/lighting), so do **not** reuse `b1_zone`.

## Remaining / follow-ups
- **Multi-frame plate voting** belongs to the external ANPR server (it owns the
  OCR + per-frame reads). The VA only receives final plate strings; it can only
  gate confidence + coalesce. Ask the ANPR vendor to per-character majority-vote
  across the dwell and send one plate + confidence.
- **Zone-occupancy pending-plate clearing** (clear on gate-zone-empty) needs a
  presence signal (loop/radar/virtual-loop or a CAM-03 presence proxy) not
  currently plumbed. The 10s bind-TTL is the interim mitigation.
- **`_reconcile_duplicate_identity`** only reconciles pre-park confirmed
  sessions; it never touches a parked/locked identity. Validate before enabling.

## Related prior context
See `reports/tracking_matching_slot_audit.md` (the earlier tracking/matching/slot
audit) and the recalibration bench in `data/facility_top20/eval_report_ps_carmatching.json`.
