# Slot identity — measurements and findings, 2026-07-13

Written while investigating "why do occupied slots have no plate". Several long-held
beliefs in this repo turned out to be **false**, and the actual root cause was not in the
matching layer at all. Read this before doing any more work on slot identity.

---

## 1. ReID at the slot is NOT inverted. That claim is dead.

The code and comments repeatedly assert that ReID at a parked slot ranks the *wrong* car
first (0.583 for the right car vs 0.634 for a different one, measured 2026-07-11). That
measurement was taken on the **retired `PS_carMatching` model** with a gallery that held
**only gate photos**. It is no longer true and must not be used to justify design.

Re-measured 2026-07-13 against the live 34-car / 362-ref gallery, through production's
own `_best_weighted_score`, querying with **real parked-pose slot crops** (query crop
withheld — an early version of this harness left it in and produced a meaningless
self-match at cosine 1.0):

| regime | rank-1 | recall@5 |
|---|---|---|
| cold — car has only gate/ramp/entry refs | **87.8%** | **97.7%** |
| warm — car has a parked ref from this camera | **98.5%** | 100% |

Scored against all 34 cars regardless of presence, so 87.8% is a **floor** — production
`reid_shortlist` only scores active sessions.

**Slot identity is a RANKING problem**: the right car is in the top-5 ~98% of the time
and first ~88% of the time. It is not a "no signal" problem.

## 2. `slot_camera_ref_weight` = 0.80. Not 0.6, and NOT 1.0.

`_best_weighted_score` weights every non-gate reference by `secondary_camera_weight`
(0.6) — including a car's **own parked pose**, taught by `save_parked_reference`. At 0.6
that pose (cosine ~0.9 → 0.54) loses to a *stranger's* full-weight gate photo, so a car
could never be recognised on a return visit.

Full weight is also wrong. The uplift is **symmetric**: it lifts every candidate's
same-camera refs, including the regulars who park at that camera daily. Same-view
similarity between *different* cars (~0.66) exceeds cross-view similarity on the *same*
car (~0.55) — suppressing that is exactly what the discount was for.

Swept on the live gallery (263 real parked-pose queries, query withheld):

| same-cam weight | WARM rank-1 | COLD rank-1 |
|---|---|---|
| 0.60 (old) | 98.5% | 87.8% |
| **0.80 (chosen)** | **100%** | **87.8%** |
| 1.00 (naive) | 100% | 87.5% |

0.80 takes the whole warm gain at zero cold cost, mid-plateau. `tests/test_slot_candidate_ranking.py`
asserts that at 1.0 the wrong car wins. **Re-derive on any ReID model swap.**

## 3. Zoning is LIVE, not inert.

`config.yaml`'s `cameras:` block is a **stale 14-camera fallback roster**;
`load_cameras_from_db()` replaces it wholesale, and `cameras.area` is populated in the DB
for all 22 non-ground cameras. Area/route features are real and available. Do not "enable"
zoning — it is on.

## 4. Ramp boundaries were on the wrong cameras (FIXED in the DB).

Two `boundaries` rows keyed `camera_id` by the **`cameras.id` integer PK** (`'9'` = CAM-07,
`'11'` = CAM-09). Someone later re-created the *identical polygons* on the cameras those
numbers *looked* like, landing them on the wrong ones — CAM-09 (a **B2-A** camera) was
flying a **`B1-A → RAMP-DN`** band, so any B2 car crossing it was marked `IN_TRANSIT` from
`RAMP-DN`, dropped from B2-A's bucket, and became an eligible ReID candidate **in B1, on
the other floor**.

Confirmed by snapshot: **CAM-07** frames the top of the down-ramp (B1, ramp mouth left);
**CAM-09**'s own HUD reads `B2-ENTRANCE` and it frames the ramp foot (drain grate).

Fixed: repointed the two named rows to the string camera IDs, deleted the two misplaced
duplicates. 13 → 11 boundary rows. **Do not run `tools/manage_areas.py push`** — the YAML
`areas:` block is stale (ramp transit 20s vs the DB's 60s) and would downgrade the live
topology.

## 5. Phantom plates: ANPR misreads + stale test fixtures.

VA hydrates a session for every open `parking_sessions` row. **18 of 51 "cars inside" had
no gallery folder** because they are not cars:

- ANPR misreads: `BJA-7842`, `DJA-7842` (next to the real `DJS-7842`), `AJA-7642`, …
- Stale **test fixtures** never closed out: `RUH-1010` … `RUH-6060` (same family as the
  `RUH-9999`/`RUH-8888` that produced the only two historical intrusion alerts).

They cannot be matched — there is no photo of a car that does not exist — but they
**collide**. `confirm_plate()` matches on the **digit run** and abstains on ambiguity, so
those two phantoms turned a *perfect* read of `DJS-7842` into a three-way tie and the slot
stayed NULL.

The **ReID path was already immune** (a phantom has no vector, so it is never scored);
the collision only bit the `plates_inside()` fallback. `candidates_require_appearance_evidence`
now requires a gallery reference before a plate may be an OCR candidate. Live: 51 → 33
candidates, no real car dropped, and `B27` bound `DJS-7842` on the next restart.

**The `RUH-*` rows should be closed out at the source (PMS-AI owns that table).**

---

## 6. THE ROOT CAUSE: a per-frame clear destroyed every binding it could not re-derive.

> **CORRECTION.** This section originally blamed camera stream drops. **That was wrong**,
> and the retraction is below the real cause. I include the mistake because the reasoning
> that produced it is seductive and someone will reproduce it.

`_resolve_locked_plate` (`engine_runtime.py`) runs on **every frame** of an occupied slot
and used to WIPE any plate it could not re-derive on that frame. But the thing it asks —
`try_link_to_slot` — returns `None` on **every** frame: it is disabled on its third
statement by `slot_plate_requires_ocr: true`.

So the resolver destroyed bindings the instant they were made. OCR binds survived only by
accident: they lock, and a locked slot takes an early return before reaching that branch.
Every *provisional* binding — appearance-only binds, restart-restored plates — was erased
on the next frame. B19 was bound to ERS-7949 and destroyed **three times in six seconds**.

**Measured, before vs after removing the clear (same cameras, same day):**

| | before | after |
|---|---|---|
| B26 re-bound | **36x** | 3x |
| B11 re-bound | **36x** | 3x |
| B18 re-bound | **33x** | 3x |
| vacancy events | many | **0** |
| reconnects | happening | **still happening** |

**THE INVARIANT (operator's rule, now enforced): a plate dies ONLY when the slot goes
VACANT.** Absence of a re-derivation is not evidence that the car left. The slot is still
OCCUPIED — that is the only fact the resolver actually has.

### Retraction: camera drops do NOT delete plates

The original claim was: stream drops → no frames → no detections → `vehicle_present=False`
→ `confirm_leave_frames` elapses → VACANT → `unlink_slot()` → plate wiped.

**That chain cannot happen.** `RTSPCamera.read()` (`src/camera_manager.py`) returns
`self._latest_frame`, which is **never cleared on failure**. During a reconnect the grabber
keeps serving the LAST GOOD FRAME, `next_frame()` never skips the camera, the car is still
visible, and the slot never vacates. Confirmed empirically: reconnects continued at the
same rate after the fix, with **zero** vacancies and zero plate loss.

What I misread: "17 binds, only 12 survived" and "B24 bound four times" are fully explained
by the per-frame clear. And "every slot on a camera vacates in the same instant" was wrong
— the vacancies were spread across 2-3 *different* cameras, and the ground floor (G3/G8)
flaps for unrelated reasons.

### A real camera bug remains — but it is the OPPOSITE failure

A dead stream serves its last frame **forever**, so a car that leaves during an outage
stays "parked" indefinitely, and the engine keeps running the detector on a frozen image.
That is stale occupancy, not plate loss. **Unfixed.**

---

## 7. Data gaps that block the business goal

- **`vehicles.title` is blank for every car.** `_is_named_slot_vehicle_allowed` compares
  it against `parking_slots.reserved_for`, so it returns False for *everyone* — including
  the rightful owner. Intrusion detection cannot work, and correctness of a C-level bind
  **cannot even be verified**. Needs a plate ↔ title mapping.
- Only **9 of 61** reserved slots have `reserved_for` set.
- **Ground floor (CAM-00/01/02) has ReID and plate matching disabled by design**
  (`is_reid_disabled_floor`). Those slots can never be plated as architected. Out of scope
  per the 2026-07-13 decision.

## 8. Remaining OCR failure modes (after the above)

- **Plate not in frame** — B1_CRO (455 attempts, 0 reads), B13_COO, B17, B14. Side-on
  mounts. Needs the ReID-solo accept path.
- **HUD / badge text read as a plate** — B22, B25, B28 read `CAM13BZP` (the camera's own
  burned-in overlay) and `TOYOTA…L772HRS` (a rear badge) hundreds of times. The
  `hud_*_mask_ratio` settings are not masking. Contained bug; likely recovers 3 slots.
- The identify pass is capped at **12 attempts per park** and does not retry once
  exhausted, so a slot that fails stays NULL until the car leaves.
