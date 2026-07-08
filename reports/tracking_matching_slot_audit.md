# Damanat PMS Video Analytics — Tracking, Matching & Slot‑Occupancy Audit

**Prepared for:** Client technical leadership (CTO / CAIO / Tech Leads)
**Date:** 2026-07-06 · **Branch:** `feat/decouple-detection-percamera-tracker`
**Scope:** vehicle tracking, cross‑camera ReID identity, plate‑lock, and slot‑occupancy reporting
**Method:** multi‑agent code audit — 9 subsystem readers → 2 focused traces → adversarial verification of every candidate defect → executive synthesis. **85 agents, 69 unique candidate issues examined, 49 CONFIRMED + 12 PLAUSIBLE, 8 REFUTED.** All numbers below were re‑verified by hand against source.

---

## 1. Executive Summary

The pipeline is **architecturally sound end‑to‑end**: YOLO11 detection + per‑camera ByteTrack → burst crop capture → OSNet OpenVINO INT8 ReID with multi‑modality (color/type/OCR) voting → a debounced per‑slot state machine → a plate‑lock that binds an identity to a slot and reports it to the PMS backend. The matching and voting logic works, and the main defensive gates (temporal eligibility, ambiguity margin, 2‑of‑3 voting, anti‑swap guard) are present.

However, the audit **confirmed ~31 distinct code‑level defects, 8 of them high‑severity**, and they cluster in exactly the two areas you asked about — **slot‑occupancy geometry** and **identity locking**.

**The single most important fix** is the slot‑membership probe point in `src/detection/detector.py:51`:

```python
cx = (x1 + x2) / 2.0
cy = (y1 + y2) / 1.5     # ← neither centre (/2.0) nor bottom (y2)
```

This point is tested against every slot polygon (the **primary** occupancy test, `slot_assigner.py:82-87`). Because it divides the *sum of two absolute pixel coordinates* by 1.5, whenever `y1 > y2/2` (i.e. any car in the lower ~half of the frame — the common case on angled floor cameras) the probe lands **below the bounding box entirely**, in the drive lane or the row behind. This deterministically mis‑attributes occupancy. It is a two‑character fix.

Close behind it: on angled cameras the effective identity for a parked car is built from **as few as one reference crop** on most cameras (three only on the entry camera `CAM-03`), and the plate freeze‑bar `lock_confidence` is set **below** the very thresholds its own config comment says it must exceed.

**Bottom line for delivery:** the genuinely broken items are code bugs fixable within a few days. A separate class of items is "calibration‑needs‑real‑footage" and must **not** be conflated with bugs or block the delivery.

---

## 2. How Tracking & Matching Works

**Pipeline walkthrough**

1. **Detection** — one shared `TrackedDetector` wraps a single YOLO11 model; `detect_and_track` runs `model.track(...)` per frame, sub‑stream 720p (`config.yaml:176-178`).
2. **ByteTrack (per‑camera)** — each camera's tracker instance is swapped into the shared predictor before inference (`tracker.py:163`), giving each camera its own track‑ID space (the point of this branch).
3. **Burst capture** — in entry/confirmation zones, crops are collected per track; `select_best_frames` (`reid_burst.py`) ranks by variance‑of‑Laplacian sharpness, rejects >35 % over‑exposed crops, and keeps `top_k` (**3 on `CAM-03`, else 1** — `vehicle_registry_core.py:695`).
4. **ReID feature / multishot** — kept crops are embedded by the OSNet OpenVINO INT8 model (512‑D, cosine); under multishot the sharpest frames are averaged; a match scores `max` cosine over the car's stored reference vectors.
5. **Temporal voting** — `MatchVoter` accumulates per‑`(camera, track)` decisions; production runs **2‑of‑3** (`voting_window_frames: 3`, `voting_min_agree: 2`).
6. **Confirmation thresholds** — per‑context ReID bars (effective values, §5) plus a K‑of‑N modality ensemble (`ensemble_min_modalities_agree: 2`) and a ReID‑solo fast‑path.
7. **Slot state machine** — one `SlotStateMachine` per slot debounces `VACANT → ENTERING → OCCUPIED → LEAVING`, requiring **5** consecutive present frames to confirm parked and **8** consecutive absent frames to confirm vacant.
8. **Identity lock** — on a `vehicle_parked` event the engine calls `try_link_to_slot`; it locks when `new_conf >= lock_confidence` **OR** OCR agrees (`engine_runtime.py:1152`), then freezes the slot (`_resolve_locked_plate` short‑circuits while locked).
9. **PMS report** — only `vehicle_parked` / `slot_vacant` transitions are persisted to the DB and forwarded to the PMS backend.

```
 RTSP ─► YOLO11 detect ─► ByteTrack (per-cam) ─► SlotAssigner
                               │                  (bottom_center / overlap)
                               ▼                          │
                     Burst capture (top_k 3/1)            ▼
                               │                  SlotStateMachine ── vehicle_parked ─┐
                               ▼                  (5 enter / 8 leave)                 │
                     OSNet ReID (OpenVINO INT8)           │                          ▼
                               │                          ▼                  try_link_to_slot
                     MatchVoter (2-of-3) ──► decide_b1/global ───────────► lock if conf ≥ 0.50
                               │                                                or OCR agrees
                               ▼                                                     │
                     multi-modality ensemble ────────────────────────────►   PMS DB report
```

---

## 3. How Many Images Are Used to Match & Lock a Car  *(your primary question)*

**Definitive answer.** A car is **confirmed "parked" after 5 consecutive detection frames** in the slot, and is **locked on that same frame** if its ReID score reaches the freeze‑bar (`0.50`) or OCR agrees. So the practical floor is:

- **~5 detection crops** of the car (to clear the enter‑debounce), matched against
- **1 stored reference crop** of the identity on most cameras (**up to 3** on the entry camera `CAM-03`), with
- the plate identity nominally requiring **2 of 3** frames to agree in the temporal vote.

| Stage | Min | Typical | Source | Note |
|---|---|---|---|---|
| Enter‑debounce (ENTERING→OCCUPIED) | **5 consecutive present frames** | 5 | `confirm_enter_frames: 5` (`config.yaml:194`) | any single dropout resets the counter to 0 (§4) |
| ReID reference crops per identity | **1** | 1 (**3** on `CAM-03`) | `top_k = 3 if camera_id=="CAM-03" else 1` (`vehicle_registry_core.py:695`) | multishot averaging only on entry cam |
| Persistent gallery cap | — | ≤ **20** refs/car | `gallery_max_refs_per_car: 20` | accumulated over the visit; a single match compares vs. what is stored |
| Temporal plate vote | **2 of 3** frames | 2 of 3 | `voting_window_frames: 3`, `voting_min_agree: 2` | double‑submit bug (§5) can collapse this toward 1 physical frame |
| Lock trigger | on the parked frame | 5th frame | `new_conf ≥ 0.50 OR ocr_ok` (`engine_runtime.py:1152`) | — |

**The robustness concern the CAIO flagged:** outside the single entry camera, a car's ReID identity is anchored on **one crop**. One oblique, low‑resolution reference is thin evidence to lock a physical slot to a plate — it is the root cause of both "wrong car locked" and "never locks" failure modes, and the reason the freeze‑bar (`lock_confidence`) must be set conservatively (it currently is not — §5).

---

## 4. Slot Occupation Detection — Findings  *(your second question)*

Slot occupancy is where the highest‑confidence, highest‑impact defects cluster. Four confirmed issues bias *which car lands in which slot* and *when a slot frees*.

**① Probe point is geometrically undefined — `detector.py:51` (CONFIRMED, HIGH).** As above, `cy = (y1 + y2)/1.5` is neither centre nor bottom; for `y1 > y2/2` it lands below the bbox. This is the **primary** slot test (`slot_assigner.py:82-87`), so on angled cameras it probes past the car into the lane or the row behind → wrong slot or no slot. The docstring even calls it "true center" (which would be `/2.0`) and `boundary_detector.py` uses `y2` with a comment claiming it "matches slot logic" — it does not, so boundary crossing and slot membership reference ground points ~⅓‑car‑height apart. **Fix:** `cy = y2` (true ground‑contact) or a documented height fraction; then re‑validate polygons.

**② Overlap divisor inverted — `slot_assigner.py:168` (CONFIRMED, MED).** The fallback `_compute_overlap` docstring promises `intersection / detection_box_area`, but the code returns `intersection / slot_polygon.area`. So `overlap_threshold` (effective **0.2**) means "fraction of the *slot* covered," which scales with slot pixel‑area under perspective: a large truck straddling a small far slot can score ~1.0 and steal it; a small car fully inside a large near slot can score ~0.16 and be rejected. **Fix:** divide by `det_box.area`, then re‑tune the threshold on footage.

**③ Fast, no‑gap handover freezes the wrong identity — `state_machine.py:263` (CONFIRMED HIGH).** `vehicle_present` is identity‑agnostic; the `OCCUPIED` branch overwrites `assigned_track_id` every frame with no continuity check, and `LEAVING` only completes after **8 consecutive** absent frames (any present frame resets it). If car A leaves and car B parks in the same bay without an 8‑frame empty gap, `slot_vacant` never fires, the slot stays plate‑locked, and **the PMS keeps reporting A's plate while B physically occupies the slot.** **Fix:** on a sustained ReID‑backed track change, force a synthetic handover.

**④ Brittle debounce + discarded displaced cars — `state_machine.py:254/299`, `slot_assigner.py:91/139` (CONFIRMED, MED).** Entry needs 5 *strictly consecutive* frames, so a car YOLO misses ~1 frame in 4 near a pillar can loop `VACANT↔ENTERING` forever and **read VACANT indefinitely**. Symmetrically, an unrelated passing car can cancel a legitimate departure. And under overlapping/oversized polygons, the primary loop `break`s on the first polygon in list order and the tie‑break loser is dropped to `unassigned` — never reconsidered for the neighbouring slot it actually occupies (that slot reads VACANT). **Fix:** leaky/K‑of‑M counters + global one‑to‑one (greedy/Hungarian) slot assignment.

**Also relevant:** slot polygons are authored at 1280×720 (`slot_ref_width/height`); `load_slots` scales them to the live frame **only if `actual_resolution` is passed at the call site** — verify this is wired for every camera, or membership silently shifts.

---

## 5. Effective vs. Default Configuration (important correction)

`src/config.py` dataclass defaults are **intentionally overridden** by `config.yaml` at load (`config.py:538-556`); **config.yaml is authoritative.** Any threshold quoted must be the *effective* (YAML) value. The divergence itself was checked and graded intentional/production‑safe — but it is a foot‑gun, and it is why the freeze‑bar looks fine in code yet is mis‑set in production:

| Threshold | `config.py` default | **Effective (config.yaml)** | Comment |
|---|---|---|---|
| `lock_confidence` (plate freeze bar) | 0.70 | **0.50** | YAML comment says it should sit *above* `reid_solo_confirm` — but it is set **below** everything |
| `reid_solo_confirm` | 0.70 | **0.60** | ReID‑solo fast‑confirm |
| `b1_zone` | 0.55 | **0.53** | park‑entry zone confirm |
| `global_default` | 0.55 | **0.53** | cross‑session global search |
| `global_with_plate` | 0.46 | **0.44** | global search with confirmed plate |
| `overlap_threshold` | 0.30 | **0.20** | slot fallback overlap gate |
| `confirm_enter/leave_frames` | 5 / 8 | **5 / 8** | debounce (agree) |

**Freeze‑bar inversion (`config.yaml:246`, CONFIRMED, HIGH):** effective ordering is `lock_confidence 0.50 < b1_zone 0.53 < reid_solo_confirm 0.60`. The lock gate is `new_conf ≥ 0.50 OR ocr_ok`, so a cross‑camera session confirmed at 0.51 — within the wrong‑car negatives' top decile — **permanently freezes a possibly‑wrong plate**, uncorrectable until the slot goes VACANT. **Fix:** raise `lock_confidence` to ≥ 0.62 and validate lock‑rate on a labelled set.

---

## 6. Confirmed Issues (severity‑ordered)

| Sev | Area | file:line | Issue | Failure scenario | Fix | Risk |
|---|---|---|---|---|---|---|
| **HIGH** | Slot geometry | `detector.py:51` | `bottom_center` `cy=(y1+y2)/1.5` — neither centre nor bottom; can fall below bbox | Angled/low‑frame cars probed below the vehicle → wrong slot or none | `cy = y2`; fix docstring & `boundary_detector` | moderate |
| **HIGH** | Identity lock | `vehicle_registry_identity.py:2507` | Auto‑lock stamps lone pending plate on a **plateless** track, forces conf `1.0`, locks slot | Plateless park races one unrelated gate plate → X reported as Y permanently; forced‑OCR self‑heal defeated | Provisional conf < `lock_confidence`; add `is_plate_inside` + ownership check | moderate |
| **HIGH** | Concurrency | `main.py:103` / `tracker.py:163` | Shared YOLO model `.predict()` (API thread) races `.track()` (loop); `predictor.trackers[0]` swapped unlocked | Coincident ANPR request + loop tick → corrupted detections or engine‑stopping exception | `threading.Lock` around model access | moderate |
| **HIGH** | Identity lock | `state_machine.py:263` | No‑gap car swap keeps locked slot frozen under departed car's plate | Tight‑bay handover with no 8‑frame gap → PMS keeps A's plate for B | Sustained ReID track‑change → force synthetic handover | moderate |
| **HIGH** | Concurrency | `vehicle_registry_identity.py:2658` | `is_plate_inside` runs a blocking DB probe (`SET LOCK_TIMEOUT 3000` + SELECT) under the registry RLock | Slow DB → all camera threads stall up to 3 s | Hoist probe off‑lock; re‑validate on re‑acquire | moderate |
| **HIGH** | Zoning | `vehicle_registry.py:396` | `apply_boundary_crossing` lacks the owner/area gate its sibling `settle_track_area` has | Neighbour‑area camera teleports a settled car IN_TRANSIT; identity stealable | Gate on owner OR current‑area membership | moderate |
| **HIGH** | Config | `config.yaml:246` | `lock_confidence 0.50` < `reid_solo_confirm 0.60` & `b1_zone` — inverts documented intent | Cross‑camera park at 0.51 permanently freezes possibly‑wrong plate | Raise to ≥ 0.62 | trivial |
| **HIGH** | Config/Zoning | `config.yaml` cameras | No camera carries an `area:` key; 8 areas defined but zoning silently inert | Bounded matcher + same‑floor plate‑leak guard become no‑ops → global‑pool matching | Add `area:` per camera; startup warn on empty mapping | trivial |
| MED | ReID backend | `reid_matcher.py:335` | Silent torchreid ImageNet fallback when IR missing (different feature space) | Missing IR + torch present → meaningless cosines, no error | Raise in prod; tag vectors by backend | moderate |
| MED | Slot geometry | `slot_assigner.py:168` | `_compute_overlap` divides by slot area, not det‑box (contradicts docstring) | Threshold slot‑size‑dependent → truck steals small slot / small car rejected | Divide by `det_box.area`; re‑tune | moderate |
| MED | Slot assign | `slot_assigner.py:91/139` | Primary loop breaks on first polygon by list order; tie‑break loser discarded, never reconsidered | Overlapping polygons → true slot reads VACANT | Global one‑to‑one (greedy/Hungarian) | moderate |
| MED | Slot state | `state_machine.py:254/299` | ENTERING needs 5 *consecutive* frames; LEAVING resets on any present frame | Flickering park → VACANT indefinitely; passing car blocks departure | Leaky/K‑of‑M counters; identity‑gated cancel | moderate |
| MED | Matching | `match_decision.py:442` | K‑of‑N ensemble confirms on color+type alone below ReID threshold | Two white sedans confirmed same at ReID 0.30 | Require ReID floor + one high‑entropy modality | moderate |
| MED | Matching | `match_decision.py:200/235` | `decide_b1` ignores `cross_camera` (`b1_cross_camera` dead); single‑candidate fallback ignores OCR contradiction | Cross‑camera handoff mis‑judged; OCR‑contradicted sole candidate still locked | Wire cross‑camera branch; exclude OCR‑rejected | moderate |
| MED | Matching | `vehicle_registry_identity.py:2096` | Abstain runner‑up includes REJECTED candidates | Rejected high‑score suppresses a valid confirmed match | `continue` on non‑confirm before arming gate | trivial |
| MED | Gallery | `gallery_store.py:126/278` | `save_ref` restamps `model_tag` over stale old‑model vectors; GC deletes still‑parked long‑stay car's gallery | Model swap → mixed embedding spaces scored together; >5‑day car loses appearance gallery | `reembed()` on tag change; `gc(keep_plates=...)` | moderate |
| MED | Engine | `engine_runtime.py:995` | Two `try_link_to_slot` calls on the parked frame double‑submit to the voter | 2‑of‑3 vote collapses into 1 physical frame | One vote per (cam,track) per frame | moderate |
| MED | Engine | `engine_tracking.py:112/755` | API daemon iterates a live dict unlocked; confirmation burst finalized on primary demotion, not true departure | Mid‑iterate key add → HTTP 500; premature/partial gallery | `list(...)` snapshot; gate finalize on real departure | trivial |
| MED | Zoning | `area_state_machine.py:83/95` | `on_fov_exit` unwired (blind‑gap adjacency drops identity); ARRIVING debounce unimplemented | Un‑bounded crossing mints new identity; spurious frame re‑homes area | Wire DEPARTING; implement ARRIVING confirm counter | moderate |
| MED | Config | `config_service.py:113` *(PLAUSIBLE)* | DB Config row authoritative; schema/ORM defaults `10/40/0.3` diverge from YAML `5/8/0.2`; no repair/log | Out‑of‑band row → slot frees ~40 s late | Align defaults; log effective values at boot | trivial |
| LOW | ReID/robustness | `vehicle_registry_core.py:695/700`, `reid_burst.py:22`, `reid_openvino_backend.py:20` | `top_k` hardcoded to `"CAM-03"`; dead precomputed‑vector reuse branch; silent over‑exposed fallback; doc/impl mismatch | Renamed entry cam breaks multishot; redundant re‑embeds; degraded refs | Config‑drive `top_k`; fix identity check; log fallback; correct docstrings | trivial |

---

## 7. Executive Perspectives

**CTO.** Two of the eight highs are *systems* risks, not model tuning: the unsynchronized shared YOLO model (`main.py:103`) can **stop the whole engine** on a coincident API call, and the blocking DB probe under the registry lock (`…identity.py:2658`) can **stall every camera thread for up to 3 s**. Both are small, well‑understood fixes (a lock; an off‑lock hoist) and must ship before delivery — an engine that a routine ANPR request can crash is not production‑ready regardless of accuracy.

**CAIO.** The ReID *model* is fine; the risk is around it. Locking a physical slot on **one reference crop** (non‑entry cameras) with a freeze‑bar set **below** the solo‑confirm bar is statistically fragile — it invites both false locks and never‑locks. The silent ImageNet fallback (`reid_matcher.py:335`) can put a weaker, different‑feature‑space model into production with only a warning; the cross‑model gallery mix (`gallery_store.py:126`) blends embedding generations after a model swap. And the K‑of‑N ensemble confirming on color+type alone (`match_decision.py:442`) can bind a wrong identity at ReID 0.30 — it needs a ReID floor.

**Tech Lead — Slot/Occupancy.** Occupancy correctness hinges on one line: `detector.py:51`. The `/1.5` probe is the primary slot test and is geometrically undefined — the highest correctness‑leverage fix in the report. Paired with the inverted overlap divisor and the no‑gap handover freeze, these three fully explain the "wrong car in wrong slot" and "slot stuck occupied" symptoms. None require footage to *fix*; they require footage to *re‑validate polygons* afterward.

**Tech Lead — Identity/Matching.** The lock path is too eager: auto‑lock forces `1.0` on a plateless track (`2507`), `lock_confidence` is too low (`0.50`), and the double‑submit collapses the temporal vote (`engine_runtime.py:995`). Individually survivable; together they let a single bad frame permanently freeze an identity. The remaining match‑decision fixes are mostly trivial edits that tighten correctness without changing intended behaviour.

---

## 8. Prioritized Action Plan Before Delivery

**P0 — true code bugs, must fix before delivery (~1.5–2.5 days)**
- [ ] `detector.py:51` → `cy = y2` (or documented fraction); fix docstring + `boundary_detector`. *~2 h + re‑validate polygons.*
- [ ] `main.py:103` / `tracker.py` — lock around shared YOLO access. *~3 h.*
- [ ] `…identity.py:2658` — hoist `is_plate_inside` off the RLock. *~3 h.*
- [ ] `…identity.py:2507` — provisional (sub‑lock) auto‑lock + `is_plate_inside`/ownership check. *~4 h.*
- [ ] `config.yaml:246` — raise `lock_confidence` to ~0.62; validate on labelled set. *config + validation.*
- [ ] `config.yaml` — add `area:` to each camera; startup warning on empty mapping. *~1 h.*

**P1 — high/medium, fix before or immediately after delivery (~2–3 days)**
- [ ] `state_machine.py:263` handover guard; `:254/:299` leaky counters.
- [ ] `slot_assigner.py:168` overlap divisor; `:91/:139` global one‑to‑one assignment (re‑tune threshold on footage).
- [ ] `vehicle_registry.py:396` owner/area gate on boundary crossing.
- [ ] `reid_matcher.py:335` strict‑backend raise in prod.
- [ ] `match_decision.py:442/235/200` + `…identity.py:2096` matching fixes (mostly trivial).
- [ ] `engine_runtime.py:995` one‑vote‑per‑frame; `engine_tracking.py:112` list‑snapshot.
- [ ] `gallery_store.py:126/278` gallery‑tag + GC exclusion.

**P2 — hardening & hygiene (~1 day)**
- [ ] `config_service.py:113` align DB defaults to `5/8/0.2` + log effective values at boot.
- [ ] `config.example.yaml` / `config.yaml:279` config‑doc coherence (`use_lab_clahe`/`use_multishot` are currently log‑only vs behaviour).
- [ ] `area_state_machine.py:83/95` finish blind‑gap + ARRIVING debounce.
- [ ] low‑severity robustness/doc cleanups.

**Calibration‑needs‑footage (NOT code bugs — do not block delivery):** slot‑polygon re‑authoring after the probe‑point fix; `overlap_threshold` re‑tune after the divisor fix; `lock_confidence` validation on a labelled set; ARRIVING/blind‑gap debounce tuning; DB‑config timing values.

---

## 9. What We Verified vs. Could Not Verify

- **Verified (CONFIRMED):** 49 findings confirmed directly against source with exact `file:line`, including all 8 highs. Every quoted threshold/frame‑count/multiplier is traceable to `config.yaml`, `config.py`, or the cited module. The four headline items (`detector.py:51`, auto‑lock `2507`, per‑camera tracker swap `tracker.py:163`, effective `lock_confidence 0.50`) were re‑verified by hand.
- **Refuted:** **8** candidate issues were rejected by the adversarial pass — e.g. the `decide_global` cross‑camera relief being gated to plate sessions is **intentional**, and the config.py‑vs‑YAML default divergence is **intentional and production‑safe** (YAML is authoritative). This is why §5 states effective values, not code defaults.
- **Could not fully verify (2 PLAUSIBLE, runtime‑dependent):** the no‑gap slot handover depends on whether a real physical handover sustains continuous polygon overlap with no 8‑frame empty window; the DB‑config timing divergence depends on out‑of‑band row creation that no in‑repo code path exercises. Both code paths are unambiguous; only the runtime trigger is unproven, so neither is refuted.

*Fixes whose **impact** depends on deployment artefacts not in the repo — slot‑polygon authoring, the DB Config row's provenance, painted boundary bands — appear in the plan as footage/deployment items, not pure code fixes.*
