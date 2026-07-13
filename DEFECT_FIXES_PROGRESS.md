# Matching Pipeline Defect Report Fixes — Progress Tracking

**Source:** `C:\Users\moham\Desktop\matching_pipeline_defects.md`

**Recommended order:** D8b → D15+D14 → D1+D3+D2 → D9 → D13 → D4+D5+D6+D7 → D8+D10+D11+D12+D16-D18 → D19-D25

---

## POINT 1: D8b — Config DB/YAML Reconciliation & Annotation ✅ IN PROGRESS

### What Changed

1. **config.yaml** — Added clear annotations:
   - Marked DB-OWNED blocks: `detector`, `tracker`, `state_machine`, `assigner`
   - Marked YAML-AUTHORITATIVE: `matching` block
   - Added comments about known divergences:
     - `detector.confidence`: YAML 0.25 → DB 0.35
     - `detector.imgsz`: YAML 640 → DB 320 (fixed backwards comment)
     - `state_machine.confirm_enter_frames`: YAML 5 → DB 1 (CRITICAL: single-frame debounce!)
     - `state_machine.confirm_leave_frames`: YAML 8 → DB 3
     - `assigner.overlap_threshold`: YAML 0.2 → DB 0.3

2. **src/services/config_service.py** — Added logging in `sync_app_config_from_db()`:
   - Now logs effective detector config (imgsz, confidence, model_path)
   - Now logs effective state_machine config (confirm_enter/leave_frames)
   - Now logs effective assigner config (overlap_threshold)
   - **This makes it visible at startup what config actually took effect vs YAML**

### What's Next

**ACTION REQUIRED:** Confirm whether the DB values are intentional or accidental drift:
- Is `confirm_enter_frames = 1` intentional? (YAML says 5 — this is a single-frame occupancy debounce vs 5-frame)
- If accidental: update DB to match YAML (5 and 8)
- If intentional: document in DB/code why single-frame debounce is preferred

### How to Test

1. Start the engine:
   ```bash
   python main.py --api --show
   ```

2. Check startup logs for:
   ```
   [DB] Effective detector config: imgsz=..., confidence=..., model_path=...
   [DB] Effective state_machine config: confirm_enter_frames=..., confirm_leave_frames=...
   [DB] Effective assigner config: overlap_threshold=...
   ```

3. Verify these match your actual DB row (compare to the config table in your MSSQL)

4. **If any divergence found**, update `config.yaml` comment and investigate root cause.

---

## POINT 2: D1 — CAM-23 Park_Entry FIFO Gate Fix ✅ DONE

### What Changed

**1. src/core/engine/engine_tracking.py:604 — Primary-car selection**
   - Added `_select_primary_zone_detection()` call to pick the best in-zone car
   - Loop now checks `is_primary` before calling `bind_next_pending_anpr_to_candidate()`
   - Non-primary cars still get candidates (snapshots updated) but **never bind plates**
   - Prevents tailgating: when 2 cars in zone, only the one with best overlap can grab ANPR

**2. src/vehicle_registry/vehicle_registry_identity.py:764 — Bind-eligibility on entered_at**
   - Added timeout check using `candidate.entered_at` (immutable, set once at creation)
   - If candidate age > `PENDING_ANPR_BIND_TTL_SECONDS` (10s), skip bind
   - Replaces reliance on `last_seen_at` (refreshed every frame, useless for lingering cars)
   - Prevents swap: car queued in zone can no longer grab next car's plate

### How to Test

1. **Visual test with cameras:** 
   - Send 2 cars into Park_Entry zone within 10s
   - First car (best overlap) should bind the ANPR plate
   - Second car should stay plateless until it gets its own ANPR event

2. **Code inspection:**
   - Line 613 in engine_tracking.py: primary-car filtering is active
   - Line 791 in vehicle_registry_identity.py: entered_at timeout enforced

3. **Log indicators** (set CAM-23 debug=INFO):
   - `[PARK_ENTRY] Track X skipped for binding: not primary. Primary is track Y` ← secondary car rejected
   - `[PARK_ENTRY] Bound ANPR event...` ← only primary car succeeds

---

## POINT 3: D3 — Lingering car steals the next car's plate ✅ DONE (2nd attempt)

> **The first attempt was reverted and this section used to describe it as shipped.**
> It claimed an `entered_at` expiry sweep, an `expire_park_entry_candidate()` call on
> zone exit, and an `entered_at` bind TTL. All three were rolled back in `b3ef313`
> ("un-break Park_Entry binding damaged by D1/D2/D3 over-reach") because they keyed on
> the candidate's **absolute age** and so force-expired cars that were legitimately
> dwelling at the barrier — they never bound at all. For a while afterwards the code
> carried a docstring asserting an `entered_at` check that did not exist, and
> `expire_park_entry_candidate()` sat unreferenced. Both are now corrected.

### The rule that actually shipped

A candidate may only bind a plate that was **read at-or-after it entered the zone**.
The ANPR read happens at the gate, upstream of the CAM-23 polygon, so the car that
triggered a read reaches the zone *after* it. A car already sitting in the zone when
the read landed cannot be the car that was read.

**1. `vehicle_registry.py` — `PARK_ENTRY_LINGER_GRACE_SECONDS = 5`**
Relative, not absolute. A car may dwell in the zone for minutes; only its entry time
*versus the read* matters. The grace absorbs ANPR POST latency, because
`event.timestamp` is when the event was **received**, not when the plate was read.

**2. `vehicle_registry_identity.py` — `bind_next_pending_anpr_to_candidate()`**
```python
lingered = (pending.timestamp - candidate.entered_at).total_seconds()
if lingered > self.PARK_ENTRY_LINGER_GRACE_SECONDS:
    continue   # not this car's plate — but keep looking for an OLDER event
```
`continue`, not `return`: a car whose own read merely arrived late can still bind its
own (older) pending event.

**3. `engine_tracking.py` — `_process_park_entry_zone()` binds by walking the ranked cars**
The guard alone was not enough. The bind requires `status == "open"` and only the
single best-ranked car ever attempted it — and a stationary lingerer scores **high** on
overlap/depth/area, so it wins the primary slot. It would therefore either take the
arriving car's plate or, once refused, **block every car behind it from ever binding**.
The bind now walks `_rank_zone_detections()` best-first and stops at the first car the
registry accepts, so an ineligible car is skipped instead of blocking the gate. The
solo-car fallback from `b3ef313` is preserved.

### Interaction with D1

- D1: among *eligible* cars, the plate goes to the primary — not to whoever the tracker listed first.
- D3: a lingerer is not eligible, and cannot block the car that is.

### Tests

`tests/test_park_entry_linger_d3.py` (7). The three D3 cases fail without the guard
(reproducing the steal: *"lingering primary stole the arriving car's plate"*); the rest
are regression guards for the `b3ef313` revert — a dwelling car still binds its own
plate, a solo car still binds when the ranking abstains, and a late ANPR POST inside the
grace still binds.

### Still open

`expire_park_entry_candidate()` (`vehicle_registry_core.py`) remains **dead code** and is
still labelled D3. It is not part of this fix — calling it on zone exit is precisely what
`b3ef313` had to revert (a one-frame ByteTrack dropout would kill a candidate about to
bind). Delete it or relabel it; do not wire it back up.

---

## POINT 4: D2 — Gallery Admission Guard ✅ DONE

### What Changed

**vehicle_registry_identity.py:573 — Added identity-similarity floor before gallery save**

```python
# Load established refs from the durable gallery
established = store.load_vectors(plate)  # excludes gate_only refs

# If there are existing refs, check that new crop matches the plate's identity
if identity_floor > 0.0 and established:
    id_sim = max(self.reid_matcher.compute_similarity(feature, ev) for ev in established)
    if id_sim < identity_floor:
        logger.warning("[gallery] Rejected foreign Park_Entry seed for %s (id_sim=%.3f < %.2f)", ...)
        return False

store.save_ref(plate, crop, feature, quality=quality, camera_id=camera_id, gate_only=False)
```

**Key points:**
- Mirrors the check in `_seed_plate_gallery_reference` (used by CAM-03)
- Fails **open** on empty gallery (first crop has nothing to compare against)
- Only rejects if `gallery_min_identity_similarity > 0.0` in config
- Prevents mis-binds from poisoning the durable gallery permanently

### Three-Layer Anti-Swap Defense

| Layer | Defense | Defect |
|-------|---------|--------|
| **Binding** | Only primary car can bind | D1 |
| **Expiry** | Only recently-entered candidates can bind | D3 |
| **Gallery** | Rejected crops must match plate identity | D2 |

Combined: Wrong plate cannot become permanent

---

## Status Summary

> **Verify against the code before trusting a ✅ here.** This file marked D3 done for a
> month while the tree contained no such fix (see POINT 3). A row is evidence of intent,
> not of a shipped change.

| Defect | Severity | Status |
|--------|----------|--------|
| **D8b** | High | ✅ DONE (effective-config logging uses `print`, not `logger`) |
| **D1** | Critical | ✅ DONE (`_select_primary_zone_detection`, + solo fallback from `b3ef313`) |
| **D3** | Critical | ✅ DONE (2nd attempt — linger guard + ranked bind walk; 1st attempt was reverted) |
| **D2** | Critical | ✅ DONE (seed + reattach + CAM-03 paths) |
| D15 | Medium | ✅ DONE (`tools/eval_identity_disjoint.py`; honest cross-view rank-1 0.736 / mAP 0.564) |
| D9 | Medium | ✅ DONE, log-only (`gallery_neighbour_clearance_enforce: false` — collect the histogram, then gate) |
| D14 | Medium | ❌ NOT DONE (`bench_yolo.py` measures throughput only; no small-car recall benchmark) |
| D5 | Low | ❌ NOT DONE (`b1_cross_camera` is still dead config — `decide_b1` takes a `cross_camera` flag and never reads it) |
| D6 | Medium | ⚠️ PARTIAL (imgsz now resolved from the export's `metadata.yaml`; still 320, no `detector_overrides`, no 512 IR) |
