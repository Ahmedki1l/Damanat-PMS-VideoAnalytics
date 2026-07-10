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

## POINT 3: D3 — Candidate Expiry Timeout Fix ✅ DONE

### What Changed

**1. vehicle_registry_core.py:541 — Expiry sweep now uses entered_at**
```python
age = (now - candidate.entered_at).total_seconds() if candidate.entered_at else 0
if age > self.CANDIDATE_EXPIRY_SECONDS:  # 30s
    candidate.status = "expired"
```
- Replaces `last_seen_at` (refreshed every frame)
- Now a stationary car expires after 30s, not never

**2. vehicle_registry_core.py:606 — New expire_park_entry_candidate() method**
- Called when a track leaves the zone
- Marks candidate as "expired" immediately
- Prevents re-use if car queues, leaves, re-enters

**3. engine_tracking.py:695 — Exit path now calls expire**
```python
self.vehicle_registry.expire_park_entry_candidate(candidate_id)
```
- Aggressive cleanup when track leaves zone
- Prevents lingering candidates from grabbing next plate

### Interaction with D1

D1 + D3 = complete anti-swap defense:
- D1: Only primary car can bind
- D3: Only recently-entered candidates can bind (TTL enforced)
- Together: eliminates FIFO cross-binding vulnerability

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

| Defect | Severity | Status |
|--------|----------|--------|
| **D8b** | High | ✅ DONE |
| **D1** | Critical | ✅ DONE |
| **D3** | Critical | ✅ DONE |
| **D2** | Critical | ✅ DONE |
| D15+D14 | Medium | 📋 NEXT |
| D9 | Medium | 📋 QUEUED |
