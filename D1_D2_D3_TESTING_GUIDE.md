# D1+D2+D3 Critical Gate Fixes — Testing Guide

## Summary of Changes

Three surgical fixes to prevent tailgating and gallery poisoning at CAM-23 Park_Entry gate.

### D1 — Primary-Car Selection (Tailgating Fix)
**File:** `src/core/engine/engine_tracking.py:604`
- Only the car with best zone overlap can bind ANPR plates
- Other in-zone cars get candidates (snapshots tracked) but never grab plates
- **Prevents:** two cars entering simultaneously → wrong car gets plate

### D3 — Candidate Expiry Timeout (Lingering-Car Fix)  
**Files:** 
- `src/vehicle_registry/vehicle_registry_core.py:541` — expiry sweep uses `entered_at` not `last_seen_at`
- `src/vehicle_registry/vehicle_registry_core.py:606` — new `expire_park_entry_candidate()` method
- `src/core/engine/engine_tracking.py:695` — exit path calls expire

- Stationary cars now expire after 30s (not never)
- Candidates marked expired when track leaves zone
- **Prevents:** car queues in zone → grabs next car's plate

### D2 — Gallery Admission Guard (Contamination Fix)
**File:** `src/vehicle_registry/vehicle_registry_identity.py:573`
- Added identity-similarity check before saving CAM-23 crop to gallery
- Rejects crops that don't match the plate's existing identity
- Only applies when gallery already has established refs (fails open on first crop)
- **Prevents:** mis-bind at gate → permanent wrong identity in gallery

---

## Testing Checklist

### 1. Pre-Test Setup
```bash
# Confirm DB values (should be from earlier):
# confirm_enter_frames = 5 (was 1)
# confirm_leave_frames = 10 (was 3)
```

### 2. Single-Car Baseline (Sanity Check)
- [ ] One car enters CAM-23 Park_Entry zone
- [ ] ANPR reads plate at gate
- [ ] Car binds to plate within 10s
- [ ] CAM-23 top-view crop saved to gallery
- [ ] Car moves to CAM-03, matches with high ReID score (0.8+)
- [ ] **Expected:** Clean single entry, no errors

### 3. Tailgating Test (D1)
- [ ] **Two cars** simultaneously in CAM-23 Park_Entry zone (within 10s)
- [ ] ANPR fires once, reading one plate
- [ ] **Expected:** Only the car with **best zone overlap** binds the plate
- [ ] Other car stays plateless (gets candidate but no binding)
- [ ] **Check logs** for: `[PARK_ENTRY] Track X skipped for binding: not primary. Primary is track Y`
- [ ] ✅ **PASS:** Right car got the plate; wrong car rejected

### 4. Lingering-Car Test (D3)
- [ ] Car enters CAM-23 zone, ANPR fires
- [ ] Car **stays in zone** (queue at barrier) for >10s
- [ ] Another car enters separately, new ANPR plate is read
- [ ] **Expected:** First car does NOT steal second car's plate (expired candidate)
- [ ] **Check logs** for: `[PARK_ENTRY] Track X entered too long ago; it did not just trigger ANPR`
- [ ] ✅ **PASS:** Second car got its own plate; first car didn't steal it

### 5. Gallery Contamination Test (D2)
- [ ] Trigger a mis-bind scenario (if possible: two cars, D1 pick wrong one)
- [ ] Check gallery folder for that plate: `snapshots/gallery/<plate>/`
- [ ] **Expected (with D2):** If the crop would poison identity, it's rejected
- [ ] **Check logs** for: `[gallery] Rejected foreign Park_Entry seed for <plate> (id_sim=...)`
- [ ] ✅ **PASS:** Bad crop not saved to gallery; plate identity stays clean

### 6. Multi-Car Flow (Integration)
- [ ] Send 3-4 cars through the gate in sequence
- [ ] Each should bind to its own ANPR plate
- [ ] All should move through CAM-03 confirmation
- [ ] All should park in slots with correct plates
- [ ] **Expected:** No mis-bindings, no plate swaps
- [ ] ✅ **PASS:** Clean multi-car flow

---

## Key Indicators

### Good Signs ✅
- `[PARK_ENTRY] Bound ANPR event ... to candidate ...` — binding succeeded
- `[GLOBAL] matched session ... (score=0.9+)` — ReID confident
- No `[PARK_ENTRY] Track X skipped for binding` messages (only one car in zone)
- Gallery folders created with correct plate names
- No `[gallery] Rejected foreign` messages (no contamination)

### Warning Signs ⚠️
- `[PARK_ENTRY] Track X skipped for binding: not primary` when only one car in zone → D1 working
- Multiple `[PARK_ENTRY] Bound` messages in quick succession → separate cars, expected
- `[GLOBAL] ambiguous match (best=X runner_up=Y)` → voting abstained, check margin

### Bad Signs ❌
- Car bound to wrong plate number → D1 not working
- Lingering car grabbed next car's plate → D3 not working
- Gallery folder has wrong car's image → D2 not working
- Python exceptions in vehicle_registry

---

## Rollback (if needed)
```bash
# Revert to pre-fix state:
git diff src/core/engine/engine_tracking.py
git diff src/vehicle_registry/vehicle_registry_identity.py
git diff src/vehicle_registry/vehicle_registry_core.py

# To rollback one file:
git checkout src/core/engine/engine_tracking.py
```

---

## Notes for Next Phase

- D1+D2+D3 fix the **gate binding** problem
- D9 (occlusion detection) fixes the **input quality** problem (prevents garbage crops)
- D4+D5+D6+D7 fix **config divergence** problems
- D15+D14 provide **benchmarks** to validate everything works

All critical safety fixes are now in place. Test thoroughly before deploying to production.
